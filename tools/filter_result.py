import cv2, os
import torch
import numpy as np
import math
import json
from collections import defaultdict
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser, default_setup
from detectron2.evaluation import inference_on_dataset
from detectron2.structures import BoxMode

import mvdnet.modeling
from mvdnet.data import RobotCarMapper
from mvdnet.evaluation import RobotCarEvaluator
from mvdnet.config import get_mvdnet_cfg_defaults
from tqdm import tqdm
from map_evaluation import Box, get_ious, group_by_key, wrap_in_box



def get_robotcar_dicts(label_path, split_file_path) -> list:
    with open(split_file_path, 'r') as f:
        image_list = [image_info.strip().split(' ') for image_info in f.readlines()]
    
    dataset_dicts = []
    object_index = 1
    for image_info in image_list:
        image_index = int(image_info[0])
        label_name = os.path.join(label_path, image_info[1]+'.txt')
        with open(label_name, 'r') as f:
            label_list = [label_info.strip().split(' ') for label_info in f.readlines()]

        record = {}
        record['data_root'] = "/data/oxford/2019-01-10-11-46-21-radar-oxford-10k/processed_center_free/radar"
        record['timestamp'] = image_info[1]
        record['image_id'] = image_index

        cars = []
        for label_info in label_list:
            car_index = float(label_info[1])
            xc = float(label_info[2])
            yc = float(label_info[3])
            width = float(label_info[4])
            height = float(label_info[5])
            yaw = float(label_info[6])
            assert yaw >= -180 and yaw < 180, 'rotation angles of labels should be within [-180,180).'
            car_instance = {
                'bbox': [xc, yc, width, height, yaw],
                'bbox_mode': BoxMode.XYWHA_ABS,
                'category_id': 0,
                'iscrowd': 0,
                'id': object_index,
                'car_id': car_index
            }
            cars.append(car_instance)
            object_index += 1
        record['annotations'] = cars
        
        dataset_dicts.append(record)

    return dataset_dicts 


def filter_annotation(gt_file_path, predict_file_path) -> None:
    """
    Use to filter those predictions overlap with the ground truth larger than centain threshold

    ex: Use static ground truth to filter the true positive static prediction
    """
    
    split_path = "/data/Codespace/MVDNet/data/RobotCar/ImageSets/eval.txt"

    # convert the label_2d to coco format
    gt = get_robotcar_dicts(gt_file_path, split_path)  
    dataset_coco_prediction = json.load(open(predict_file_path, "r"))
    
    # parse the result into custom format to do the filtering 
    dataset_gt_custom = coco2custom(gt)
    dataset_prediction_custom = coco2custom(dataset_coco_prediction)

    # filter the prediction
    filtered_coco_prediction = filter(dataset_gt_custom, dataset_prediction_custom)
    
    # merge the filtered prediction into the original coco format
    for item_ori in dataset_coco_prediction:
        item_ori['instances'] = []
        for item in filtered_coco_prediction:
            if item['image_id'] == item_ori['image_id']:
                item_ori['instances'].append(item)
    
    # save the filtered result into json file 
    parsed_path = "/data/Codespace/MVDNet/output/parse_result_mvd"
    json_file_path = os.path.join(parsed_path, "coco_mvd_radaronly_all_filtered.json")
    with open(json_file_path, "w") as json_file:
        json.dump(dataset_coco_prediction, json_file, indent=4)

        
def coco2custom(dataset_prediction_coco) -> list:
    """
    Convert the coco format prediction into custom format
    
    return: converted custom format prediction
    """
    dataset_custom = []
 
    for predict_dict in dataset_prediction_coco:
        # this part is used to parse the saved prediction result
        try:
            for item in predict_dict['instances']:
                dataset_custom.append(parse_rotated_box(item['bbox'], predict_dict['image_id'], item['score'], scale=1, center="image"))

        # this part is used to parse the ground truth annotations 
        except:
            for item in predict_dict['annotations']:
                dataset_custom.append(parse_rotated_box(item['bbox'], predict_dict['image_id'], 0, scale=0.2, center="sensor"))
                # print(dataset_custom)
            
    
    return dataset_custom


def filter(gt, predictions, viz=False) -> list:
    """
    Use given prediction bounding box to calculate the overlap area with the all ground truth bounding box
    If exceed the threshold, then filter out the prediction bounding box
    
    return: the filtered prediction bounding box
    """

    # remap the ground truth bounding box by sample_token
    image_gts = group_by_key(gt, "sample_token")
    image_gts = wrap_in_box(image_gts)
    
    filtered_prediction = []
    filtered_coco_prediction = []
    
    for prediction_index, prediction in tqdm(enumerate(predictions), total=len(predictions)):
        
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        predicted_box = Box(**prediction)
        
        sample_token = prediction["sample_token"]
        
        sample_gt_checked = {
        sample_token: np.zeros(len(boxes)) for sample_token, boxes in image_gts.items()
        }

        try:
            gt_boxes = image_gts[sample_token]  # gt_boxes per sample
            gt_checked = sample_gt_checked[sample_token]  # gt flags per sample
            
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            overlaps = get_ious(gt_boxes, predicted_box)
            max_overlap = np.max(overlaps)

            if max_overlap > 0.3:

                continue
            jmax = np.argmax(overlaps)


        prediction_coco = {
            "image_id": prediction["sample_token"],
            "category_id": 0,
            "bbox": prediction["bbox"],
            "score": prediction["scores"],
        }
        # use to save the filtered prediction
        filtered_coco_prediction.append(prediction_coco)

        # use to visualize the filtered prediction
        filtered_prediction.append(prediction)
        
        # draw the ground truth bounding box
        for item in gt:
            if item["sample_token"] == prediction["sample_token"]:
                points = np.array(item["points"]).astype(int)
                if viz:
                    image = cv2.drawContours(image, [points], -1, (0, 255, 0), 1)

        # draw the filtered prediction bounding box
        for item in filtered_prediction:
            if item["sample_token"] == prediction["sample_token"]:
                points = np.array(item["points"]).astype(int)
                if viz:
                    image = cv2.drawContours(image, [points], -1, (255, 0, 0), 1)
        if viz:
            cv2.imshow("img", image)
            cv2.waitKey(20)

    return filtered_coco_prediction

def parse_rotated_box(rotated_box, frame_ts, score, scale=0.2, center="image") -> dict:
    """
    Convert the rotated box [x_center, y_center, width, height, angle] with score to  

    {
        "sample_token": frame_ts, \n
        "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]], \n
        "name": "car", \n
        "scores": score \n
    } 

    scale : the pixel resolution of the image 
    center : the coordinate system of the rotated box
    """
    def _topixel(meter):
        return int(meter / scale)
    cx, cy = _topixel(rotated_box[0]), _topixel(rotated_box[1])
    width, height = _topixel(rotated_box[2]), _topixel(rotated_box[3])
    angle = rotated_box[4]
    cx = cx - (width // 2)
    cy = cy - (height // 2)
    
    theta = np.deg2rad(-angle)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    points = np.array(
        [
            [cx, cy],
            [cx + width, cy],
            [cx + width, cy + height],
            [cx, cy + height],
        ]
    ).T

    cx, cy = _topixel(rotated_box[0]), _topixel(rotated_box[1])
    T = np.array([[cx], [cy]])

    # the points are in the image coordinate
    if center == "image":
        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(int)

    # the points are in the sensor coordinate
    elif center == "sensor":
        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(int)
        points += 160
        # print(points)

    return {
            "sample_token": frame_ts,
            "points": points.T.tolist(),
            "bbox": rotated_box,
            "name": "car",
            "scores": score
            }


def vis_annotation(file_path):
    """
    Used to visualize the annotation in the RobotCar dataset
    
    file_path: the path of the annotation file
    """
    gt_list = get_robotcar_dicts(file_path, "/data/Codespace/MVDNet/data/RobotCar/ImageSets/eval.txt")

    for gt in gt_list:
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        print(gt["image_id"])
        radar_filename = os.path.join(gt["data_root"], gt["timestamp"] + ".jpg")
        radar_img = cv2.imread(radar_filename, 1)

        
        for data in gt["annotations"]:
            item = parse_rotated_box(data["bbox"], gt["timestamp"], 0, scale=0.2, center="sensor")
            
            image = cv2.drawContours(image, [np.array(item["points"]).astype(int)], -1, (0, 255, 0), 1)
        final = cv2.addWeighted(radar_img, 1, image, 0.5, 0)
        cv2.imshow("img", final)
        cv2.waitKey(0)



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    filter_annotation("/data/Codespace/MVDNet/data/RobotCar/object/label_cached/label2d_static", "/data/Codespace/MVDNet/output/parse_result_mvd/coco_mvd_radaronly_all.json")
    
    #vis_annotation("/data/Codespace/MVDNet/data/RobotCar/object/label_cached/label2d_moving")
