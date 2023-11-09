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


def get_robotcar_dicts(label_path, split_file_path):
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
        # hard codeded the data path of where you store the processed radar images
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

def instances_to_json(instances, img_id):
        num_instance = len(instances)
        if num_instance == 0:
            return []

        boxes = instances.pred_boxes.tensor.numpy()
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        results = []
        for k in range(num_instance):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }

            results.append(result)
        return results

def draw_rotated_box_with_label(
         rotated_box, image, label, score
    ):  
        
        cx, cy = rotated_box[0], rotated_box[1]
        width, height = rotated_box[2], rotated_box[3]
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
        cx, cy = rotated_box[0], rotated_box[1]
        T = np.array([[cx], [cy]])
        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(int)
       
        image = cv2.drawContours(image, [np.array(points.T)], -1, (0, 255, 0), 1)
        return {
                "sample_token": label,
                "points": points.T.tolist(),
                "name": "car",
                "scores": score
            }

def process(inputs, outputs):
    for input, output in zip(inputs, outputs):
        prediction = {"image_id": input["image_id"]}

        if "instances" in output:
            instances = output["instances"].to('cpu')
            prediction["instances"] = instances_to_json(instances, input["image_id"])

        # comment out the following lines if you want to get the proposals
        # if "proposals" in output:
        #     prediction["proposals"] = output["proposals"].to('cpu')

        return prediction
       

def setup(args):
    cfg = get_cfg()
    get_mvdnet_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    scaleVizValue = False

    only_eval = True

    model = build_model(cfg)
    model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    dataset_name = cfg.DATASETS.TEST[0]
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=RobotCarMapper(cfg, "eval"))
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', (800, 800))
    
    # initialize the prediction dictionary to store the predict result
    predictions = []
    for idx, inputs in tqdm(enumerate(data_loader), total=len(data_loader)):
        
        with torch.no_grad():
            # Run the model to get proposals
            radar_images = inputs[0]['radar_intensity']
            lidar_images = inputs[0]['lidar_intensity']
            
            # disable the lidar input since we want to only evaluate the radar input result
            inputs[0]['lidar_intensity'] = [torch.zeros_like(data) for data in inputs[0]['lidar_intensity']]
            inputs[0]['lidar_occupancy'] = [torch.zeros_like(data) for data in inputs[0]['lidar_occupancy']]
            
            proposals = model(inputs)

        predict_result = process(inputs, proposals)
        predictions.append(predict_result)

 

        if only_eval:
            pass

        else:
            dataset_dicts[idx]['file_name'] = os.path.join(dataset_dicts[idx]['data_root'], 'radar', dataset_dicts[idx]['timestamp'] + ".jpg")
            
            image = dataset_dicts[idx]["file_name"]

            im = np.zeros((320, 320, 3), dtype=np.uint8)

            radar_image = (radar_images[0].numpy() * 255).astype(np.uint8).squeeze()
            radar_image = np.expand_dims(radar_image, axis=2)
            radar_image = np.repeat(radar_image, repeats=3, axis=2)
            im = radar_image

            v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=2.0)

            predict = proposals[0]['instances'].to('cpu')

            boxes = predict.pred_boxes if predict.has("pred_boxes") else None
            scores = predict.scores if predict.has("scores") else None
            classes = predict.pred_classes if predict.has("pred_classes") else None

            scores = scores.numpy()
            scores = [float(score) for score in scores ]
            boxes =  boxes.tensor.numpy()
            num_instances = len(boxes)
            
            if boxes is not None:
                areas = boxes[:, 2] * boxes[:, 3]

            sorted_idxs = np.argsort(-areas).tolist()

            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs]

            res_dict = []
            ann = np.zeros((320, 320, 3), dtype=np.uint8)
            for i in range(num_instances):
                res = draw_rotated_box_with_label(boxes[i], ann, image.split("/")[-1][:-4], score=scores[i])
                res_dict.append(res)

            out = v.draw_instance_predictions(proposals[0]['instances'].to('cpu'))
            result_image = out.get_image()[:, :, ::-1]

            result_image = cv2.resize(result_image, (800, 800))
            cv2.imshow('img', result_image)
            cv2.waitKey(20)

    # save the prediction result to json file for evaluation
    parsed_path = "/data/Codespace/MVDNet/output/parse_result_mvd"
    json_file_path = os.path.join(parsed_path, "coco_mvd_radaronly_all.json")
    with open(json_file_path, "w") as json_file:
        json.dump(predictions, json_file, indent=4)   

    



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
