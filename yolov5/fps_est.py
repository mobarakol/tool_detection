# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (MacOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import math
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (
    LOGGER,
    box_iou,
    check_dataset,
    check_img_size,
    check_requirements,
    check_yaml,
    coco80_to_coco91_class,
    colorstr,
    increment_path,
    non_max_suppression,
    print_args,
    scale_coords,
    xywh2xyxy,
    xyxy2xywh,
)
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def clip_boxes(boxes, shape=[256, 320]):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    # print(type(boxes))
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    return boxes.T


def scale_bbox(coordinates, feature_spatial_size, height=256, width=320):
    """
    Given the bounding box coordinates, feature map size, and input image size,
    scale the bounding box coordinates to fit the feature map.

    :param coordinates: the bounding box coordinates
    :param feature_spatial_size: The spatial size of the feature map
    :param height: the height of the image, defaults to 256 (optional)
    :param width: the width of the image, defaults to 320 (optional)
    :return: The bounding box coordinates are being returned.
    """
    x1 = coordinates[0]
    y1 = coordinates[1]
    x2 = coordinates[2]
    y2 = coordinates[3]

    feature_map_x = feature_spatial_size[1]
    feature_map_y = feature_spatial_size[0]

    x1 = x1 * feature_map_x / width
    y1 = y1 * feature_map_y / height
    x2 = x2 * feature_map_x / width
    y2 = y2 * feature_map_y / height

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    if x2 - x1 == 0:
        if x2 != feature_map_x:
            x2 = x1 + 1
        else:
            x1 = x1 - 1

    if y2 - y1 == 0:
        if y2 != feature_map_y:
            y2 = y1 + 1
        else:
            y1 = y1 - 1

    return [int(x1), int(y1), int(x2), int(y2)]


def get_feature(
    pred_box, cached_features, p_cls, instance_counter, paths, threshold=[5120, 20480]
):
    """
    :param pred_box: the predicted bounding box
    :param cached_features: A dictionary of cached features
    :param p_cls: the idx of the predicted class
    :param instance_counter: the number of instance of the class
    :param paths: a list of paths to the images
    :param threshold: [5120, 20480]
    """
    x1 = pred_box[0]
    y1 = pred_box[1]
    x2 = pred_box[2]
    y2 = pred_box[3]

    bbox_size = (x2 - x1) * (y2 - y1)
    if bbox_size < threshold[0]:
        feature_map = cached_features["layer_16"]  # [0]: layer 2
    elif bbox_size < threshold[1]:
        feature_map = cached_features["layer_12"]
    else:
        feature_map = cached_features["layer_9"]

    ## To check if have 'nan' value in the original features
    feature_sum = np.sum(feature_map.data.cpu().float().numpy())
    array_has_nan = np.isnan(feature_sum)
    if array_has_nan:
        print("**********", (x2 - x1) * (y2 - y1))

    feature_spatial_size = feature_map.shape[2:]
    cropped_bbox = scale_bbox(pred_box, feature_spatial_size)

    cropped_feature = feature_map[
        :, :, cropped_bbox[1] : cropped_bbox[3], cropped_bbox[0] : cropped_bbox[2]
    ]

    # print(f"Cropped Feature Shape: {cropped_feature.shape}, Original Feature Shape:{feature_map.shape}, Cropped BBox: {cropped_bbox}, Original BBox: {pred_box}")

    cropped_feature = F.adaptive_avg_pool2d(cropped_feature, [1, 1]).permute(0, 2, 3, 1)

    if cropped_feature.shape[-1] != 512:
        cropped_feature = F.adaptive_avg_pool2d(cropped_feature, [1, 512]).permute(
            0, 1, 2, 3
        )

    cropped_feature = torch.flatten(cropped_feature)

    ## To check if have 'nan' value in the extracted features due to small bbox area
    feature_sum = np.sum(cropped_feature.data.cpu().float().numpy())
    array_has_nan = np.isnan(feature_sum)
    if array_has_nan:
        print(
            "!!!!!!!!!!",
            (x2 - x1) * (y2 - y1),
            cropped_bbox,
            (cropped_bbox[2] - cropped_bbox[0]) * (cropped_bbox[3] - cropped_bbox[1]),
            cropped_feature.shape,
        )

    cropped_feature = cropped_feature.data.cpu().float().numpy()
    return cropped_feature

def get_num_preds(
    image, predictions, cached_features, class_names, paths, max_size=1920
):

    _, _, h, w = image.shape
    x = y = 0

    scale = max_size / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()

    if len(predictions) > 0:
        pi = predictions[predictions[:, 0] == 0]
        pred_boxes = xywh2xyxy(pi[:, 2:6]).T
        pred_classes = pi[:, 1].astype("int")
        pred_labels = pi.shape[1] == 6
        pred_conf = None if pred_labels else pi[:, 6]

        if pred_boxes.shape[1]:
            if pred_boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                pred_boxes[[0, 2]] *= w  # scale to pixels
                pred_boxes[[1, 3]] *= h

            elif scale < 1:  # absolute coords need scale if image scales
                pred_boxes *= scale

        pred_boxes[[0, 2]] += x
        pred_boxes[[1, 3]] += y

        pred_boxes = clip_boxes(pred_boxes.T)

        conf_thr = 0.3
        all_bbox_features = []
        tissue_bboxes = []
        instance_counter = [0] * len(class_names)

        for k, pred_box in enumerate(pred_boxes.T.tolist()):
            if pred_conf[k] > conf_thr:
                p_cls = pred_classes[k]

                if p_cls != 0:
                    instance_counter[p_cls] += 1
                    cropped_bbox_feats = get_feature(
                        pred_box, cached_features, p_cls, instance_counter, paths
                    )
                    all_bbox_features.append(cropped_bbox_feats)

                else:
                    tissue_cls = p_cls
                    tissue_bboxes.append(pred_box)

        if len(tissue_bboxes) > 0:
            x1 = min(x[0] for x in tissue_bboxes)
            y1 = min(x[1] for x in tissue_bboxes)
            x2 = max(x[2] for x in tissue_bboxes)
            y2 = max(x[3] for x in tissue_bboxes)
            pred_box = [x1, y1, x2, y2]
            instance_counter[tissue_cls] += 1
            cropped_bbox_feats = get_feature(pred_box, cached_features, tissue_cls, instance_counter, paths)
            all_bbox_features.append(cropped_bbox_feats)
            
        all_bbox_features = np.array(all_bbox_features)   
        num_bboxes = all_bbox_features.shape[0] 
                
    else:
        return None

    return num_bboxes

def clip_boxes(boxes, shape=[256, 320]):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    # print(type(boxes))
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

    return boxes.T

def calc_iou(
    image, targets, predictions, class_names, max_size=1920
):  # Return a list here of IoU per class

    iou_class_list = [0.0] * len(class_names)
    counter_list = [0] * len(class_names)

    bs, _, h, w = image.shape
    x = y = 0

    scale = max_size / max(h, w)
    if scale < 1:
        h = math.ceil(scale * h)
        w = math.ceil(scale * w)

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    if isinstance(predictions, torch.Tensor):
        targets = targets.cpu().numpy()

    if len(targets) > 0 and len(predictions) > 0:
        ti = targets[targets[:, 0] == 0]
        pi = predictions[predictions[:, 0] == 0]

        gt_boxes = xywh2xyxy(ti[:, 2:6]).T
        pred_boxes = xywh2xyxy(pi[:, 2:6]).T

        gt_classes = ti[:, 1].astype("int")
        pred_classes = pi[:, 1].astype("int")

        gt_labels = ti.shape[1] == 6
        pred_labels = pi.shape[1] == 6

        gt_conf = None if gt_labels else ti[:, 6]
        pred_conf = None if pred_labels else pi[:, 6]

        if gt_boxes.shape[1]:
            if gt_boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                gt_boxes[[0, 2]] *= w  # scale to pixels
                gt_boxes[[1, 3]] *= h
            elif scale < 1:  # absolute coords need scale if image scales
                gt_boxes *= scale

        if pred_boxes.shape[1]:
            if pred_boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                pred_boxes[[0, 2]] *= w  # scale to pixels
                pred_boxes[[1, 3]] *= h
            elif scale < 1:  # absolute coords need scale if image scales
                pred_boxes *= scale

        gt_boxes[[0, 2]] += x
        gt_boxes[[1, 3]] += y

        pred_boxes[[0, 2]] += x
        pred_boxes[[1, 3]] += y

        gt_boxes = clip_boxes(gt_boxes.T)
        pred_boxes = clip_boxes(pred_boxes.T)

        # print(gt_boxes.shape, pred_boxes.shape, pred_boxes.min(), pred_boxes.max())
        
        
        threshold_pred_classes = []
        conf_thr = 0.5

        for k, _ in enumerate(pred_boxes.T.tolist()):
            p_cls = pred_classes[k]
            if pred_conf[k] > conf_thr:  # 0.25 conf thresh
                threshold_pred_classes.append(p_cls)

        missing_from_pred = list(set(gt_classes) - set(threshold_pred_classes))
        missing_from_GT = list(set(threshold_pred_classes) - set(gt_classes))

        for k, pred_box in enumerate(pred_boxes.T.tolist()):
            p_cls = pred_classes[k]

            if pred_conf[k] > conf_thr:
                if p_cls in missing_from_GT:
                    iou_class_list[p_cls] += 0.0
                    counter_list[p_cls] += 1

                elif p_cls in gt_classes:
                    gt_indices = [i for i, e in enumerate(gt_classes) if e == p_cls]
                    for g_idx in gt_indices:
                        gt_box = gt_boxes.T.tolist()[g_idx]
                        pred_box_tensor = torch.tensor(pred_box).unsqueeze(0)
                        gt_box_tensor = torch.tensor(gt_box).unsqueeze(0)
                        iou_value = custom_bbox_iou(pred_box_tensor, gt_box_tensor)
                        iou_class_list[p_cls] += iou_value.item()
                        counter_list[p_cls] += 1

        for j, gt_box in enumerate(gt_boxes.T.tolist()):
            cls = gt_classes[j]
            if cls in missing_from_pred:
                iou_class_list[cls] += 0.0
                counter_list[cls] += 1

        counter_zero_indices = [i for i, e in enumerate(counter_list) if e == 0]

        for c_idx in counter_zero_indices:
            iou_class_list[c_idx] = np.nan

        total_counter = np.sum(np.array(counter_list))
        final_iou_class_list = np.array(iou_class_list) / total_counter

    else:
        return None

    return final_iou_class_list


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (
            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
        )  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, "a") as f:
            f.write(("%g " * len(line)).rstrip() % line + "\n")


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append(
            {
                "image_id": image_id,
                "category_id": class_map[int(p[5])],
                "bbox": [round(x, 3) for x in b],
                "score": round(p[4], 5),
            }
        )


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(
        detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device
    )
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where(
        (iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5])
    )  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = (
            torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        )  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def custom_bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes

    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


@torch.no_grad()
def run(
    data,
    weights=None,  # model.pt path(s)
    batch_size=32,  # batch size
    imgsz=640,  # inference size (pixels)
    conf_thres=0.001,  # confidence threshold
    iou_thres=0.6,  # NMS IoU threshold
    task="val",  # train, val, test, speed or study
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    single_cls=False,  # treat as single-class dataset
    augment=False,  # augmented inference
    verbose=False,  # verbose output
    save_txt=False,  # save results to *.txt
    save_hybrid=False,  # save label+prediction hybrid results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_json=False,  # save a COCO-JSON results file
    project=ROOT / "runs/val",  # save to project/name
    name="exp",  # save to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    half=True,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    model=None,
    dataloader=None,
    save_dir=Path(""),
    plots=True,
    callbacks=Callbacks(),
    compute_loss=None,
    rect=True,
):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = (
            next(model.parameters()).device,
            True,
            False,
            False,
        )  # get model device, PyTorch model

        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(
            Path(project) / name, exist_ok=exist_ok
        )  # increment run
        (save_dir / "labels" if save_txt else save_dir).mkdir(
            parents=True, exist_ok=True
        )  # make dir

        # Load model
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= (
            pt or jit or engine
        ) and device.type != "cpu"  # half precision only supported by PyTorch on CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()
        elif engine:
            batch_size = model.batch_size
        else:
            half = False
            batch_size = 1  # export.py models default to batch-size 1
            device = torch.device("cpu")
            LOGGER.info(
                f"Forcing --batch-size 1 square inference shape(1,3,{imgsz},{imgsz}) for non-PyTorch backends"
            )

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    is_coco = isinstance(data.get("val"), str) and data["val"].endswith(
        "coco/val2017.txt"
    )  # COCO dataset
    nc = 1 if single_cls else int(data["nc"])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()


    # Configure resnet model
    resnet_model = models.resnet50(pretrained=True)
    resnet_model = torch.nn.Sequential(*(list(resnet_model.children())[:-1]))
    resnet_model.eval().to(device)

    # Dataloader
    if not training:
        model.warmup(imgsz=(1, 3, imgsz, imgsz), half=half)  # warmup
        pad = 0.0 if task == "speed" else 0.5
        task = (
            task if task in ("train", "val", "test") else "val"
        )  # path to train/val/test images
        # dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=pt,
        #                                workers=workers, prefix=colorstr(f'{task}: '))[0]

        dataloader = create_dataloader(
            data[task],
            imgsz,
            batch_size,
            stride,
            single_cls,
            augment=False,
            rect=True,
            workers=workers,
            prefix=colorstr(f"{task}: "),
            shuffle=False,
        )[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {
        k: v
        for k, v in enumerate(
            model.names if hasattr(model, "names") else model.module.names
        )
    }
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ("%20s" + "%11s" * 6) % (
        "Class",
        "Images",
        "Labels",
        "P",
        "R",
        "mAP@.5",
        "mAP@.5:.95",
    )
    dt, p, r, f1, mp, mr, map50, map, miou = (
        [0.0, 0.0, 0.0],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    mIoUs = np.zeros((447, 9))
    pbar = tqdm(
        dataloader, desc=s, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )  # progress bar
    stat_num_preds = []

    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        t1 = time_sync()
        if pt or jit or engine:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        # print(im.shape)
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # visualize = increment_path(save_dir / Path(paths[batch_i]).stem, mkdir=True)
        visualize = False

        # Inference
        outs, cached_features = (
            model(im)
            if training
            else model(
                im, augment=augment, val=True, visualize=visualize
            )  # Setting vis to true
        )  # inference, loss outputs

        dt[1] += time_sync() - t2

        out, train_out = outs

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[
                1
            ]  # box, obj, cls

        # NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(
            device
        )  # to pixels
        lb = (
            [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []
        )  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(
            out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls
        )
        dt[2] += time_sync() - t3

        num_bboxes = get_num_preds(im, output_to_target(out), cached_features, names, paths)
        stat_num_preds.append(num_bboxes)
        
        # Setup resnet forward pass
        resnet_sample_input = torch.rand(num_bboxes, 3, 224, 224).to(device)
        t4 = time_sync()

        output = resnet_model(resnet_sample_input)
        output = torch.flatten(output, start_dim=1)
        output = F.adaptive_avg_pool1d(output.unsqueeze(0), 512).squeeze(0)
        
        dt[1] += time_sync() - t4

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append(
                        (
                            torch.zeros(0, niou, dtype=torch.bool),
                            torch.Tensor(),
                            torch.Tensor(),
                            tcls,
                        )
                    )
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(
                im[si].shape[1:], predn[:, :4], shape, shapes[si][1]
            )  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(
                    im[si].shape[1:], tbox, shape, shapes[si][1]
                )  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append(
                (correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls)
            )  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(
                    predn,
                    save_conf,
                    shape,
                    file=save_dir / "labels" / (path.stem + ".txt"),
                )
            if save_json:
                save_one_json(
                    predn, jdict, path, class_map
                )  # append to COCO-JSON dictionary
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

        # print(targets.shape, output_to_target(out).shape)
        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f"val_batch{batch_i}_labels.jpg"  # labels
            Thread(
                target=plot_images,
                args=(im, targets, paths, f, names, "GT"),
                daemon=True,
            ).start()
            f = save_dir / f"val_batch{batch_i}_pred.jpg"  # predictions
            Thread(
                target=plot_images,
                args=(im, output_to_target(out), paths, f, names, "PRED"),
                daemon=True,
            ).start()

        # Compute mIoU here
        # Get back a list here

        IoU_list = calc_iou(im, targets, output_to_target(out), names)
        mIoUs[batch_i] = np.array(IoU_list)
        # break

    # for idx in range(9):
    #     print(np.all(np.isnan(mIoUs[:, idx])))

    mIoUs = np.nanmean(mIoUs, axis=0)
    print(f"Avg box preds: {np.array(stat_num_preds).mean()}")
    np.save("num_preds.npy", np.array(stat_num_preds))

    print(f"Class-wise IoU Values for the validation set {mIoUs}")
    print(f"mIoU for the validation set {np.nanmean(mIoUs)}")

    # Compute metrics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            *stats, plot=plots, save_dir=save_dir, names=names
        )
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(
            stats[3].astype(np.int64), minlength=nc
        )  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = "%20s" + "%11i" * 2 + "%11.3g" * 4  # print format
    LOGGER.info(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    # print(ap_class)

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1e3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(
            f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}"
            % t
        )
        LOGGER.info(f"Average FPS with batch size {batch_size} is {1e3/t[1]}")

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run("on_val_end")

    # Save JSON
    if save_json and len(jdict):
        w = (
            Path(weights[0] if isinstance(weights, list) else weights).stem
            if weights is not None
            else ""
        )  # weights
        anno_json = str(
            Path(data.get("path", "../coco")) / "annotations/instances_val2017.json"
        )  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f"\nEvaluating pycocotools mAP... saving {pred_json}...")
        with open(pred_json, "w") as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(["pycocotools"])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, "bbox")
            if is_coco:
                eval.params.imgIds = [
                    int(Path(x).stem) for x in dataloader.dataset.img_files
                ]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            LOGGER.info(f"pycocotools unable to run: {e}")

    # Return results
    model.float()  # for training
    if not training:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path"
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s.pt",
        help="model.pt path(s)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        type=int,
        default=320,
        help="inference size (pixels)",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.001, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.5, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--task", default="val", help="train, val, test, speed or study"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="max dataloader workers (per RANK in DDP mode)",
    )
    parser.add_argument(
        "--single-cls", action="store_true", help="treat as single-class dataset"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--verbose", action="store_true", help="report mAP by class")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-hybrid",
        action="store_true",
        help="save label+prediction hybrid results to *.txt",
    )
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-json", action="store_true", help="save a COCO-JSON results file"
    )
    parser.add_argument(
        "--project", default=ROOT / "runs/val", help="save to project/name"
    )
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith("coco.yaml")
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(
        requirements=ROOT / "requirements.txt", exclude=("tensorboard", "thop")
    )

    if opt.task in ("train", "val", "test"):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(
                f"WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values."
            )
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == "speed":  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == "study":  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f"study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt"  # filename to save to
                x, y = (
                    list(range(256, 1536 + 128, 128)),
                    [],
                )  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f"\nRunning {f} --imgsz {opt.imgsz}...")
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt="%10.4g")  # save
            os.system("zip -r study.zip study_*.txt")
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


