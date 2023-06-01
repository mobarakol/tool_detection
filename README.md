# tool_detection

## YOLOv5

### Model training on EndoVis18

`python3 -W ignore train.py --img 320 --cfg yolov5s.yaml --hyp hyp.scratch.yaml --batch 64 --epochs 100  --data data/endovis18.yaml --weights '' --workers 4 --name yolo_endo18_sc`

Change `--cfg` to train using other YOLOv5 models. Options: `yolov5s.yaml`, `yolov5m.yaml`, `yolov5l.yaml`, `yolov5x.yaml`.

### Inference

`python3 detect.py --source ../dataset/processed/images/val/ --weights runs/train/exp6/weights/best.pt --conf 0.25 --name yolo_endo18_sc`

`../dataset/processed/images/` is the directory of the images together with their bounding box annotation in `.xml` files.

### Validation

`python3 val.py --img 320 --weights runs/train/yolo_endo18_sc5/weights/best.pt --data data/endovis18.yaml --task test --name yolo_endo18_sc5`

## YOLOv7

### Model training on EndoVis18

`python3 train.py --workers 4 --device 0 --batch-size 64 --data data/endovis18.yaml --img 320 320 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7-endo18 --hyp data/hyp.scratch.yaml`

Change `--cfg` to train using other YOLOv7 models. Options: `yolov7.yaml`, `yolov7x.yaml`, `yolov7-w6.yaml`, `yolov7-d6.yaml`, `yolov7-e6.yaml`, `yolov7-e6e.yaml`, `yolov7-tiny.yaml`.

### Inference
`python3 detect.py --weights runs/train/exp6/weights/best.pt --conf 0.25 --img-size 320 --source ../dataset/processed/images/val/`

`../dataset/processed/images/` is the directory of the images together with their bounding box annotation in `.xml` files.

### Validation
`python3 test.py --data data/endovis18.yaml --img 320 --batch 32 --conf 0.001 --iou 0.5 --device 0 --weights runs/train/yolo_endo18_sc5/weights/best.pt --name yolov7_endo18_320_val`

### Aknowledgement

The code is modified from [YOLOv5 🚀 in PyTorch](https://github.com/ultralytics/yolov5) and [Official YOLOv7](https://github.com/WongKinYiu/yolov7).

