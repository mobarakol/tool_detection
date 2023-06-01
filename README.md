# tool_detection

### Model training on EndoVis18

`python3 -W ignore train.py --img 320 --cfg yolov5s.yaml --hyp hyp.scratch.yaml --batch 64 --epochs 100  --data endovis18.yaml --weights '' --workers 4 --name yolo_endo18_sc`

Change `--cfg` to train using other YOLOv5 models. Option: `yolov5s.yaml`, `yolov5m.yaml`, `yolov5l.yaml`, `yolov5x.yaml`.

### Inference

`python3 detect.py --source ../dataset/processed/images/val/ --weights runs/train/exp6/weights/best.pt --conf 0.25 --name yolo_endo18_sc`

`../dataset/processed/images/` is the directory of the images together with their bounding box annotation in `.xml` files.

### Validation

`python3 val.py --img 320 --weights runs/train/yolo_endo18_sc5/weights/best.pt --data endovis18.yaml --task test --name yolo_endo18_sc5`

### Aknowledge

The code is modified from [YOLOv5 🚀 in PyTorch](https://github.com/ultralytics/yolov5).

