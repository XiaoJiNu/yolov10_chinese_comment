from ultralytics import YOLOv10

# bug: 需要给出yaml文件路径，否则会默认使用yolov10s.pt，导致报错
model = YOLOv10(model="ultralytics/cfg/models/v10/yolov10n.yaml")
# If you want to finetune the model with pretrained weights, you could load the
# pretrained weights like below
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
# model = YOLOv10('yolov10{n/s/m/b/l/x}.pt')

model.train(data='coco128.yaml', epochs=500, batch=4, imgsz=640)