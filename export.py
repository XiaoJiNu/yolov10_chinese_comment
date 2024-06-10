from ultralytics import YOLOv10


"""
https://docs.ultralytics.com/integrations/tensorrt/#configuring-int8-export
"""
# model = YOLOv10.from_pretrained('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10('yolov10s.pt')

model.export(format="onnx", dynamic=True, simplify=True, opset=13)


# from ultralytics import YOLO
#
# model = YOLO("yolov10s.pt")
#
# success = model.export(format="onnx", dynamic=True, simplify=True, opset=13)
