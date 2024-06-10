from ultralytics import YOLOv10

model = YOLOv10("yolov10n.pt")
model.export(
    format="engine",
    dynamic=True,
    batch=8,
    workspace=4,
    int8=True,
    data="coco128.yaml",
)

# Load the exported TensorRT INT8 model
model = YOLOv10("yolov10n.engine", task="detect")

# Run inference
result = model.predict("https://ultralytics.com/images/bus.jpg")