from ultralytics import YOLO

model = YOLO("Models\yolov8x-pose.onnx")

results = model.predict(source=0, show=True)