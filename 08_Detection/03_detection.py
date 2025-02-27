from ultralytics import YOLO

# Load a pretrained YOLO8m model
model = YOLO("yolov8m.pt")

model.predict("./soccer.jpg", show=True, save=True, imgsz=320, conf=0.8)
