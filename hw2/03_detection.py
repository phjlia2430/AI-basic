from ultralytics import YOLO

# Load a pretrained YOLO8m model
model = YOLO("yolov8m-seg.pt")

model.predict("./image1.jpg", show=True, save=True, conf=0.8)
