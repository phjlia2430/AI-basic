from ultralytics import YOLO

# Load a pretrained YOLO8m model
model = YOLO("yolov8m-seg.pt")

model.predict("./VIRAT_CCTV1.mp4", show=True, save=True)
