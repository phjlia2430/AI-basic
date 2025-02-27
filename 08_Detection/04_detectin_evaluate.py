from ultralytics import YOLO
import torch

def run():
    torch.multiprocessing.freeze_support()

    model = YOLO("./runs/detect/train4/weights/best.pt")

    # Train the model with MPS
    results = model.predict(source='./test/images', save=True, conf=0.5)

if __name__ == '__main__':
    run()
