from ultralytics import YOLO
import torch

def run():
    torch.multiprocessing.freeze_support()

    dataset = "C:/Users/shw/PycharmProjects/pythonProject/2024/DL/08_Detection/data.yaml"

    # Load a pretrained YOLO8m model
    model = YOLO("yolov8m.pt")

    # Train the model with MPS
    results = model.train(data=dataset, epochs=10)

if __name__ == '__main__':
    run()
