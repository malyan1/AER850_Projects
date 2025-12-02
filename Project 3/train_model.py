"""
@author: Muhammad Alyan 501096627
AER850 Project 3
Section 1

Training Script
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # check for GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = '0'
    else:
        print("No GPU found")
        device = 'cpu'

    # load model
    model = YOLO("yolo11n.pt")

    print("Training...")

    # train model
    model.train(
        data="Project 3 Data/data/data.yaml",
        epochs=75,
        batch=4,
        imgsz=960,            
        name='pcb_component_detection',
        workers=8,
        cache='disk',
        amp=True,
        patience=20,
        verbose=True,
        device=0,
        close_mosaic=10,
    )

    print("\nModel saved to runs/detect/pcb_component_detection/weights/best.pt")

