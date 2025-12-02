"""
@author: Muhammad Alyan 501096627
AER850 Project 2
Section 1

Prediction Script
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch
import cv2
import numpy as np

if __name__ == '__main__':
    # check for GPU
    if torch.cuda.is_available():
        device = 0
    else:
        device = 'cpu'

    # load model
    model = YOLO("runs/detect/pcb_component_detection/weights/best.pt")

    # directories
    imgs_dir = "Project 3 Data/data/evaluation"
    output = "prediction_results"

    os.makedirs(output, exist_ok=True)

    # get images
    imgs = [f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    print(f"Processing {len(imgs)} images...")

    # predict with visualization
    for img in imgs:
        path = os.path.join(imgs_dir, img)
        results = model.predict(source=path, save=True, project=output, name='', exist_ok=True, device=device, 
                               conf=0.1, line_width=4, show_labels=True, show_conf=True)
        print(f"  {img}: {len(results[0].boxes)} components")

    print(f"\nSaved to: {output}")

