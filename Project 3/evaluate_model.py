"""
@author: Muhammad Alyan 501096627
AER850 Project 3
Section 1

Model Evaluation Script
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # check for GPU
    if torch.cuda.is_available():
        device = 0
    else:
        device = 'cpu'

    # load model
    model = YOLO("runs/detect/pcb_component_detection/weights/best.pt")

    # test model
    results = model.val(data="Project 3 Data/data/data.yaml", split='test', device=device)

    print("\nResults:")
    print(f"Precision: {results.box.mp:.3f}")
    print(f"Recall: {results.box.mr:.3f}")
    print(f"mAP@0.5: {results.box.map50:.3f}")
    print(f"mAP@0.5:0.95: {results.box.map:.3f}")

