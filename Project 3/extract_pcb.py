"""
@author: Muhammad Alyan 501096627
AER850 Project 3
Section 1

PCB Extractor Script
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from ultralytics import YOLO

def extract_pcb(img_path, output='pcb_output'):
    # make folder
    os.makedirs(output, exist_ok=True)
    
    # read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Can't read: {img_path}")
        return None
    
    print(f"Image: {img.shape[1]}x{img.shape[0]}")
    
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # add blur
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # threshold - inverted for dark PCB
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f'{output}/pcb_threshold.jpg', thresh)
    
    # morphology to clean up
    kernel = np.ones((11, 11), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=6)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    
    # dilate to fill internal gaps only
    kernel2 = np.ones((15, 15), np.uint8)
    thresh = cv2.dilate(thresh, kernel2, iterations=3)
    
    # Canny edge detection
    edges = cv2.Canny(blur, 30, 90)
    cv2.imwrite(f'{output}/pcb_edges.jpg', edges)
    
    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No contours")
        return img
    
    # get largest contour
    biggest = max(contours, key=cv2.contourArea)
    
    # area filtering
    img_area = img.shape[0] * img.shape[1]
    if cv2.contourArea(biggest) < img_area * 0.2:
        print("Contour too small")
        return img
    
    # use actual contour as mask
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [biggest], -1, 255, -1)
    
    # minimal dilation
    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel_small, iterations=1)
    
    cv2.imwrite(f'{output}/pcb_mask.jpg', mask)
    
    # apply mask with bitwise_and
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(f'{output}/pcb_extracted.jpg', result)
    
    print(f"Done! PCB extracted")
    print(f"Saved to {output}/")
    
    return result

if __name__ == '__main__':
    print("\nExtracting PCB...")
    pcb = extract_pcb("motherboard_image.JPEG")
    print("\nComplete")
