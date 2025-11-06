"""
@author: Muhammad Alyan 501096627
AER850 Project 2
Section 1

Prediction Visualization Script
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths and constants
base_path = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_path, 'models')
figures_dir = os.path.join(base_path, 'figures')
test_dir = os.path.join(base_path, 'Data', 'test')
os.makedirs(figures_dir, exist_ok=True)

IMG_SIZE = (500, 500)

# Class ordering and naming
classes = ['crack', 'missing-head', 'paint-off']
pretty = {'crack': 'Crack', 'missing-head': 'Missing Head', 'paint-off': 'Paint-off'}

# Get model file directories
model1_path = os.path.join(models_dir, 'defect_classifier_model1.h5')
model2_path = os.path.join(models_dir, 'defect_classifier_model2.h5')

models = []

# Load models
models.append(('model1', load_model(model1_path)))
models.append(('model2', load_model(model2_path)))

# Get test images directories
test_images = {
    'crack': os.path.join(test_dir, 'crack', 'test_crack.jpg'),
    'missing-head': os.path.join(test_dir, 'missing-head', 'test_missinghead.jpg'),
    'paint-off': os.path.join(test_dir, 'paint-off', 'test_paintoff.jpg'),
}

print('\nGenerating Figure 3-style prediction visuals...')

# Generate predictions and figures for each model and test image
for model_name, mdl in models:
    for true_class, img_path in test_images.items():

        # Load and preprocess image to model's expected size, normalized to [0,1]
        pil_img = load_img(img_path, target_size=IMG_SIZE)
        img_arr = img_to_array(pil_img).astype('float32') / 255.0
        batch = np.expand_dims(img_arr, axis=0)

        # Predict probabilities and find most likely class
        probs = mdl.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = classes[pred_idx]

        # Compose overlay text lines
        lines = []
        for k in range(len(probs)):
            cname = pretty.get(classes[k] if k < len(classes) else str(k), str(k))
            lines.append(f"{cname}:{probs[k]*100:.1f}%")
        overlay_text = "\n".join(lines)

        # Build figure similar to the provided example
        fig = plt.figure(figsize=(5.2, 6.0))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(pil_img)
        ax.axis('off')

        # Titles above image
        fig.text(0.5, 0.98, f"True Crack Classification Label: {true_class}", ha='center', va='top', fontsize=12)
        fig.text(0.5, 0.94, f"Predicted Crack Classification Label: {pred_class}", ha='center', va='top', fontsize=12)

        # Overlay probabilities on the image in green
        ax.text(0.05, 0.75, overlay_text, transform=ax.transAxes, color='green', fontsize=14)

        # Save figure
        out_path = os.path.join(figures_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}_{model_name}_figure3.png")
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")
