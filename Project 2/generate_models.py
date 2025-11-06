"""
@author: Muhammad Alyan 501096627
AER850 Project 2
Section 1

Model Generator Script
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D, Input, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set paths and parameters
base_path = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_path, 'Data', 'train')
valid_dir = os.path.join(base_path, 'Data', 'valid')
test_dir = os.path.join(base_path, 'Data', 'test')
figures_dir = os.path.join(base_path, 'figures')
models_dir = os.path.join(base_path, 'models')
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

IMG_SIZE = (500, 500)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 3

# %%
# Step 1: Data Processing

print("Step 1: Preparing data generators with augmentation...")
# Training data: rescale + simple augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Validation/Test data: only rescale
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
valid_generator = val_datagen.flow_from_directory(valid_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
test_generator = val_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# %%
# Step 2 & 3: Neural Network Architecture Design & Hyperparameter Analysis
def create_model_1():

    model = Sequential([
        # Block 1: small number of filters to learn low-level features
        Input(shape=(500, 500, 3)),

        Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2: more filters to learn more complex patterns
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 3: deepen slightly
        Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten + dense head
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def create_model_2():

    model = Sequential([
        Input(shape=(500, 500, 3)),
        
        # Block 1 small number of filters to learn low-level features
        Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation=LeakyReLU(alpha=0.01), padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 2 more filters to learn more complex patterns
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation=LeakyReLU(alpha=0.01), padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 3 deepen slightly
        Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation=LeakyReLU(alpha=0.01), padding="same"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Block 4 deepen slightly
        Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation=LeakyReLU(alpha=0.01), padding="same"),
        BatchNormalization(),

        # Flatten + dense head
        Flatten(),
        Dense(32, activation=LeakyReLU(alpha=0.01)),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Early stopping to prevent overfitting and reduce training time
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, min_delta=0.001)

# Shared training configuration
steps_per_epoch = len(train_generator)
validation_steps = len(valid_generator)
models_to_run = [
    ("Model 1", "defect_classifier_model1.h5", create_model_1)
    ("Model 2", "defect_classifier_model2.h5", create_model_2)
]

trained_models = {}
histories = {}

# Train each model
for label, filename, builder in models_to_run:
    print(f"\nStep 2: Training {label}...")
    model = builder()
    model.summary()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        callbacks=[early_stop]
    )
    trained_models[label] = (model, os.path.join(models_dir, filename))
    histories[label] = history.history

# %%
# Step 4: Model Evaluation
print("\nStep 4: Evaluating both models...")
test_steps = len(test_generator)
for label, (model, _) in trained_models.items():
    _, acc = model.evaluate(test_generator, steps=test_steps)
    print(f"{label} Test Accuracy: {acc:.4f}")

# Create figures comparing accuracy/loss
def save_history_plot(history, title, filename):
    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history.get('accuracy', []), label='Training')
    ax1.plot(history.get('val_accuracy', []), label='Validation')
    ax1.set_title(f'{title} Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history.get('loss', []), label='Training')
    ax2.plot(history.get('val_loss', []), label='Validation')
    ax2.set_title(f'{title} Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    plt.tight_layout()
    out_path = os.path.join(figures_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved per-model history figure: {out_path}")

for idx, (label, hist) in enumerate(histories.items(), 1):
    save_history_plot(hist, label, f'model{idx}_accuracy_loss.png')

# Save the models
print("\nSaving models...")
for label, (model, path) in trained_models.items():
    model.save(path)
    print(f"Saved {label} to: {path}")