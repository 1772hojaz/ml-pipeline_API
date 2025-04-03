#!/usr/bin/env python3

import os
import re
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from sklearn.utils import class_weight

# Configuration
MODELS_DIR = Path(__file__).parent.parent / "models"  # At project root
MODELS_DIR.mkdir(exist_ok=True, parents=True)

def build_model(fine_tune=False):
    """Build or load model with optional fine-tuning setup"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    if fine_tune:
        # Unfreeze last 20 layers for fine-tuning
        base_model.trainable = True
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def save_model_with_version(model):
    """Save model with incremental version number"""
    existing_versions = []
    for f in MODELS_DIR.glob("final_model*.keras"):
        match = re.search(r"final_model(\d+)\.keras", f.name)
        if match:
            existing_versions.append(int(match.group(1)))
    
    new_version = max(existing_versions) + 1 if existing_versions else 1
    new_name = f"final_model{new_version}.keras"
    model.save(MODELS_DIR / new_name)
    return new_name

def train_model(train_generator, val_generator, class_weight_dict):
    """Complete training workflow"""
    # Initial model setup
    model = build_model(fine_tune=False)
    
    # Callbacks configuration
    best_weights_path = MODELS_DIR / "best_weights.weights.h5"
    callbacks = [
        ModelCheckpoint(
            str(best_weights_path),
            save_best_only=True,
            monitor='val_loss',
            save_weights_only=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            verbose=1
        )
    ]
    
    # Initial training phase
    print("\n=== Initial Training Phase ===")
    initial_history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Fine-tuning phase
    print("\n=== Fine-Tuning Phase ===")
    fine_tune_model = build_model(fine_tune=True)
    fine_tune_model.load_weights(best_weights_path)
    
    fine_tune_history = fine_tune_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    # Save new version
    new_model_name = save_model_with_version(fine_tune_model)
    print(f"\nSaved new model to: {MODELS_DIR/new_model_name}")
    
    # Combine training histories
    combined_history = {
        'accuracy': initial_history.history['accuracy'] + fine_tune_history.history['accuracy'],
        'val_accuracy': initial_history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'],
        'loss': initial_history.history['loss'] + fine_tune_history.history['loss'],
        'val_loss': initial_history.history['val_loss'] + fine_tune_history.history['val_loss']
    }
    
    return fine_tune_model, combined_history
