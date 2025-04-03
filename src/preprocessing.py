#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input


def preprocess_data(dataset_path):
    # Correcting class labels for binary classification
    binary_classes = {'4. Atelectasis': 1, '9. Normal': 0}

    X, Y = [], []
    for folder, label in binary_classes.items():
        disease_path = os.path.join(dataset_path, folder, 'CXR')
        if not os.path.isdir(disease_path):
            disease_path = os.path.join(dataset_path, folder)  # Use direct folder if 'CXR' is missing

        if os.path.isdir(disease_path):
            for img_name in os.listdir(disease_path):
                img_path = os.path.join(disease_path, img_name)
                X.append(img_path)
                Y.append(label)

    # Converting lists to NumPy arrays
    X, Y = np.array(X), np.array(Y)
    print(f"Total images: {len(X)}")

    # Convert integer labels to string labels
    Y = np.array([['Normal', 'Atelectasis'][y] for y in Y])

    # Convert to DataFrame
    df = pd.DataFrame({'Image': X, 'Label': Y})

    # Split dataset
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Label'], random_state=42)

    # Image Data Generator with EfficientNet preprocessing
    image_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # create data generators with explicit class order
    def create_data_generator(df, shuffle=True):
        generator = image_generator.flow_from_dataframe(
            df,
            x_col='Image',
            y_col='Label',
            target_size=(224, 224),
            batch_size=64,
            class_mode='binary',
            classes=['Normal', 'Atelectasis'],  # Enforce correct class order
            shuffle=shuffle
        )
        return generator

    # generators with correct shuffle settings
    train_generator = create_data_generator(train_df, shuffle=True)
    val_generator = create_data_generator(val_df, shuffle=False)
    test_generator = create_data_generator(test_df, shuffle=False)

    return train_generator, val_generator, test_generator

