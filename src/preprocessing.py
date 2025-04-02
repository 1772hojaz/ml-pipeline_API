#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImagePreprocessor:
    def __init__(self, dataset_path, binary_classes):
        self.dataset_path = dataset_path
        self.binary_classes = binary_classes

    def load_data(self):
        X, Y = [], []
        for folder, label in self.binary_classes.items():
            disease_path = os.path.join(self.dataset_path, folder, 'CXR')
            if not os.path.isdir(disease_path):
                disease_path = os.path.join(self.dataset_path, folder)  # Use direct folder if 'CXR' is missing
            
            if os.path.isdir(disease_path):
                for img_name in os.listdir(disease_path):
                    img_path = os.path.join(disease_path, img_name)
                    X.append(img_path)
                    Y.append(label)  # Assign binary labels
        
        X, Y = np.array(X), np.array(Y)
        Y = np.array([['Normal', 'Atelectasis'][y] for y in Y])
        
        # Convert lists to DataFrame for easier splitting
        df = pd.DataFrame({'Image': X, 'Label': Y})
        return df

    def split_data(self, df):
        train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Label'], random_state=42)
        return train_df, val_df, test_df

    def create_data_generator(self, df, image_generator):
        generator = image_generator.flow_from_dataframe(
            df, x_col='Image', y_col='Label',
            target_size=(224, 224), batch_size=64, class_mode='binary'
        )
        return generator

    def get_image_generator(self):
        image_generator = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        return image_generator

