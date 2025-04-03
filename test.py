#!/usr/bin/env python3
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/final_model.keras")

# Load a test image (ensure it's from your dataset)
img_path = "data/Chest_data/A2157.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)
print(f"Prediction: {prediction[0][0]}")

