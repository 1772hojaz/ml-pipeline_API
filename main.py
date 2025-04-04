#!/usr/bin/env python3

import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from src.preprocessing import preprocess_data  # Assuming this is now aligned with preprocess_input
from src.model import train_model
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils import class_weight
import zipfile
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
        allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = "data"
os.makedirs(BASE_DIR, exist_ok=True)

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def get_next_folder_name():
    existing_folders = [d for d in os.listdir(BASE_DIR) if d.startswith("data")]
    numbers = [int(folder[4:]) for folder in existing_folders if folder[4:].isdigit()]
    next_number = max(numbers, default=0) + 1
    return f"data{next_number}"

def get_latest_model():
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)
    
    model_files = sorted(
        [f for f in os.listdir(model_dir) if f.startswith("final_model") and f.endswith(".keras")],
        key=lambda x: int(x.replace("final_model", "").replace(".keras", "") or 0)
    )
    
    if model_files:
        latest_model_path = os.path.join(model_dir, model_files[-1])
        logging.info(f"Loading latest model: {latest_model_path}")
        return load_model(latest_model_path)
    else:
        logging.error("No trained model found in 'models/' directory.")
        raise FileNotFoundError("No trained model found in 'models/' directory.")

try:
    model = get_latest_model()
except FileNotFoundError as e:
    model = None
    logging.warning(str(e))

@app.get("/")
def home():
    return {"message": "Lung Disease Classification API is running!"}

# Corrected to use EfficientNet's preprocessing for both training and prediction
image_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded! Retrain or upload a model first.")

        img_path = os.path.join(TEMP_DIR, file.filename)
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not os.path.exists(img_path):
            raise HTTPException(status_code=500, detail="Image file was not saved correctly.")

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_processed = preprocess_input(img_array)  # Consistent with training preprocessing

        logging.info(f"Image shape after preprocessing: {img_processed.shape}")

        prediction = model.predict(img_processed)
        confidence = float(prediction[0][0])
        result = "Atelectasis" if confidence > 0.5 else "Normal"

        logging.info(f"Prediction result: {result} with confidence: {confidence:.4f}")

        return {"prediction": result, "confidence": confidence}

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)

@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...)):
    try:
        folder_name = get_next_folder_name()
        folder_path = os.path.join(BASE_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        zip_path = os.path.join(folder_path, file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(folder_path)

        os.remove(zip_path)
        logging.info(f"ZIP file uploaded and extracted to {folder_path}")

        return {"message": f"ZIP file uploaded and extracted to {folder_name}."}
    except Exception as e:
        logging.error(f"Error during file upload: {str(e)}")
        return {"error": str(e)}

def find_atelectasis_normal(base_dir):
    """Find ALL directories containing 'atelectasis' or 'normal' (case-insensitive)."""
    atelectasis_dirs = []
    normal_dirs = []

    # Walk through all subdirectories recursively
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            lower_name = dir_name.lower()
            if "atelectasis" in lower_name:
                atelectasis_dirs.append(os.path.join(root, dir_name))
            if "normal" in lower_name:
                normal_dirs.append(os.path.join(root, dir_name))
    
    return atelectasis_dirs, normal_dirs

@app.post("/retrain/")
async def retrain_endpoint():
    global model
    temp_dataset = None
    try:
        # 1. Find latest dataset directory
        dataset_folders = sorted(
            [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))],
            key=lambda x: os.path.getmtime(os.path.join(BASE_DIR, x)),
            reverse=True
        )
        
        if not dataset_folders:
            raise HTTPException(404, detail={
                "message": "No training data found",
                "error": "Please upload a ZIP file first"
            })

        latest_dataset = os.path.join(BASE_DIR, dataset_folders[0])
        logging.info(f"Using dataset: {latest_dataset}")

        # 2. Find class directories (case-insensitive search)
        def find_class_dirs(base_path):
            class_dirs = {"atelectasis": [], "normal": []}
            for root, dirs, _ in os.walk(base_path):
                for d in dirs:
                    lower = d.lower()
                    if "atelectasis" in lower:
                        class_dirs["atelectasis"].append(os.path.join(root, d))
                    elif "normal" in lower:
                        class_dirs["normal"].append(os.path.join(root, d))
            return class_dirs

        class_dirs = find_class_dirs(latest_dataset)

        # 3. Validate directories
        if not class_dirs["atelectasis"] or not class_dirs["normal"]:
            missing = []
            if not class_dirs["atelectasis"]: missing.append("Atelectasis")
            if not class_dirs["normal"]: missing.append("Normal")
            raise HTTPException(400, detail={
                "message": "Missing required classes",
                "error": f"Could not find directories containing: {', '.join(missing)}"
            })

        # 4. Create temporary dataset
        temp_dataset = os.path.join(TEMP_DIR, "retrain_dataset")
        if os.path.exists(temp_dataset):
            shutil.rmtree(temp_dataset)
        
        os.makedirs(os.path.join(temp_dataset, "4. Atelectasis"), exist_ok=True)
        os.makedirs(os.path.join(temp_dataset, "9. Normal"), exist_ok=True)

        # 5. Copy files with unique names
        def safe_copy(sources, target_dir):
            count = 0
            for src_dir in sources:
                for file in os.listdir(src_dir):
                    src = os.path.join(src_dir, file)
                    if os.path.isfile(src):
                        dest = os.path.join(target_dir, f"{count}_{file}")
                        shutil.copy2(src, dest)
                        count += 1
            return count

        ate_count = safe_copy(class_dirs["atelectasis"], os.path.join(temp_dataset, "4. Atelectasis"))
        norm_count = safe_copy(class_dirs["normal"], os.path.join(temp_dataset, "9. Normal"))

        if ate_count == 0 or norm_count == 0:
            raise HTTPException(400, detail={
                "message": "Insufficient training data",
                "error": f"Atelectasis: {ate_count} images, Normal: {norm_count} images"
            })

        # 6. Preprocess data
        logging.info(f"Preprocessing data from {temp_dataset}")
        train_gen, val_gen, _ = preprocess_data(temp_dataset)

        # 7. Calculate class weights
        y_train = train_gen.classes
        class_weights = class_weight.compute_class_weight(
            'balanced', 
            classes=np.unique(y_train), 
            y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))

        # 8. Train and save new version
        logging.info("Starting model training...")
        trained_model, history = train_model(train_gen, val_gen, class_weight_dict)
        model = trained_model  # Update global model reference

        return {
            "message": "Model retrained successfully",
            "metrics": {
                "final_accuracy": round(history['accuracy'][-1], 4),
                "final_val_accuracy": round(history['val_accuracy'][-1], 4),
                "final_loss": round(history['loss'][-1], 4),
                "final_val_loss": round(history['val_loss'][-1], 4)
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Retraining failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail={
            "message": "Retraining process failed",
            "error": str(e)
        })
    finally:
        if temp_dataset and os.path.exists(temp_dataset):
            shutil.rmtree(temp_dataset, ignore_errors=True)
            logging.info(f"Cleaned up temporary dataset: {temp_dataset}")
            

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
