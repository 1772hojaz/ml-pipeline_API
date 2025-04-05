#  Lung Disease Prediction API (FastAPI & Docker)

This is a backend API built with **FastAPI** to support a lung disease prediction app. It uses a trained deep learning model (CNN) to classify chest X-ray images into multiple lung diseases and supports uploading data and retraining the model. This backend is containerized with Docker.


---
# Link to Video
[![Watch the video](https://img.youtube.com/vi/6kN2Jg3XTcY/0.jpg)](https://youtu.be/6kN2Jg3XTcY)


---

# Link to Front-End
https://1772hojaz.github.io/LungDisease-Front-End/

---
# Link to Front-End Repository
https://github.com/1772hojaz/LungDisease-Front-End

---
## Features

-  Predict lung disease from chest X-ray image
-  Upload new training data (images)
-  Trigger model retraining
-  Get model performance metrics after retraining
-  CORS enabled for local frontend development

---

##  Tech Stack

- **FastAPI** for REST API
- **TensorFlow / Keras** for ML model
- **Uvicorn** ASGI server
- **Docker** for containerization
- **Python 3.8+**

---

##  Project Structure

```graphql
├── app/
│   ├── main.py        # FastAPI app with API endpoints
│   ├── model.py       # Model loading and prediction logic
│   ├── train.py       # Model retraining functionality
│   
├── data/              # Directory for uploaded training data
├── model/             # Directory for saved/retrained models
├── Dockerfile         # Docker configuration for containerization
├── requirements.txt   # Python dependencies
└── README.md          # 
```

---

##  Docker Setup

### 1. Clone the repository

```bash
   git clone https://github.com/1772hojaz/ml-pipeline_API.git
  cd lung-disease-api
  ```

### 2. Build Docker image
  ```bash
    docker build -t lung-api .
  ```
### 3. Run the container
  ```bash
    docker run -d -p 8000:8000 lung-api
```
The API will be accessible at: http://127.0.0.1:8000

# API Endpoints
**POST**

/predict/


**Description:** Predict disease from an uploaded image.

Form Data: file (image)

Response:

```json
{
  "prediction": "Atelectasis",
  "confidence": 0.88
}
```
**POST**

/upload/

**Description:** Upload training data (images).

Form Data: One or more image files.

Response:

```json
{
  "message": "Files uploaded successfully."
}
```
**POST**

/retrain/

**Description:** Retrain model with new uploaded data.

Response:

```json
{
  "message": "Model retrained successfully",
  "metrics": {
    "final_accuracy": 0.75,
    "final_val_accuracy": 1,
    "final_loss": 0.6285,
    "final_val_loss": 0.2701
  }
}
