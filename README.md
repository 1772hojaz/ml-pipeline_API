# 🧠 Lung Disease Prediction API (FastAPI + Docker)

This is a backend API built with **FastAPI** to support a lung disease prediction app. It uses a trained deep learning model (e.g., CNN) to classify chest X-ray images into multiple lung diseases and supports uploading data and retraining the model. This backend is containerized with Docker.

---

## 🚀 Features

- ✅ Predict lung disease from chest X-ray image
- ⬆️ Upload new training data (images)
- 🔁 Trigger model retraining
- 📊 Get model performance metrics after retraining
- 🔒 CORS enabled for local frontend development

---

## 🧱 Tech Stack

- **FastAPI** for REST API
- **TensorFlow / Keras** for ML model
- **Uvicorn** ASGI server
- **Docker** for containerization
- **Python 3.8+**

---

## 📂 Project Structure

. ├── app/ │ ├── main.py # FastAPI app │ ├── model.py # Model loading & prediction │ ├── train.py # Retraining logic │ └── utils.py # Utilities ├── data/ # Uploaded training data ├── model/ # Saved models ├── Dockerfile ├── requirements.txt └── README.md

yaml
Copy
Edit

---

## 🐳 Docker Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/lung-disease-api.git
cd lung-disease-api
2. Build Docker image
bash
Copy
Edit
docker build -t lung-api .
3. Run the container
bash
Copy
Edit
docker run -d -p 8000:8000 lung-api
The API will be accessible at: http://127.0.0.1:8000

🧪 API Endpoints
POST /predict/
Description: Predict disease from an uploaded image.

Form Data: file (image)

Response:

json
Copy
Edit
{
  "prediction": "Atelectasis",
  "confidence": 0.88
}
POST /upload/
Description: Upload training data (images).

Form Data: One or more image files.

Response:

json
Copy
Edit
{
  "message": "Files uploaded successfully."
}
POST /retrain/
Description: Retrain model with new uploaded data.

Response:

json
Copy
Edit
{
  "message": "Model retrained successfully",
  "metrics": {
    "final_accuracy": 0.75,
    "final_val_accuracy": 1,
    "final_loss": 0.6285,
    "final_val_loss": 0.2701
  }
}
🔧 Development Setup (without Docker)
1. Create virtual environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run FastAPI server
bash
Copy
Edit
uvicorn app.main:app --reload
🌍 CORS Configuration
In main.py:

python
Copy
Edit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Update for frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
📦 Requirements
Example requirements.txt:

nginx
Copy
Edit
fastapi
uvicorn
python-multipart
tensorflow
pillow
scikit-learn
📤 Frontend Integration
You can use this API with any frontend (HTML/JS or React). Example frontend is available here.

