# Image Classification with Deployment Pipeline

## Overview
A production-ready, end-to-end machine learning system for image classification using deep learning and modern MLOps practices. This project leverages transfer learning (ResNet50), robust model training, and a scalable deployment pipeline with real-time API serving, feedback-driven retraining, reporting dashboard, and an optional web interface.

---

## Features
- **Transfer Learning**: ResNet50-based feature extraction for robust image embeddings
- **Model Training**: Logistic Regression and MLP with cross-validation and hyperparameter tuning
- **Real-Time Serving**: FastAPI-based REST API for predictions and feedback
- **Feedback Loop**: Automated retraining on user feedback for continuous improvement
- **Reporting Dashboard**: Data drift and performance monitoring with Evidently
- **Web Interface**: User-friendly Streamlit app for image upload and feedback
- **Containerized Deployment**: Docker and docker-compose for reproducible, scalable deployment
- **CI/CD Ready**: Modular structure for easy integration with CI/CD pipelines

---

## Project Structure
```text
.
├── artifacts/      # Trained models and preprocessing objects
├── data/           # Reference and production data (CSV, CIFAR-10, etc.)
├── reporting/      # Monitoring and reporting service (Evidently)
├── scripts/        # Training, preprocessing, and data embedding scripts
├── serving/        # FastAPI service for model predictions and feedback
├── webapp/         # Streamlit web interface (optional)
└── README.md       # Project documentation
```

---

## Quick Start
### 1. Clone the Repository
```bash
git clone <repo-url>
cd <repo-root>
```

### 2. Prepare Data & Artifacts
- Place CIFAR-10 data in `data/` (or use provided scripts/notebooks to download and preprocess)
- Run the notebooks/scripts in `scripts/` to generate embeddings, scalers, and initial models:
  - `data_embedding.ipynb` or `model_training.ipynb` for feature extraction and model training
  - Artifacts will be saved in `artifacts/` and reference data in `data/ref_data.csv`

### 3. Build & Run with Docker Compose
You can launch all services (API, reporting, webapp) with Docker Compose:
```bash
# From the root directory
cd serving && docker-compose up --build
# In separate terminals, do the same for reporting/ and webapp/ if needed
cd ../reporting && docker-compose up --build
cd ../webapp && docker-compose up --build
```
- API: http://localhost:8080
- Reporting Dashboard: http://localhost:8082
- Web Interface: http://localhost:8081

---

## API Usage
### Prediction Endpoint
- **POST** `/predict`
- **Request Body:**
  ```json
  { "image": "<base64-encoded-image>" }
  ```
- **Response:**
  ```json
  { "prediction": "Cat" }
  ```

### Feedback Endpoint
- **POST** `/feedback`
- **Request Body:**
  ```json
  { "image": "<base64-encoded-image>", "prediction": "Cat", "feedback": "Dog" }
  ```
- **Response:**
  ```json
  { "message": "Feedback enregistré avec succès." }
  ```

### Example (Python)
```python
import requests, base64
with open('cat.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()
res = requests.post('http://localhost:8080/predict', json={'image': img_b64})
print(res.json())
```

---

## Reporting Dashboard
- Built with [Evidently](https://evidentlyai.com/)
- Monitors data drift, model performance, and feedback
- Accessible at [http://localhost:8082](http://localhost:8082)
- See `reporting/project.py` for customization

---

## Web Interface (Optional)
- Streamlit app for uploading images, viewing predictions, and submitting feedback
- Accessible at [http://localhost:8081](http://localhost:8081)
- See `webapp/api.py` for details

---

## Local Development & Installation
### Requirements
- Python 3.8+
- See `requirements.txt`, `serving/requirements.txt`, `reporting/requirements.txt`, `webapp/requirements.txt`

### Install Locally (for development)
```bash
pip install -r requirements.txt
cd serving && pip install -r requirements.txt
cd ../reporting && pip install -r requirements.txt
cd ../webapp && pip install -r requirements.txt
```

---

## Main Scripts & Notebooks
- `scripts/data_embedding.ipynb`: Extract embeddings from images using ResNet50
- `scripts/model_training.ipynb`: Model training, evaluation, and serialization
- `scripts/log_reg_training.py`: Logistic Regression training utility
- `scripts/preprocess_image.py`: Image preprocessing for inference
- `serving/api.py`: FastAPI app for serving and feedback
- `reporting/project.py`: Evidently dashboard setup
- `webapp/api.py`: Streamlit web interface


---

## Author
Stevan Le Stanc