# Image Classification with Deployment Pipeline

## Project Overview
This project implements an end-to-end machine learning solution for image classification using deep learning techniques. The system processes images through a ResNet50 feature extraction pipeline, trains classification models on the embeddings, and provides both API serving capabilities and reporting functionalities.

## Key Features
- **Feature Extraction**: Implementation of ResNet50 for image feature extraction and embedding generation
- **Model Training**: Multiple classification models including Logistic Regression and Multi-Layer Perceptron
- **Containerized Deployment**: Docker-based deployment for both API serving and reporting services
- **Continuous Deployment**: Complete CI/CD pipeline for model deployment
- **Scalable Architecture**: Separated serving and reporting components

## Technical Architecture

### Data Processing Pipeline
1. Image data processing and transformation using PyTorch and torchvision
2. Feature extraction using pretrained ResNet50 model
3. Dimensionality reduction and scaling of embeddings
4. Creation of reference data for model training

### Model Development
- Model selection through cross-validation and hyperparameter optimization
- Performance evaluation using classification metrics
- Model serialization for deployment
- Support for incremental learning in production

### Deployment Components
- **Serving API**: REST API for real-time predictions
- **Reporting Module**: Analytics and monitoring dashboard
- **Containerization**: Docker and docker-compose for consistent deployment

## Technologies Used
- **Machine Learning**: scikit-learn, PyTorch
- **Data Processing**: Pandas, NumPy
- **Image Processing**: PIL, torchvision
- **Model Serialization**: pickle, joblib
- **Containerization**: Docker
- **API Development**: (Based on the API file in the serving directory)

## Project Structure
```
.
├── artifacts/               # Trained models and preprocessing objects
├── data/                    # Reference and production data
├── reporting/               # Monitoring and reporting service
├── scripts/                 # Training and processing scripts
├── serving/                 # API service for model predictions
└── webapp/                  # Web interface (if applicable)
```

## Key Achievements
- Successfully implemented transfer learning using ResNet50 for feature extraction
- Achieved high classification accuracy with optimized machine learning models
- Designed a scalable and maintainable production-ready ML system
- Created a complete CI/CD pipeline for model deployment and monitoring

## Future Improvements
- Implement A/B testing for model comparison
- Add automated model retraining based on performance metrics
- Extend the system to support additional image classification tasks

Stevan Le Stanc