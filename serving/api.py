import pickle
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torchvision import transforms
from PIL import Image
import base64
from io import BytesIO
import numpy as np
import joblib
import csv
import os
import logging
import pandas as pd
from scripts.log_reg_training import train_log_reg, save_model
from scripts.preprocess_image import preprocess_image


logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s [%(levelname)s] %(message)s",  # Include timestamp and log level
    handlers=[
        logging.StreamHandler()  # Print logs to the console
    ]
)



# Initialisation de l'application FastAPI
app = FastAPI()

# Variables globales pour charger les modèles
embedding_model = None
scaler_model = None
prediction_model = None
cifar10_classes = {
            0: "Airplane",
            1: "Automobile",
            2: "Bird",
            3: "Cat",
            4: "Deer",
            5: "Dog",
            6: "Frog",
            7: "Horse",
            8: "Ship",
            9: "Truck"
        }
k = 10

# Charger les modèles au démarrage de l'API
@app.on_event("startup")
async def load_models():
    global embedding_model, scaler_model, prediction_model
    try:

        with open('./artifacts/resnet50_embedding.pkl', 'rb') as f:
            embedding_model = pickle.load(f)
        embedding_model.eval()

        with open('./artifacts/scaler_resnet50.pkl', 'rb') as f:
            scaler_model = pickle.load(f)

        with open('./artifacts/reglog_model_resnet50.pkl', 'rb') as f:
            prediction_model = pickle.load(f)

    except Exception as e:
        print(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail="Error loading models")

# Définir la structure des données reçues par l'API (utiliser Pydantic pour la validation)
class PredictRequest(BaseModel):
    image: str  # L'image en base64


# Endpoint de prédiction
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Extraire l'image base64 de la requête
        base64_image = request.image
        
        embedding_scaled = preprocess_image(base64_image, embedding_model, scaler_model)
        
        # Étape 3 : Prédiction
        output = prediction_model.predict(embedding_scaled)
        prediction = output.tolist()[0]
        prediction = cifar10_classes[prediction]
        
        # Retourner la prédiction sous forme de JSON
        return {"prediction": prediction}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")


# Définir la structure des données reçues par l'API (utiliser Pydantic pour la validation)
class FeedbackRequest(BaseModel):
    image: str  # L'image en base64
    prediction: str  # La prédiction
    feedback: str  # Le feedback (vrai valeur)

# Endpoint pour gérer les feedbacks
@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    try:
        # Extraire l'image base64 de la requête
        base64_image = request.image
        
        # Prétraiter l'image
        embedding_data = preprocess_image(base64_image, embedding_model, scaler_model)
        embedding_data = embedding_data.flatten()


        for key, val in cifar10_classes.items():
            if val == request.prediction:
                key_cifar10_prediction = key
        for key, val in cifar10_classes.items():
            if val == request.feedback:
                key_cifar10_feedback = key

        #Créer une ligne pour le csv de reporting
        row = embedding_data.tolist()
        row.append(key_cifar10_feedback)
        row.append(key_cifar10_prediction)
        logging.info(f"\n----Received prediction request : \nembedding size : {len(row)-2}\nprediction : {key_cifar10_feedback}\nfeedback : {key_cifar10_prediction}\n----")

        
        # Charger le fichier prod_data.csv et écrire la ligne dedans
        prod_data_path = "./data/prod_data.csv"
        with open(prod_data_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # Write the data (embedding + truth_value + prediction)
            writer.writerow(row)

        # Vérifier si le nombre de lignes dans prod_data.csv est un multiple de k
        data_prod = pd.read_csv(prod_data_path)
        if len(data_prod)%k == 0:
            global prediction_model
            ref_data_path = "./data/ref_data.csv"
            #Réentrainer le modèle
            logging.info("On réentraine le modèle")
            # Charger les données de prod_data.csv
            data_ref = pd.read_csv(ref_data_path)

            data = pd.concat([data_ref, data_prod], ignore_index=True)

            # Séparer les features et les labels
            X = data.iloc[:, :-2].values
            y = data.iloc[:, -2].values

            # Entraîner le modèle
            new_prediction_model = train_log_reg(X, y)
            save_model(new_prediction_model, "./artifacts/reglog_model_resnet50.pkl")
            logging.info("Modèle réentrainé avec succès.")

            # Vider le fichier prod_data.csv sauf la première ligne (en-tête)
            with open(prod_data_path, 'r') as file:
                lines = file.readlines()
            with open(prod_data_path, 'w') as file:
                file.write(lines[0])
                
            logging.info("Fichier prod_data.csv vidé.")

            #Réenregistrer les données ref_data.csv
            data["prediction"] = prediction_model.predict(X)
            data.to_csv(ref_data_path, index=False)

            # Charger le nouveau modèle
            prediction_model = new_prediction_model

            return {"message" : "Feedback enregistré avec succès. Modèle réentrainé et chargé avec succés."}
            
        return {"message" : "Feedback enregistré avec succès."}
    
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=f"Error during feedback: {str(e)}")

