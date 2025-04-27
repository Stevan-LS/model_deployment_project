import base64
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms


# Fonction pour prétraiter l'image
def preprocess_image(base64_image: str, embedding_model, scaler_model):

    # Décoder l'image depuis sa chaîne base64
    img_data = base64.b64decode(base64_image)
    img = Image.open(BytesIO(img_data))  # Convertir en image PIL

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Redimensionner l'image à 224x224 pour ResNet50
        transforms.ToTensor(),  # Convertir l'image en un tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation
    ])

    # Appliquer la transformation à l'image
    img_tensor = transform(img).unsqueeze(0)  # Ajouter une dimension batch (1, C, H, W)

    # Passer l'image dans le modèle pour obtenir l'embedding
    with torch.no_grad():
        embedding = embedding_model(img_tensor)  # Obtenir l'embedding
        embedding = embedding.view(embedding.size(0), -1)  # Redimensionner les embeddings

    # Convertir l'embedding en numpy array
    embedding_np = embedding.numpy()

    # Appliquer la transformation du scaler sur l'embedding
    embedding_scaled = scaler_model.transform(embedding_np)

    return embedding_scaled