from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# -----------------------------
# Create FastAPI App
# -----------------------------
app = FastAPI(title="Skin Disease Prediction API")

# -----------------------------
# Enable CORS (for React)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load Trained Model
# -----------------------------
model = tf.keras.models.load_model(
    "skin_disease_model_stage2_finetuned.h5",
    compile=False
)

# -----------------------------
# Class Labels (UPDATE if needed)
# Make sure order matches training
# -----------------------------
class_names = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions"
]

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Skin Disease API is running successfully"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)[0]

    # Get predicted class
    predicted_index = int(np.argmax(prediction))
    predicted_label = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # Risk classification (optional logic)
    if predicted_label in ["Melanoma", "Basal Cell Carcinoma", "Actinic Keratoses"]:
        risk_level = "High Risk"
    else:
        risk_level = "Low Risk"

    return {
        "disease": predicted_label,
        "confidence": f"{confidence:.2f}%",
        "risk_level": risk_level
    }

# -----------------------------
# Run Server (for Hosting)
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )