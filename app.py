from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agriculture Pests Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000/"],  # Adresse frontend React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Charger le modèle sauvegardé
model = tf.keras.models.load_model("mobilenetv2_agriculture.keras")

# Définir les classes (même ordre que dans ton entraînement)
categories = [
    "Adristyrannus", "Aphids", "Beetle", "Bugs", "Cabbage Looper","Peach___Bacterial_spot","Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight","Potato___healthy","Potato___Late_blight","Raspberry___healthy",
    "Cicadellidae", "Cutworm", "Earwig", "FieldCricket", "Grasshopper","Soybean___healthy","Squash___Powdery_mildew","Strawberry___healthy","Strawberry___Leaf_scorch","Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___healthy","Tomato___Late_blight",
    "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot","Tomato___Tomato_mosaic_virus","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Mediterranean fruit fly", "Mites", "RedSpider", "Riptortus", "Slug","Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Orange___Haunglongbing_(Citrus_greening)" ,
    "Snail", "Thrips", "Weevil", "Whitefly","Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy","Background_without_leaves","Blueberry___healthy","Cherry___healthy","Cherry___Powdery_mildew","Corn___Cercospora_leaf_spot Gray_leaf_spot","Corn___Common_rust","Corn___healthy","Corn___Northern_Leaf_Blight"
]

app = FastAPI(title="Agriculture Pests Classifier API")

# Prétraitement image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Lire fichier image
        contents = await file.read()
        img_array = preprocess_image(contents)

        # Prédiction
        predictions = model.predict(img_array)
        predicted_class = categories[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return JSONResponse({
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
