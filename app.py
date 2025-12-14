from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration de l'application Flask
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'clé_par_défaut_pour_development')

# Configuration depuis .env
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
MODEL_PATH = os.getenv('MODEL_PATH', 'mobilenetv2_agriculture.keras')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 16777216))  # 16MB par défaut
ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'png,jpg,jpeg,gif,bmp').split(','))

app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Charger le modèle
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Modèle chargé avec succès depuis %s", MODEL_PATH)
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {e}")
    model = None

# Définir les classes
categories = [
    "Adristyrannus", "Aphids", "Beetle", "Bugs", "Cabbage Looper", "Peach___Bacterial_spot", 
    "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", 
    "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy",
    "Cicadellidae", "Cutworm", "Earwig", "FieldCricket", "Grasshopper", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch", 
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Mediterranean fruit fly", "Mites", "RedSpider", "Riptortus", "Slug", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)", "Snail", "Thrips", "Weevil", "Whitefly",
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Background_without_leaves", "Blueberry___healthy", "Cherry___healthy", "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", "Corn___healthy", 
    "Corn___Northern_Leaf_Blight"
]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes):
    """Prétraiter l'image pour la prédiction"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
        return img_array
    except Exception as e:
        logger.error(f"Erreur prétraitement image: {e}")
        raise

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html', categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint pour la prédiction"""
    try:
        # Vérifier si un fichier a été uploadé
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        
        # Vérifier si un fichier a été sélectionné
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
        # Vérifier l'extension du fichier
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Type de fichier non autorisé. Utilisez: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Lire et prétraiter l'image
        file_bytes = file.read()
        
        if len(file_bytes) > MAX_FILE_SIZE:
            return jsonify({
                'error': f'Fichier trop volumineux. Taille max: {MAX_FILE_SIZE / 1024 / 1024}MB'
            }), 400
        
        img_array = preprocess_image(file_bytes)
        
        # Faire la prédiction
        if model is None:
            return jsonify({'error': 'Modèle non disponible'}), 500
            
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = categories[predicted_index]
        confidence = float(np.max(predictions[0]))
        
        # Formater la réponse
        result = {
            'filename': secure_filename(file.filename),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'confidence_percentage': f"{confidence * 100:.2f}%",
            'status': 'success'
        }
        
        logger.info(f"Prédiction réussie: {predicted_class} ({confidence:.2f})")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return jsonify({'error': f'Erreur lors de la prédiction: {str(e)}'}), 500

@app.route('/about')
def about():
    """Page À propos"""
    return render_template('about.html')

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Fichier trop volumineux'}), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Erreur interne du serveur'}), 500

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    
    logger.info(f"Démarrage de l'application en mode {FLASK_ENV}")
    app.run(
        host=host, 
        port=port, 
        debug=DEBUG
    )