import os
import io
import logging
import numpy as np
import cv2
import joblib

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from skimage.feature import hog
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
app = Flask(__name__, static_folder='.')
CORS(app)
IMG_SIZE = (64, 64)
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'visualize': False,
    'channel_axis': -1
}
MODEL_PATH = os.path.join('models', 'hog_svm_malaria.joblib')
SCALER_PATH = os.path.join('models', 'scaler_malaria.joblib')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024
model = None
scaler = None


def load_model():
    """Charge le modèle SVM et le scaler depuis le dossier models/."""
    global model, scaler
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Modèle introuvable : {MODEL_PATH}")
            return False
        if not os.path.exists(SCALER_PATH):
            logger.error(f"Scaler introuvable : {SCALER_PATH}")
            return False

        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Modèle HOG+SVM chargé avec succès !")
        return True

    except Exception as e:
        logger.error(f"Erreur lors du chargement : {e}")
        return False


def allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """
    Décode les bytes d'image, redimensionne et convertit en RGB.
    Retourne None si l'image est invalide.
    """
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return img_resized


def extract_hog_features(img: np.ndarray) -> np.ndarray:
    """Extrait le vecteur HOG d'une image RGB (H, W, 3)."""
    try:
        features = hog(img, **HOG_PARAMS)
    except TypeError:
        # Fallback pour les anciennes versions de skimage
        params = {k: v for k, v in HOG_PARAMS.items() if k != 'channel_axis'}
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        features = hog(img_gray, **params)

    return features.reshape(1, -1)


@app.route('/')
def index():
    """Sert le fichier index.html."""
    return send_from_directory('.', 'index.html')


@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé pour vérifier que l'API et le modèle sont prêts."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'img_size': IMG_SIZE
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({
            'error': 'Modèle non chargé. Vérifiez que les fichiers .joblib '
                     'sont dans le dossier models/.'
        }), 503

    if 'file' not in request.files:
        return jsonify({'error': "Aucun fichier fourni (champ 'file' requis)."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide.'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f"Format non supporté. Formats acceptés : {ALLOWED_EXTENSIONS}"
        }), 400

    img_bytes = file.read()
    if len(img_bytes) > MAX_FILE_SIZE:
        return jsonify({'error': f'Fichier trop grand (max {MAX_FILE_SIZE // 1024 // 1024} MB).'}), 413

    try:
        img = preprocess_image(img_bytes)
        if img is None:
            return jsonify({'error': 'Image invalide ou corrompue.'}), 400

        # Extraction HOG
        features = extract_hog_features(img)
        features_scaled = scaler.transform(features)

        # Prédiction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        label = 'Parasitized' if prediction == 1 else 'Uninfected'
        confidence = float(probabilities[int(prediction)]) * 100
        prob_uninfected = float(probabilities[0]) * 100
        prob_parasitized = float(probabilities[1]) * 100

        # Niveau de risque
        if label == 'Parasitized':
            risk = 'Élevé' if confidence > 80 else 'Modéré'
        else:
            risk = 'Faible'

        result = {
            'prediction': label,
            'confidence': round(confidence, 2),
            'risk_level': risk,
            'probabilities': {
                'Uninfected': round(prob_uninfected, 2),
                'Parasitized': round(prob_parasitized, 2)
            },
            'image_info': {
                'filename': file.filename,
                'size_bytes': len(img_bytes),
                'processed_size': f"{IMG_SIZE[0]}x{IMG_SIZE[1]}"
            }
        }

        logger.info(
            f"Prédiction : {label} | Confiance : {confidence:.1f}% | "
            f"Fichier : {file.filename}"
        )
        return jsonify(result)

    except Exception as e:
        logger.exception("Erreur lors de la prédiction")
        return jsonify({'error': f'Erreur interne : {str(e)}'}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """Retourne les informations sur le modèle chargé."""
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 503

    info = {
        'model_type': type(model).__name__,
        'img_size': IMG_SIZE,
        'hog_params': {k: str(v) for k, v in HOG_PARAMS.items()},
        'classes': ['Uninfected (0)', 'Parasitized (1)'],
        'model_size_kb': round(os.path.getsize(MODEL_PATH) / 1024, 1),
        'scaler_size_kb': round(os.path.getsize(SCALER_PATH) / 1024, 1),
    }
    return jsonify(info)


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)

    logger.info("=" * 55)
    logger.info("  🦟 API Détection Malaria — HOG + SVM")
    logger.info("=" * 55)

    model_ok = load_model()
    if not model_ok:
        logger.warning("Démarrage sans modèle — placez les fichiers .joblib dans models/")

    logger.info("Serveur démarré sur http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
