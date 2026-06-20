import os
import sys
import pickle
from typing import Tuple, Dict, Any
import numpy as np
import tensorflow as tf

# Add root directory to path to import utils
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils import extract_features as utils_extract_features

# Cache objects
_model: Any = None
_scaler: Any = None
_label_encoder: Any = None

def load_model() -> Tuple[Any, Any, Any]:
    """
    Loads model assets from root directory and caches them.
    """
    global _model, _scaler, _label_encoder
    
    if _model is None or _scaler is None or _label_encoder is None:
        model_dir = os.path.join(ROOT_DIR, "model")
        model_path = os.path.join(model_dir, "emotion_model.h5")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        le_path = os.path.join(model_dir, "label_encoder.pkl")
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
            raise FileNotFoundError(
                f"Trained model artifacts not found in: '{model_dir}'.\n"
                "Please run train.py first to train the network."
            )
            
        print("Backend loading model and assets...")
        _model = tf.keras.models.load_model(model_path)
        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)
        with open(le_path, "rb") as f:
            _label_encoder = pickle.load(f)
            
    return _model, _scaler, _label_encoder

def predict_emotion(file_path: str) -> Tuple[str, float, Dict[str, float]]:
    """
    Runs local prediction on the provided audio file.
    """
    model, scaler, label_encoder = load_model()
    
    # Extract features
    features = utils_extract_features(file_path)
    
    # Handle NaNs and Infs
    if np.any(np.isnan(features)):
        features = np.nan_to_num(features, nan=0.0)
    if np.any(np.isinf(features)):
        features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
        
    features_reshaped = features.reshape(1, -1)
    features_scaled = scaler.transform(features_reshaped)
    
    # Predict
    probabilities = model.predict(features_scaled, verbose=0)[0]
    
    pred_idx = int(np.argmax(probabilities))
    predicted_emotion = str(label_encoder.classes_[pred_idx])
    confidence = float(probabilities[pred_idx])
    
    prob_dict = {
        str(label_encoder.classes_[i]): float(probabilities[i])
        for i in range(len(probabilities))
    }
    
    return predicted_emotion, confidence, prob_dict
