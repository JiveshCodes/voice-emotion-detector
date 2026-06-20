import os
import pickle
from typing import Tuple, Dict, Any
import numpy as np
import tensorflow as tf
from utils import extract_features as utils_extract_features

# Global cache variables to store models once loaded
_model: Any = None
_scaler: Any = None
_label_encoder: Any = None

def load_model() -> Tuple[Any, Any, Any]:
    """
    Loads the trained TensorFlow/Keras model, scaler, and label encoder.
    Caches the loaded objects in global variables to speed up subsequent requests.

    Returns:
        Tuple[Any, Any, Any]: (keras_model, standard_scaler, label_encoder)
        
    Raises:
        FileNotFoundError: If model files have not been generated.
    """
    global _model, _scaler, _label_encoder
    
    if _model is None or _scaler is None or _label_encoder is None:
        model_dir = "model"
        model_path = os.path.join(model_dir, "emotion_model.h5")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        le_path = os.path.join(model_dir, "label_encoder.pkl")
        
        # Verify model files exist
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)):
            raise FileNotFoundError(
                f"Trained model artifacts not found in directory: '{model_dir}'.\n"
                "Please run train.py first to train the neural network and export model files."
            )
            
        print("Loading saved model and preprocessing assets...")
        # Load keras model
        _model = tf.keras.models.load_model(model_path)
        
        # Load scaler
        with open(scaler_path, "rb") as f:
            _scaler = pickle.load(f)
            
        # Load label encoder
        with open(le_path, "rb") as f:
            _label_encoder = pickle.load(f)
            
    return _model, _scaler, _label_encoder

def extract_features(file_path: str) -> np.ndarray:
    """
    Extracts audio features using the standard implementation in utils.py.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        np.ndarray: Combined feature vector of shape (384,).
    """
    return utils_extract_features(file_path)

def predict_emotion(file_path: str) -> Tuple[str, float, Dict[str, float]]:
    """
    Predicts the human emotion from a voice recording.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        Tuple[str, float, Dict[str, float]]: A tuple containing:
            - str: Predicted emotion label.
            - float: Confidence score (between 0.0 and 1.0).
            - Dict[str, float]: Probability distribution for all supported emotions.
    """
    # 1. Load model and scaler
    model, scaler, label_encoder = load_model()
    
    # 2. Extract features from audio
    features = extract_features(file_path)
    
    # 3. Handle data validation
    if np.any(np.isnan(features)):
        features = np.nan_to_num(features, nan=0.0)
    if np.any(np.isinf(features)):
        features = np.nan_to_num(features, posinf=0.0, neginf=0.0)
        
    # Reshape for single sample prediction: (1, 384)
    features_reshaped = features.reshape(1, -1)
    
    # 4. Scale features
    features_scaled = scaler.transform(features_reshaped)
    
    # 5. Predict emotion probabilities
    probabilities = model.predict(features_scaled, verbose=0)[0]
    
    # 6. Parse predicted index and confidence
    pred_idx = int(np.argmax(probabilities))
    predicted_emotion = str(label_encoder.classes_[pred_idx])
    confidence = float(probabilities[pred_idx])
    
    # 7. Construct complete probability dict
    prob_dict = {
        str(label_encoder.classes_[i]): float(probabilities[i])
        for i in range(len(probabilities))
    }
    
    return predicted_emotion, confidence, prob_dict

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("\n[INFO] Usage: python predict.py <path_to_audio_file.wav>")
        print("Example: python predict.py dataset/Actor_01/03-01-01-01-01-01-01.wav\n")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    print(f"Inference run for file: {audio_path}")
    try:
        emotion, confidence, probs = predict_emotion(audio_path)
        print("\n" + "="*30)
        print(f"Emotion: {emotion.capitalize()}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print("="*30)
        print("\nEmotion Probabilities:")
        for emo, prob in sorted(probs.items(), key=lambda item: item[1], reverse=True):
            print(f"  {emo.capitalize():<12}: {prob*100:.2f}%")
    except Exception as e:
        print(f"\n[ERROR] Inference failed: {str(e)}")
        sys.exit(1)
