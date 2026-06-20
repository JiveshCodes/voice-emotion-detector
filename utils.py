import os
import logging
from typing import Tuple, Dict, Optional
import librosa
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# RAVDESS Emotion Mapping
# Filename format: modality (01) - vocal channel (02) - emotion (03) - intensity (04) - statement (05) - repetition (06) - actor (07)
# Example: 03-01-01-01-01-01-01.wav
EMOTION_MAP: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

EMOTION_EMOJIS: Dict[str, str] = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😊",
    "sad": "😢",
    "angry": "😡",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲",
    "crying": "😭",
    "laughing": "😂"
}

def load_audio(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file, converts it to mono, and resamples to the target rate.

    Args:
        file_path (str): Absolute or relative path to the audio file.
        sr (int): Target sample rate. Defaults to 22050.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the audio signal array and the sample rate.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: If librosa fails to load the audio file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found at: {file_path}")
    
    try:
        y, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        return y, sample_rate
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}")
        raise RuntimeError(f"Failed to load audio: {str(e)}")

def extract_features(file_path: Optional[str] = None, y: Optional[np.ndarray] = None, sr: int = 22050) -> np.ndarray:
    """
    Extracts acoustic features from an audio file or raw audio array and combines them into a single 1D feature vector.

    Features extracted:
        - MFCC (40 coefficients)
        - Chroma Features (12)
        - Mel Spectrogram (128)
        - Spectral Contrast (7)
        - Zero Crossing Rate (1)
        - RMS Energy (1)
        - Spectral Centroid (1)
        - Spectral Rolloff (1)
        - Spectral Bandwidth (1)
    
    Both mean and standard deviation (std) are calculated for each feature type.

    Args:
        file_path (str, optional): Path to the audio file.
        y (np.ndarray, optional): Pre-loaded raw audio array.
        sr (int): Target sample rate. Defaults to 22050.

    Returns:
        np.ndarray: A 1D feature vector of shape (384,).

    Raises:
        Exception: If extraction fails.
    """
    try:
        # Load audio file if path is provided, otherwise use pre-loaded array
        if file_path is not None:
            y, sample_rate = load_audio(file_path, sr=sr)
        elif y is not None:
            sample_rate = sr
        else:
            raise ValueError("Either file_path or y must be provided for feature extraction.")
        
        # 1. MFCC (40 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # 2. Chroma Features (12 pitch classes)
        stft = np.abs(librosa.stft(y))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # 3. Mel Spectrogram (128 mel bands)
        mel = librosa.feature.melspectrogram(y=y, sr=sample_rate)
        mel_mean = np.mean(mel, axis=1)
        mel_std = np.std(mel, axis=1)
        
        # 4. Spectral Contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        # 5. Zero Crossing Rate (1)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr, axis=1)
        zcr_std = np.std(zcr, axis=1)
        
        # 6. RMS Energy (1)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms, axis=1)
        rms_std = np.std(rms, axis=1)

        # 7. Spectral Centroid (1)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)
        centroid_mean = np.mean(centroid, axis=1)
        centroid_std = np.std(centroid, axis=1)

        # 8. Spectral Rolloff (1)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate)
        rolloff_mean = np.mean(rolloff, axis=1)
        rolloff_std = np.std(rolloff, axis=1)

        # 9. Spectral Bandwidth (1)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)
        bandwidth_mean = np.mean(bandwidth, axis=1)
        bandwidth_std = np.std(bandwidth, axis=1)
        
        # Combine all features into a single 1D array
        # Sizes: (40*2) + (12*2) + (128*2) + (7*2) + (1*2) + (1*2) + (1*2) + (1*2) + (1*2) = 384
        feature_vector = np.hstack([
            mfccs_mean, mfccs_std,
            chroma_mean, chroma_std,
            mel_mean, mel_std,
            contrast_mean, contrast_std,
            zcr_mean, zcr_std,
            rms_mean, rms_std,
            centroid_mean, centroid_std,
            rolloff_mean, rolloff_std,
            bandwidth_mean, bandwidth_std
        ])
        
        return feature_vector

    except Exception as e:
        input_desc = f"file {file_path}" if file_path is not None else "raw array"
        logger.error(f"Error extracting features from {input_desc}: {str(e)}")
        raise RuntimeError(f"Feature extraction failed: {str(e)}")

def augment_audio(y: np.ndarray, sr: int) -> list:
    """
    Returns a list of augmented versions of the input audio array.
    """
    augmented = []
    
    # 1. Pitch shift up (2 semitones)
    try:
        y_pitch_up = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
        augmented.append(y_pitch_up)
    except Exception as e:
        logger.warning(f"Pitch shift up failed: {e}")
        
    # 2. Pitch shift down (-2 semitones)
    try:
        y_pitch_down = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2)
        augmented.append(y_pitch_down)
    except Exception as e:
        logger.warning(f"Pitch shift down failed: {e}")
        
    # 3. Add noise
    try:
        noise = np.random.randn(len(y))
        y_noise = y + 0.005 * noise
        augmented.append(y_noise)
    except Exception as e:
        logger.warning(f"Noise addition failed: {e}")
        
    return augmented
