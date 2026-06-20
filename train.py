import os
import sys
import pickle
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Import feature extraction functions and mappings
from utils import extract_features, EMOTION_MAP, load_audio, augment_audio

# Setup directories
DATASET_DIR = "dataset"
MODEL_DIR = "model"
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def download_crying_dataset():
    """
    Downloads crying_baby files from the ESC-50 dataset on GitHub if not already present.
    """
    crying_dir = os.path.join(DATASET_DIR, "crying")
    os.makedirs(crying_dir, exist_ok=True)
    
    # Check if we already have files in the crying directory
    existing_files = [f for f in os.listdir(crying_dir) if f.lower().endswith(".wav")]
    if len(existing_files) >= 40:
        print("Crying dataset already downloaded and extracted. Skipping.")
        return
        
    print("\nCrying dataset not found. Downloading crying_baby category from ESC-50 GitHub...")
    csv_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
    
    try:
        # Download and read CSV
        import csv
        import io
        
        response = urllib.request.urlopen(csv_url)
        csv_content = response.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(csv_content))
        
        filenames = []
        for row in reader:
            if row['category'] == 'crying_baby':
                filenames.append(row['filename'])
                
        total_crying = len(filenames)
        print(f"Found {total_crying} crying files in ESC-50 metadata. Starting download...")
        
        for i, filename in enumerate(filenames):
            file_url = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{filename}"
            target_path = os.path.join(crying_dir, filename)
            
            # Simple progress log
            sys.stdout.write(f"\rDownloading crying audio: {i+1}/{total_crying} files ({(i+1)*100/total_crying:.1f}%)")
            sys.stdout.flush()
            
            if not os.path.exists(target_path):
                urllib.request.urlretrieve(file_url, target_path)
                
        print("\nCrying dataset download completed successfully.")
    except Exception as e:
        print(f"\n[ERROR] Failed to download crying dataset: {str(e)}")
        sys.exit(1)

def download_laughing_dataset():
    """
    Downloads laughing files from the ESC-50 dataset on GitHub if not already present.
    """
    laughing_dir = os.path.join(DATASET_DIR, "laughing")
    os.makedirs(laughing_dir, exist_ok=True)
    
    # Check if we already have files in the laughing directory
    existing_files = [f for f in os.listdir(laughing_dir) if f.lower().endswith(".wav")]
    if len(existing_files) >= 40:
        print("Laughing dataset already downloaded and extracted. Skipping.")
        return
        
    print("\nLaughing dataset not found. Downloading laughing category from ESC-50 GitHub...")
    csv_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
    
    try:
        # Download and read CSV
        import csv
        import io
        
        response = urllib.request.urlopen(csv_url)
        csv_content = response.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(csv_content))
        
        filenames = []
        for row in reader:
            if row['category'] == 'laughing':
                filenames.append(row['filename'])
                
        total_laughing = len(filenames)
        print(f"Found {total_laughing} laughing files in ESC-50 metadata. Starting download...")
        
        for i, filename in enumerate(filenames):
            file_url = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{filename}"
            target_path = os.path.join(laughing_dir, filename)
            
            # Simple progress log
            sys.stdout.write(f"\rDownloading laughing audio: {i+1}/{total_laughing} files ({(i+1)*100/total_laughing:.1f}%)")
            sys.stdout.flush()
            
            if not os.path.exists(target_path):
                urllib.request.urlretrieve(file_url, target_path)
                
        print("\nLaughing dataset download completed successfully.")
    except Exception as e:
        print(f"\n[ERROR] Failed to download laughing dataset: {str(e)}")
        sys.exit(1)

def download_tess_dataset():
    """
    Downloads TESS (Toronto Emotional Speech Set) dataset from Borealis Dataverse if not already present.
    """
    tess_dir = os.path.join(DATASET_DIR, "tess")
    os.makedirs(tess_dir, exist_ok=True)

    # Check if we already have files in the TESS directory
    # There should be 2,800 files in TESS
    existing_files = []
    if os.path.exists(tess_dir):
        for root, _, files in os.walk(tess_dir):
            for f in files:
                if f.lower().endswith(".wav"):
                    existing_files.append(f)
    
    if len(existing_files) >= 2800:
        print("TESS dataset already downloaded and extracted. Skipping.")
        return

    print("\nTESS dataset not found (or incomplete). Downloading TESS from Borealis U of T Dataverse (~268 MB)...")
    url = "https://borealisdata.ca/api/access/dataset/:persistentId/?persistentId=doi:10.5683/SP2/E8H2MF"
    zip_path = os.path.join(DATASET_DIR, "tess.zip")

    try:
        # Download with progress report
        def reporthook(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = (read_so_far * 100.0) / total_size
                sys.stdout.write(f"\rDownloading TESS: {read_so_far / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({percent:.1f}%)")
            else:
                sys.stdout.write(f"\rDownloading TESS: {read_so_far / (1024 * 1024):.1f} MB downloaded")
            sys.stdout.flush()

        # Some sites block requests without a User-Agent
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)

        urllib.request.urlretrieve(url, zip_path, reporthook)
        print("\nTESS download completed successfully. Extracting...")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            members = zip_ref.infolist()
            total_members = len(members)
            for i, member in enumerate(members):
                sys.stdout.write(f"\rExtracting TESS files: {i+1}/{total_members}")
                sys.stdout.flush()
                # Extract file
                zip_ref.extract(member, tess_dir)
        print("\nTESS dataset extraction completed successfully.")
        
        # Clean up zip
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    except Exception as e:
        print(f"\n[ERROR] Failed to download/extract TESS dataset: {str(e)}")
        # Clean up partially downloaded zip
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except:
                pass
        print("Continuing training without TESS...")

def download_and_extract_ravdess():
    """
    Downloads the RAVDESS Speech dataset from Zenodo and extracts it
    if it is not already present in the dataset folder.
    """
    # Check if there are already actor directories in dataset
    has_actors = any(
        d.startswith("Actor_") and os.path.isdir(os.path.join(DATASET_DIR, d))
        for d in os.listdir(DATASET_DIR)
    )
    if has_actors:
        print("RAVDESS dataset directory contains Actor folders. Skipping download.")
        return

    url = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    zip_path = os.path.join(DATASET_DIR, "Audio_Speech_Actors_01-24.zip")

    print(f"Dataset not found locally. Downloading RAVDESS Speech dataset (~248 MB) from Zenodo...")
    
    def report_hook(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = (read_so_far * 100.0) / total_size
            sys.stdout.write(
                f"\rProgress: {percent:.2f}% ({read_so_far / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)"
            )
        else:
            sys.stdout.write(f"\rDownloaded: {read_so_far / (1024*1024):.2f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook=report_hook)
        print("\nDownload complete! Extracting files to dataset/...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)
            
        print("Extraction complete! Cleaning up zip file...")
        os.remove(zip_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to download or extract the dataset: {str(e)}")
        print("Please manually download 'Audio_Speech_Actors_01-24.zip' from:")
        print("https://zenodo.org/records/1188976")
        print(f"Extract its contents (Actor_01 to Actor_24) into: {os.path.abspath(DATASET_DIR)}")
        sys.exit(1)

# CREMA-D emotion code to standard label mapping
CREMA_D_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

# TESS folder name to standard label mapping
TESS_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "ps": "surprised"  # pleasant surprise -> surprised
}

def print_extra_dataset_instructions():
    """
    Prints instructions for manually downloading CREMA-D dataset
    to boost training data significantly.
    """
    crema_dir = os.path.join(DATASET_DIR, "crema_d")
    crema_present = os.path.isdir(crema_dir) and len([f for f in os.listdir(crema_dir) if f.endswith(".wav")]) > 100

    if not crema_present:
        print("\n" + "="*60)
        print("  OPTIONAL: Boost accuracy with CREMA-D")
        print("="*60)
        print("\n📦 CREMA-D (7,442 clips, 6 emotions) - adds ~5x more data!")
        print("  1. Install Git LFS: https://git-lfs.com")
        print("  2. Run:")
        print(f"       git lfs clone https://github.com/CheyneyComputerScience/CREMA-D")
        print(f"  3. Copy the AudioWAV/ folder to:")
        print(f"       {os.path.abspath(crema_dir)}")
        print("  OR download from Kaggle: https://www.kaggle.com/datasets/ejlok1/cremad")
        print("     and place all .wav files in the folder above.")
        print("\n  Re-run train.py after placing the dataset to unlock full accuracy.\n")
        print("="*60 + "\n")

def prepare_dataset() -> list:
    """
    Scans the dataset directory for RAVDESS, CREMA-D, TESS, crying, and laughing
    audio files and returns a list of tuples (file_path, emotion_name).
    """
    print("\nScanning dataset directory for audio files...")
    audio_files = []
    crema_dir = os.path.join(DATASET_DIR, "crema_d")
    tess_dir  = os.path.join(DATASET_DIR, "tess")
    
    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if not file.lower().endswith(".wav") or file.startswith("._"):
                continue
            file_path = os.path.join(root, file)
            filename  = os.path.basename(file_path)
            emotion_name = None

            # --- Crying / Laughing (ESC-50 subfolders) ---
            if "crying" in file_path.lower():
                emotion_name = "crying"
            elif "laughing" in file_path.lower():
                emotion_name = "laughing"

            # --- CREMA-D: files like 1001_TAI_HAP_XX.wav ---
            elif os.path.abspath(root).startswith(os.path.abspath(crema_dir)):
                parts = os.path.splitext(filename)[0].split("_")
                if len(parts) >= 3:
                    code = parts[2].upper()
                    if code in CREMA_D_EMOTION_MAP:
                        emotion_name = CREMA_D_EMOTION_MAP[code]

            # --- TESS: files inside folders named like OAF_angry or YAF_happy ---
            elif os.path.abspath(root).startswith(os.path.abspath(tess_dir)):
                folder = os.path.basename(root).lower()
                for key, label in TESS_EMOTION_MAP.items():
                    if folder.endswith("_" + key) or folder == key:
                        emotion_name = label
                        break
                
                # Suffix check fallback if folders are flat or named differently
                if not emotion_name:
                    name_part = os.path.splitext(filename)[0].lower()
                    for key, label in TESS_EMOTION_MAP.items():
                        if f"_{key}" in name_part or name_part.endswith(key):
                            emotion_name = label
                            break

            # --- RAVDESS: files like 03-01-05-01-01-01-01.wav ---
            else:
                parts = os.path.splitext(filename)[0].split("-")
                if len(parts) >= 7:
                    code = parts[2]
                    if code in EMOTION_MAP:
                        emotion_name = EMOTION_MAP[code]

            if emotion_name:
                audio_files.append((file_path, emotion_name))

    total_files = len(audio_files)
    if total_files == 0:
        print("[ERROR] No WAV files found in the dataset directory!")
        sys.exit(1)

    # Count per source
    ravdess_count = sum(1 for _, _ in audio_files if True)  # rough total
    crema_count   = sum(1 for fp, _ in audio_files if os.path.abspath(fp).startswith(os.path.abspath(crema_dir)))
    tess_count    = sum(1 for fp, _ in audio_files if os.path.abspath(fp).startswith(os.path.abspath(tess_dir)))
    esc_count     = total_files - crema_count - tess_count
    print(f"  RAVDESS + ESC-50 : {esc_count} files")
    print(f"  CREMA-D          : {crema_count} files")
    print(f"  TESS             : {tess_count} files")
    print(f"  TOTAL            : {total_files} audio files")
    return audio_files

def train_and_evaluate():
    """
    Main function to load data, preprocess features, train the TensorFlow model,
    evaluate predictions, and save all outputs.
    """
    # 1. Download & Extract RAVDESS Speech Dataset
    download_and_extract_ravdess()

    # 2. Download crying baby samples from ESC-50
    download_crying_dataset()

    # 2b. Download laughing samples from ESC-50
    download_laughing_dataset()

    # 2c. Download TESS Speech dataset from Borealis Dataverse
    download_tess_dataset()

    # Print instructions for additional datasets (CREMA-D) if not present
    print_extra_dataset_instructions()

    # 3. Retrieve dataset files
    audio_files = prepare_dataset()
    
    # 4. Train/Test Split on raw file list to completely prevent data leakage
    labels_raw = [item[1] for item in audio_files]
    train_files, test_files = train_test_split(
        audio_files, test_size=0.2, random_state=42, stratify=labels_raw
    )
    print(f"Train split size: {len(train_files)} files, Test split size: {len(test_files)} files")
    
    # 5. Extract Features for Train Split (with Crying & Laughing augmentation ONLY on training set)
    X_train_list = []
    y_train_list = []
    print("\nExtracting features for the training set (with data augmentation for crying and laughing)...")
    for i, (file_path, emotion_name) in enumerate(train_files):
        try:
            y_audio, sr = load_audio(file_path)
            # Original sample feature extraction
            feat = extract_features(y=y_audio, sr=sr)
            X_train_list.append(feat)
            y_train_list.append(emotion_name)
            
            # Apply 3x augmentation specifically to crying and laughing to balance class representation
            if emotion_name in ["crying", "laughing"]:
                augmented_audios = augment_audio(y_audio, sr)
                for y_aug in augmented_audios:
                    feat_aug = extract_features(y=y_aug, sr=sr)
                    X_train_list.append(feat_aug)
                    y_train_list.append(emotion_name)
        except Exception as e:
            print(f"\nSkipping train file {os.path.basename(file_path)} due to error: {e}")
            
        sys.stdout.write(f"\rTrain prep progress: {i + 1}/{len(train_files)} files")
        sys.stdout.flush()
    print("\nTrain feature extraction completed.")
    
    # 6. Extract Features for Test Split (strictly original, NO augmentation to prevent data leakage)
    X_test_list = []
    y_test_list = []
    print("\nExtracting features for the test set (no augmentation)...")
    for i, (file_path, emotion_name) in enumerate(test_files):
        try:
            feat = extract_features(file_path=file_path)
            X_test_list.append(feat)
            y_test_list.append(emotion_name)
        except Exception as e:
            print(f"\nSkipping test file {os.path.basename(file_path)} due to error: {e}")
            
        sys.stdout.write(f"\rTest prep progress: {i + 1}/{len(test_files)} files")
        sys.stdout.flush()
    print("\nTest feature extraction completed.")
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list)
    y_test = np.array(y_test_list)
    
    # Missing Value Handling
    print("\nValidating data and handling missing values...")
    if np.any(np.isnan(X_train)):
        X_train = np.nan_to_num(X_train, nan=0.0)
    if np.any(np.isinf(X_train)):
        X_train = np.nan_to_num(X_train, posinf=0.0, neginf=0.0)
        
    if np.any(np.isnan(X_test)):
        X_test = np.nan_to_num(X_test, nan=0.0)
    if np.any(np.isinf(X_test)):
        X_test = np.nan_to_num(X_test, posinf=0.0, neginf=0.0)
        
    print(f"Dataset shapes: Train = {X_train.shape}, Test = {X_test.shape}")

    # 7. Preprocessing: Label Encoding and Feature Scaling
    all_possible_labels = list(EMOTION_MAP.values()) + ["crying", "laughing"]
    label_encoder = LabelEncoder()
    label_encoder.fit(all_possible_labels)
    
    y_train_encoded = label_encoder.transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Fit scaler strictly on X_train to prevent leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save label encoder and scaler to model folder
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("Saved label_encoder.pkl and scaler.pkl to model directory.")
    
    X_train = X_train_scaled
    X_test = X_test_scaled
    y_train = y_train_encoded
    y_test = y_test_encoded
    
    # 6. Build TensorFlow Neural Network Model
    # Dense(256, relu) -> BN() -> Dropout(0.3) -> Dense(128, relu) -> BN() -> Dropout(0.3) -> Dense(64, relu) -> Output
    num_classes = len(label_encoder.classes_)
    
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile Model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Define callbacks
    checkpoint_path = os.path.join(MODEL_DIR, "emotion_model.h5")
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 7. Train model
    epochs = 120
    batch_size = 32
    print(f"\nTraining model for up to {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Ensure final weights are also saved
    model.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # 8. Evaluation
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*40)
    print("          EVALUATION RESULTS          ")
    print("="*40)
    print(f"Accuracy:                  {accuracy * 100:.2f}%")
    print(f"Precision (Weighted):      {precision * 100:.2f}%")
    print(f"Recall (Weighted):         {recall * 100:.2f}%")
    print(f"F1 Score (Weighted):       {f1 * 100:.2f}%")
    print("="*40)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # 9. Visualization & Curves
    print("\nGenerating training curves and confusion matrix...")
    
    # Subplots for loss and accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Train Loss', color='#FF6B6B')
    ax1.plot(history.history['val_loss'], label='Val Loss', color='#4D96FF')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Train Acc', color='#FFD93D')
    ax2.plot(history.history['val_accuracy'], label='Val Acc', color='#6BCB77')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    curves_path = os.path.join(MODEL_DIR, "training_curves.png")
    plt.savefig(curves_path)
    plt.close()
    print(f"Saved learning curves to {curves_path}")
    
    # Plot Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")
    
    print("\nTraining completed successfully! Ready for predictions.")

if __name__ == "__main__":
    train_and_evaluate()
