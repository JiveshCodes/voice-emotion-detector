# Voice Emotion Detector 🎙️🤖

A complete, end-to-end Deep Learning application that predicts human emotions from voice recordings. The model extracts acoustic features using `librosa` and processes them through a multi-layer Neural Network built with `TensorFlow/Keras`. The project comes with a modern `Streamlit` user interface to record, play, and analyze voice emotions with live visualizations.

---

## 📁 Project Architecture

```text
VoiceEmotionDetector/
│
├── dataset/                     # Directory for RAVDESS audio dataset (auto-downloaded)
│
├── model/                       # Model artifacts folder (created after training)
│   ├── emotion_model.h5         # Trained Keras neural network model
│   ├── scaler.pkl               # Fitted StandardScaler object
│   ├── label_encoder.pkl        # Fitted LabelEncoder object
│   ├── training_curves.png      # Training loss & accuracy plots
│   └── confusion_matrix.png     # Evaluation confusion matrix plot
│
├── train.py                     # Main pipeline: download dataset, extract features, train and evaluate model
├── predict.py                   # Prediction script: run inference on a single audio file via CLI
├── app.py                       # Streamlit UI: web-based file upload, playback, and visual analytics
├── utils.py                     # Utility functions: audio loading, feature engineering, and mappings
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation (this file)
```

---

## 🧠 Model Architecture & Features

### 🎧 Acoustic Features (189 features total)
We extract the following spectral, temporal, and energy features from raw audio:
*   **MFCCs (Mel-Frequency Cepstral Coefficients)**: 40 coefficients capturing vocal tract characteristics.
*   **Chroma Features**: 12 pitch classes representing harmonic content.
*   **Mel Spectrogram**: 128 Mel bands capturing sound intensity across frequencies.
*   **Spectral Contrast**: 7 bands representing sound texture and spectral peaks/valleys.
*   **Zero Crossing Rate (ZCR)**: 1 feature measuring how rapidly the signal changes sign (indicator of noise/voiceless sounds).
*   **RMS Energy**: 1 feature capturing root-mean-square amplitude (loudness/energy).

### 🕸️ Neural Network Design
The network uses the following sequential architecture:
1.  **Input Layer**: Dense (256 units, ReLU activation, shape=189)
2.  **Batch Normalization & Dropout (30%)**: For stabilization and regularization
3.  **Hidden Layer 1**: Dense (128 units, ReLU activation)
4.  **Batch Normalization & Dropout (30%)**: Further generalization defense
5.  **Hidden Layer 2**: Dense (64 units, ReLU activation)
6.  **Output Layer**: Dense (8 units, Softmax activation) for multi-class probability output.

*   **Loss Function**: `sparse_categorical_crossentropy`
*   **Optimizer**: `adam`
*   **Callbacks**: `EarlyStopping` (patience=20), `ModelCheckpoint` (saves best model), `ReduceLROnPlateau` (halves learning rate on validation plateau).

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure you have Python 3.8 to 3.11 installed. Create a virtual environment (recommended) and install the dependencies:

```bash
# Clone or navigate to the directory
cd VoiceEmotionDetector

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup
The project uses the **RAVDESS Speech Emotion Dataset** (Ryerson Audio-Visual Database of Emotional Speech and Song) featuring 8 core emotions:
`Neutral`, `Calm`, `Happy`, `Sad`, `Angry`, `Fearful`, `Disgust`, and `Surprised`.

*   **Automatic Setup (Recommended)**: The `train.py` script checks if the dataset is present. If not, it will automatically download the 248MB dataset from Zenodo and extract it directly into the `dataset/` folder.
*   **Manual Setup**: Download `Audio_Speech_Actors_01-24.zip` from Zenodo, extract it, and place the 24 folders (`Actor_01` to `Actor_24`) inside the `dataset/` directory.

### 3. Model Training & Evaluation
To run feature extraction, train the neural network, and generate metric visualizations:

```bash
python train.py
```
This script will:
1.  Verify/download the RAVDESS dataset.
2.  Extract 189 features from all 1440 audio clips.
3.  Split, scale, and encode the data.
4.  Train the model and save artifacts to `model/`.
5.  Print evaluation statistics: accuracy, precision, recall, F1 score, confusion matrix, and a classification report.
6.  Export learning curves and confusion matrix charts to `model/`.

### 4. Running the Web Application
Launch the responsive Streamlit user interface to run predictions on audio recordings:

```bash
streamlit run app.py
```
Open the provided browser link (usually `http://localhost:8501`). Drag and drop or upload any `.wav` audio recording, review playback, and click **Analyze Emotion** to see:
*   The predicted emotion accompanied by its emotional emoji.
*   The model's classification confidence.
*   An interactive visual probability distribution of all emotions.

### 5. CLI Single Inference (Optional)
To test prediction directly from your command-line interface:

```bash
python predict.py <path_to_audio_file.wav>
```

---

## 📈 Example Output

### Command Line Evaluation
When `train.py` completes, it outputs performance summaries like:
```text
========================================
          EVALUATION RESULTS          
========================================
Accuracy:                  81.25%
Precision (Weighted):      81.94%
Recall (Weighted):         81.25%
F1 Score (Weighted):       80.98%
========================================

Classification Report:
              precision    recall  f1-score   support

       angry       0.85      0.82      0.84        38
        calm       0.89      0.87      0.88        38
     disgust       0.79      0.74      0.76        38
     fearful       0.75      0.79      0.77        38
       happy       0.78      0.74      0.76        38
     neutral       0.86      0.81      0.83        19
         sad       0.74      0.84      0.79        38
   surprised       0.84      0.84      0.84        38

    accuracy                           0.81       285
   macro avg       0.81      0.81      0.81       285
weighted avg       0.81      0.81      0.81       285
```

### Streamlit UI Result Card
```text
+------------------------------------------+
|  ANALYSIS COMPLETE                       |
|                                          |
|  😊 Happy                                |
|  Confidence: 92.3%                       |
+------------------------------------------+
```
*(Also displays a horizontal bar chart displaying probabilities for all 8 emotions).*

---

## 📓 Google Colab Setup
To run this on Google Colab:
1.  Upload the directory files to your Google Drive or upload them directly to the Colab environment.
2.  Install dependencies:
    ```python
    !pip install -r requirements.txt
    ```
3.  Train the model:
    ```python
    !python train.py
    ```
4.  To launch Streamlit on Colab, you can use `local tunneling` (e.g. `localtunnel` or `ngrok` or Streamlit's sharing):
    ```python
    !streamlit run app.py & npx localtunnel --port 8501
    ```
