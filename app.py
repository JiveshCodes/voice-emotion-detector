import os
import tempfile
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Import prediction functions and emojis
from predict import predict_emotion
from utils import EMOTION_EMOJIS

# Page Configuration
st.set_page_config(
    page_title="AI Voice Emotion Detector",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
# New premium styling with orange, green, grey, white theme and sharp corners
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Global font */
        .stApp, div[data-baseweb="select"] {
            font-family: 'Outfit', sans-serif !important;
        }
        
        /* Background and overall layout */
        .stApp {
            background: #1E1E1E; /* Dark gray background */
            color: #F8F9FA; /* Light text */
        }
        
        /* Header styling */
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FF6B00 0%, #FFB400 60%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.2rem;
            padding-top: 1rem;
            border-radius: 0 !important;
        }
        .subtitle {
            font-size: 1.15rem;
            color: #A0AEC0;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 300;
            border-radius: 0 !important;
        }
        
        /* Card / container styling – sharp corners */
        .result-card {
            background: #252525; /* Slightly lighter than background */
            border: 2px solid #FF6B00; /* Orange border */
            padding: 24px;
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
            border-radius: 0 !important; /* Sharp corners */
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        
        .emotion-badge {
            display: inline-block;
            font-size: 2.5rem;
            font-weight: 800;
            color: #FF6B00; /* Orange for emotion label */
            margin: 10px 0;
        }
        
        .confidence-badge {
            font-size: 1.4rem;
            color: #00FF66; /* Green for confidence */
            font-weight: 600;
            margin-bottom: 15px;
        }
        
        /* Sidebar styling */
        .sidebar-header {
            font-weight: 600;
            font-size: 1.2rem;
            color: #FF6B00;
            margin-top: 1rem;
        }
        
        /* Button styling – orange background, no rounded corners */
        .stButton button {
            background-color: #FF6B00 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 0 !important;
            font-weight: 600;
            padding: 0.5rem 1rem;
        }
        .stButton button:hover {
            background-color: #E55A00 !important;
        }
        
        /* File uploader – remove default border radius */
        div[data-testid="stFileUploader"] {
            border: 2px dashed #FF6B00;
            border-radius: 0 !important;
            padding: 12px;
            background: #2A2A2A;
        }
        
        /* Chart styling – match theme */
        .stPlotlyChart, .stChart {
            border-radius: 0 !important;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in {
            animation: fadeIn 0.6s ease-in-out;
        }
    </style>
""", unsafe_allow_html=True)

# Main Title Layout
st.markdown("<h1 class='main-title'>🎙️ Voice Emotion Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze human emotions in real-time from audio recordings using Deep Learning</p>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("<h2 style='color:#FFD93D; font-weight:800; font-size:1.6rem;'>Project Details</h2>", unsafe_allow_html=True)
st.sidebar.markdown("""
This AI application predicts human emotion from vocal tone and acoustic characteristics (now featuring crying and laughing detection).

### 🧠 Model Architecture
The deep neural network consists of:
- **Input Layer**: Dense (512, ReLU)
- **Normalizers**: Batch Normalization
- **Regularizers**: Dropout (0.4)
- **Hidden Layer 1**: Dense (256, ReLU)
- **Normalizers**: Batch Normalization
- **Regularizers**: Dropout (0.4)
- **Hidden Layer 2**: Dense (128, ReLU)
- **Normalizers**: Batch Normalization
- **Regularizers**: Dropout (0.3)
- **Hidden Layer 3**: Dense (64, ReLU)
- **Normalizers**: Batch Normalization
- **Regularizers**: Dropout (0.2)
- **Output Layer**: Dense (10, Softmax)

### 📊 Audio Features Extracted
Features include both **Mean** and **Standard Deviation (Std)** for:
1. **MFCCs**: 40 Mel-Frequency Cepstral Coefficients (80 features)
2. **Chroma STFT**: 12 Pitch class features (24 features)
3. **Mel Spectrogram**: 128 frequency bands (256 features)
4. **Spectral Contrast**: 7 frequency contrast bands (14 features)
5. **Zero Crossing Rate**: 1 temporal feature (2 features)
6. **RMS Energy**: 1 volume/amplitude feature (2 features)
7. **Spectral Centroid**: 1 spectral texture feature (2 features)
8. **Spectral Rolloff**: 1 frequency rolloff feature (2 features)
9. **Spectral Bandwidth**: 1 spectral bandwidth feature (2 features)
*Total Vector Size: 384 statistical features*

### 📁 Datasets Used
- **RAVDESS Speech Dataset**: 8 core emotions (Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised)
- **ESC-50 Dataset**: Crying and Laughing classes (augmented with pitch-shifting and noise injection)
""")

def check_model_files() -> bool:
    """Helper to check if model is trained before running application."""
    model_dir = "model"
    model_path = os.path.join(model_dir, "emotion_model.h5")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    le_path = os.path.join(model_dir, "label_encoder.pkl")
    return os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(le_path)

def plot_probabilities(probs: dict) -> plt.Figure:
    """Generates a high-quality horizontal bar chart of probabilities."""
    sorted_probs = sorted(probs.items(), key=lambda x: x[1])
    emotions = [x[0].capitalize() for x in sorted_probs]
    values = [x[1] * 100 for x in sorted_probs] # Convert to percentages
    
    # Custom color palette matching the modern UI
    color_map = {
        "Neutral": "#8D99AE",
        "Calm": "#A8DADC",
        "Happy": "#FFD166",
        "Sad": "#457B9D",
        "Angry": "#E63946",
        "Fearful": "#9B5DE5",
        "Disgust": "#4F772D",
        "Surprised": "#F15BB5",
        "Crying": "#0077B6",
        "Laughing": "#FF9F1C"
    }
    colors = [color_map.get(emo, "#1D3557") for emo in emotions]
    
    # Set up dark-styled matplotlib chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#0E1117') # Streamlit background
    ax.set_facecolor('#1E2530') # Dark plot card background
    
    # Create horizontal bar chart
    bars = ax.barh(emotions, values, color=colors, edgecolor='none', height=0.6)
    
    # Add values text onto or next to bars
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height()/2,
            f'{width:.1f}%',
            ha='left',
            va='center',
            color='#E0E0E0',
            fontsize=10,
            fontweight='bold'
        )
        
    ax.set_xlim(0, 115) # Leave room for text labels
    
    # Style axes, remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#4A5568')
    ax.spines['bottom'].set_color('#4A5568')
    ax.tick_params(colors='#E0E0E0', labelsize=11)
    ax.set_xlabel("Probability (%)", color='#E0E0E0', fontweight='bold', labelpad=10)
    ax.set_title("Emotion Probability Distribution", color='#F8F9FA', fontsize=13, pad=15, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Check if model has been trained yet
if not check_model_files():
    st.error("⚠️ Model Artifacts Not Found!")
    st.markdown("""
    The application cannot run because the model has not been trained yet.
    
    Please run the training pipeline first to train the network and save the artifacts:
    
    ```bash
    python train.py
    ```
    
    *If you are running on Google Colab, execute the cell containing the training command.*
    """)
else:
    # File Uploader Section
    st.markdown("### Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio recording (WAV or MP3 format):",
        type=["wav", "mp3"],
        help="Upload a WAV or MP3 audio file of a human voice."
    )

    if uploaded_file is not None:
        # Audio Player Widget
        st.write("🎵 **Audio Playback:**")
        st.audio(uploaded_file)
        
        # Action button
        analyze_button = st.button("🔍 Analyze Emotion", use_container_width=True)
        
        if analyze_button:
            # Create a spinner during the processing
            with st.spinner("Processing audio file and extracting features..."):
                try:
                    # Save the uploaded file to a temporary file, preserving original extension
                    _, file_extension = os.path.splitext(uploaded_file.name)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_file_path = temp_file.name
                    
                    # Run prediction
                    emotion, confidence, probs = predict_emotion(temp_file_path)
                    
                    # Remove temporary file
                    os.remove(temp_file_path)
                    
                    # Show results in card layout
                    emoji = EMOTION_EMOJIS.get(emotion, "")
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="font-size:1.1rem; color:#A0AEC0; font-weight:400; text-transform:uppercase; letter-spacing:1px;">Analysis Complete</div>
                        <div class="emotion-badge">{emoji} {emotion.capitalize()}</div>
                        <div class="confidence-badge">Confidence: {confidence * 100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display probabilities visualization
                    st.markdown("### Probability Distribution")
                    chart_fig = plot_probabilities(probs)
                    st.pyplot(chart_fig)
                    
                except Exception as e:
                    st.error(f"An error occurred during audio processing: {str(e)}")
                    st.info("Ensure the uploaded file is a valid, uncorrupted mono/stereo WAV audio file.")
