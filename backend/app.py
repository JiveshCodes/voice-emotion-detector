import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from predict import predict_emotion

app = Flask(__name__)
# Enable CORS for the application
CORS(app)

# Limit file uploads to 50MB
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Set up uploads directory inside backend
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """
    Serves the static React-like dashboard UI.
    """
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Expects a multipart/form-data request with an 'file' key containing WAV/MP3 audio.
    Returns prediction results in JSON format.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for upload"}), 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only WAV and MP3 formats are supported."}), 400
        
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        # Save file locally
        file.save(temp_path)
        
        # Run local prediction
        emotion, confidence, probabilities = predict_emotion(temp_path)
        
        # Format response
        response = {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probabilities
        }
        return jsonify(response), 200
        
    except Exception as e:
        app.logger.error(f"Prediction failure: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500
        
    finally:
        # Ensure temporary file is deleted
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                app.logger.warning(f"Failed to remove temp file {temp_path}: {str(e)}")

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "File size exceeds the limit of 50MB."}), 413

if __name__ == '__main__':
    # Local host and port 5000 as requested
    app.run(host='127.0.0.1', port=5000, debug=True)
