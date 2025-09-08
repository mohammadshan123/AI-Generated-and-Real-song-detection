1import os
import joblib
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.speech_to_text import audio_to_text  # Ensure this file exists and works

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained LSTM model and tokenizer
model = load_model('model/lstm_model.h5')
tokenizer = joblib.load('model/tokenizer.pkl')

# Homepage: Upload form
@app.route('/')
def upload_page():
    return render_template('upload.html')

# Upload & Predict Route
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')

    if not file or file.filename == '':
        return 'No file uploaded', 400

    # Save the uploaded audio file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Convert audio to text
    lyrics = audio_to_text(filepath)

    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([lyrics])
    padded_sequence = pad_sequences(sequence, maxlen=300)

    # Predict with LSTM model
    prediction = model.predict(padded_sequence)[0][0]
    label = "Real Song" if prediction < 0.5 else "AI-Generated Song"

    return render_template('result.html', result=label)

# Run the app
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)





