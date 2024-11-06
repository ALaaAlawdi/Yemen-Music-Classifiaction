from flask import Flask, render_template, request, redirect, url_for
import os
import pickle
from preprocessing import preprocess_audio  # Preprocessing script
from features_extraction import process_audio_and_save_features  # Feature extraction script
from prediction import run_all_models  # Prediction script

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'processed/'
app.config['FEATURES_FOLDER'] = 'features/'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg', 'm4a'}

# Ensure the folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
os.makedirs(app.config['FEATURES_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


import numpy as np

# Step 1: Remove '_10s' from the predicted labels
def clean_predicted_labels(predicted_labels):
    return {model: label.replace('_10s', '') for model, label in predicted_labels.items()}

# Step 2: Assign weights to models
def get_model_weights():
    # Assigning weights based on model reliability (these are sample weights, adjust as needed)
    return {
        'VGG19': 0.25,
        'VGG16': 0.2,
        'AlexNet': 0.15,
        'ResNet50': 0.2,
        'MobileNet': 0.2
    }

# Step 3: Apply the weighted voting equation
def apply_weighted_voting(cleaned_labels, model_weights):
    # Get unique options (e.g., 'snany', 'adny', etc.)
    unique_labels = set(cleaned_labels.values())
    
    # Initialize the weighted sum dictionary for each label
    weighted_sums = {label: 0 for label in unique_labels}
    
    # Loop over each model and its prediction
    for model, label in cleaned_labels.items():
        # w_j: the weight for the model
        weight = model_weights.get(model, 0)
        
        # Add the weight to the corresponding label's weighted sum
        weighted_sums[label] += weight
    
    # Find the label with the highest weighted sum
    final_prediction = max(weighted_sums, key=weighted_sums.get)
    
    return final_prediction, weighted_sums

# Function to handle the full process
def process_predictions(predicted_labels):
    # Step 1: Clean the predicted labels
    cleaned_labels = clean_predicted_labels(predicted_labels)
    
    # Step 2: Get model weights
    model_weights = get_model_weights()
    
    # Step 3: Apply the weighted voting equation
    final_prediction, weighted_sums = apply_weighted_voting(cleaned_labels, model_weights)
    
    return final_prediction, cleaned_labels, weighted_sums


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audiofile' not in request.files:
        return redirect(request.url)
    
    file = request.files['audiofile']

    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = file.filename
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # Step 1: Preprocess the audio
        processed_audio_path = preprocess_audio(input_path, app.config['PROCESSED_FOLDER'])

        # Step 2: Extract features from the preprocessed audio and get the saved file path
        features_file = process_audio_and_save_features(processed_audio_path, app.config['FEATURES_FOLDER'])

        # Step 3: Load label encoder (you should have already trained and saved this)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)

        # Step 4: Run predictions using the extracted features file
        input_shape = (40, 1)  # Adjust this based on your feature extraction process
        experiment_folder = './results/'  # This folder should contain the saved model weights

        predicted_labels = run_all_models(input_shape, features_file, experiment_folder, label_encoder)

        final_prediction, cleaned_labels, weighted_sums = process_predictions(predicted_labels)

        # Pass the data to the template to be rendered
        return render_template('index.html', predicted_labels=cleaned_labels, final_prediction=final_prediction, weighted_sums=weighted_sums)

    return 'File type not allowed. Please upload a valid audio file.'


if __name__ == '__main__':
    app.run(debug=True)
