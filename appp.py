# app.py

import streamlit as st
import os
import pickle
from preprocessing import preprocess_audio  # Preprocessing script
from features_extraction import process_audio_and_save_features  # Feature extraction script
from prediction import run_all_models  # Prediction script
import numpy as np

# Configuration
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
FEATURES_FOLDER = 'features/'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(FEATURES_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Step 1: Remove '_10s' from the predicted labels
def clean_predicted_labels(predicted_labels):
    return {model: label.replace('_10s', '') for model, label in predicted_labels.items()}

# Step 2: Assign weights to models
def get_model_weights():
    # Assigning weights based on model reliability (adjust as needed)
    return {
        'VGG19': 0.25,
        'VGG16': 0.2,
        'AlexNet': 0.15,
        'ResNet50': 0.2,
        'MobileNet': 0.2
    }

# Step 3: Apply the weighted voting equation
def apply_weighted_voting(cleaned_labels, model_weights):
    # Get unique labels
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

def main():
    st.title("Yemeni Music Classification")
    st.write("Upload an audio file to classify its Yemeni music genre.")

    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'm4a'])

    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
            filename = uploaded_file.name
            input_path = os.path.join(UPLOAD_FOLDER, filename)

            # Save the uploaded file
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            st.audio(uploaded_file, format='audio/' + uploaded_file.name.split('.')[-1])

            with st.spinner('Processing...'):
                # Step 1: Preprocess the audio
                processed_audio_path = preprocess_audio(input_path, PROCESSED_FOLDER)

                # Step 2: Extract features from the preprocessed audio and get the saved file path
                features_file = process_audio_and_save_features(processed_audio_path, FEATURES_FOLDER)

                # Step 3: Load label encoder
                with open('label_encoder.pkl', 'rb') as f:
                    label_encoder = pickle.load(f)

                # Step 4: Run predictions using the extracted features file
                input_shape = (40, 1)  # Adjust this based on your feature extraction process
                experiment_folder = './results/'  # This folder should contain the saved model weights

                predicted_labels = run_all_models(input_shape, features_file, experiment_folder, label_encoder)

                final_prediction, cleaned_labels, weighted_sums = process_predictions(predicted_labels)

            st.success('Processing complete!')

            st.header("Results")

            st.subheader("Predictions from Individual Models:")
            for model, label in cleaned_labels.items():
                st.write(f"**{model}:** {label}")

            st.subheader("Weighted Voting Results:")
            st.write(f"**Final Prediction:** {final_prediction}")
            st.write("**Weighted Sums:**")
            for label, weight in weighted_sums.items():
                st.write(f"{label}: {weight}")

        else:
            st.error('File type not allowed. Please upload a valid audio file.')
    else:
        st.info('Please upload an audio file to proceed.')

if __name__ == '__main__':
    main()
