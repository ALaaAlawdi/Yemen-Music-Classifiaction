# app.py

import streamlit as st
import os
import pickle
from preprocessing import preprocess_audio  # Preprocessing script
from features_extraction import process_audio_and_save_features  # Feature extraction script
from prediction import run_all_models  # Prediction script
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import plotly.express as px

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

def plot_waveform(audio_path):
    y, sr = librosa.load(audio_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='white')
    ax.set_title('Waveform', color='red', fontsize=16)
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#000000')  # Black background
    ax.set_facecolor('#000000')         # Black background
    st.pyplot(fig)

def plot_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.stft(y)
    S_dB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='Reds')  # Red colormap
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    ax.set_title('Spectrogram', color='red', fontsize=16)
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Frequency (Hz)', color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('#000000')  # Black background
    ax.set_facecolor('#000000')         # Black background
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Yemeni Music Classification", page_icon="🎵", layout="wide")
    
    # Custom CSS styling with black background and red accents
    st.markdown(
        """
        <style>
        /* Overall background */
        .stApp {
            background-color: #000000; /* Black background */
        }
        /* Text color */
        html, body, [class*="css"]  {
            color: #FFFFFF; /* White text */
        }
        /* Header styling */
        h1 {
            color: #FFFFFF; /* White text */
            text-align: center;
        }
        /* Subheader styling */
        h2 {
            color: #FFFFFF; /* White text */
        }
        /* Button styling */
        .stButton>button {
            color: #FFFFFF; /* White text */
            background-color: #D6001C; /* Red background */
            border-radius: 8px;
            height: 50px;
            width: auto;
            font-size: 18px;
            border: 1px solid #FFFFFF; /* White border */
        }
        /* File uploader */
        .css-1d391kg {  /* Adjust this class to match the file uploader in your version */
            color: #FFFFFF; /* White text */
        }
        /* Table header */
        .css-1ex1afd th {
            background-color: #D6001C; /* Red header */
            color: #FFFFFF; /* White text */
        }
        /* Table rows */
        .css-1ex1afd td {
            color: #FFFFFF; /* White text */
        }
        /* Sidebar (if used) */
        .css-1d391kg {  /* Adjust this class to match the sidebar in your version */
            background-color: #000000;
        }
        /* Spinner text */
        .stSpinner > div > div {
            color: #FFFFFF;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Add the Yemen flag at the top of the page
    st.markdown(
        """
        <div style='text-align: center;'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/8/89/Flag_of_Yemen.svg' alt='Yemen Flag' width='200'>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.title("🎶 Yemeni Music Genre Classification")
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
            
            # Visualizations
            st.subheader("Audio Visualizations")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Waveform**")
                plot_waveform(input_path)
            with col2:
                st.write("**Spectrogram**")
                plot_spectrogram(input_path)
    
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
    
            st.header("📝 Results")
    
            st.subheader("Predictions from Individual Models:")
            results_df = []
            for model, label in cleaned_labels.items():
                weight = get_model_weights().get(model, 0)
                results_df.append({'Model': model, 'Prediction': label, 'Weight': weight})
            results_df = pd.DataFrame(results_df)
            st.table(results_df.style.set_properties(**{'background-color': '#000000', 'color': 'white'}))
    
            st.subheader("Weighted Voting Results:")
            st.write(f"**Final Prediction:** `{final_prediction}`")
            st.write("**Weighted Sums:**")
            weighted_sums_df = pd.DataFrame(list(weighted_sums.items()), columns=['Genre', 'Weighted Sum'])
            fig = px.bar(
                weighted_sums_df,
                x='Genre',
                y='Weighted Sum',
                title='Weighted Sum per Genre',
                text='Weighted Sum',
                color='Genre',
                color_discrete_sequence=px.colors.sequential.Reds,
            )
            fig.update_layout(
                title_font_size=24,
                xaxis_title='',
                yaxis_title='Weighted Sum',
                legend_title_text='Genre',
                plot_bgcolor='#000000',  # Black background
                paper_bgcolor='#000000',
                font=dict(color='#FFFFFF'),  # White text
                xaxis=dict(color='#FFFFFF'),
                yaxis=dict(color='#FFFFFF'),
                legend=dict(font=dict(color='#FFFFFF'))
            )
            st.plotly_chart(fig, use_container_width=True)
    
        else:
            st.error('File type not allowed. Please upload a valid audio file.')
    else:
        st.info('Please upload an audio file to proceed.')
    

if __name__ == '__main__':
    main()
