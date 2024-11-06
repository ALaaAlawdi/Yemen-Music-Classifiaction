# app.py

import streamlit as st
import os
import pickle
from preprocessing import preprocess_audio  # Your preprocessing script
from features_extraction import process_audio_and_save_features  # Your feature extraction script
from prediction import run_all_models  # Your prediction script
import numpy as np
import pandas as pd
import librosa
import librosa.display
import plotly.express as px
import plotly.graph_objects as go

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

def plot_interactive_waveform(audio_path):
    y, sr = librosa.load(audio_path)
    time = np.linspace(0, len(y) / sr, num=len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=y, mode='lines', line=dict(color='white')))
    fig.update_layout(
        title='Waveform',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(color='white'),
        yaxis=dict(color='white'),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_interactive_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    S = np.abs(librosa.stft(y))
    S_dB = librosa.amplitude_to_db(S, ref=np.max)
    fig = go.Figure(data=go.Heatmap(
        z=S_dB,
        x=np.linspace(0, len(y)/sr, S_dB.shape[1]),
        y=librosa.fft_frequencies(sr=sr),
        colorscale='Reds',
        showscale=False
    ))
    fig.update_layout(
        title='Spectrogram',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        yaxis_type='log',
        xaxis=dict(color='white'),
        yaxis=dict(color='white'),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig = go.Figure(data=go.Heatmap(
        z=S_dB,
        x=np.linspace(0, len(y)/sr, S_dB.shape[1]),
        y=librosa.mel_frequencies(n_mels=S_D_B.shape[0], fmin=0, fmax=sr/2),
        colorscale='Viridis',
        showscale=False
    ))
    fig.update_layout(
        title='Mel Spectrogram',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(color='white'),
        yaxis=dict(color='white'),
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_chroma_features(audio_path):
    y, sr = librosa.load(audio_path)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    fig = go.Figure(data=go.Heatmap(
        z=chroma,
        x=np.linspace(0, len(y)/sr, chroma.shape[1]),
        y=['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
        colorscale='Electric',
        showscale=False
    ))
    fig.update_layout(
        title='Chroma Features',
        xaxis_title='Time (s)',
        yaxis_title='Pitch Class',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(color='white'),
        yaxis=dict(color='white'),
    )
    st.plotly_chart(fig, use_container_width=True)

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
        /* Table header */
        .css-1ex1afd th {
            background-color: #D6001C; /* Red header */
            color: #FFFFFF; /* White text */
        }
        /* Table rows */
        .css-1ex1afd td {
            color: #FFFFFF; /* White text */
        }
        /* Spinner text */
        .stSpinner > div > div {
            color: #FFFFFF;
        }
        /* Border around the flag */
        .flag-container {
            display: flex;
            justify-content: center;   /* Centers content horizontally */
            align-items: center;       /* Centers content vertically */
            border: 5px solid #FFFFFF; /* White border */
            margin-bottom: 20px;
            padding: 10px;             /* Optional: Add padding inside the border */
        }
        /* Adjust image inside the container */
        .flag-container img {
            display: block;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Add the Yemen flag at the top of the page with a border
    st.markdown(
        """
        <div class='flag-container'>
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
            with st.spinner('Generating visualizations...'):
                plot_interactive_waveform(input_path)
                plot_interactive_spectrogram(input_path)
                plot_mel_spectrogram(input_path)
                plot_chroma_features(input_path)
    
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
