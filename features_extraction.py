# import os
# import librosa
# import numpy as np
# from tqdm import tqdm

# # Ensure the output directory exists
# def ensure_dir(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# # Function to extract audio features
# def extract_audio_features(audio_path, target_duration=10, sr=22050):
#     # Load the audio file
#     y, sr = librosa.load(audio_path, sr=sr)
    
#     # Trim or extend to the target duration (default: 10s)
#     current_duration = librosa.get_duration(y=y, sr=sr)
#     if current_duration > target_duration:
#         y = y[:int(target_duration * sr)]
#     elif current_duration < target_duration:
#         # Extend by repeating if shorter than the target duration
#         y = np.tile(y, int(np.ceil(target_duration / current_duration)))[:int(target_duration * sr)]

#     # Extract features (MFCCs, chroma, spectral contrast, tonnetz, ZCR, RMSE)
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
#     spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
#     tonnetz = librosa.feature.tonnetz(y=y, sr=sr).mean(axis=1)
#     zcr = librosa.feature.zero_crossing_rate(y).mean()
#     rmse = librosa.feature.rms(y=y).mean()

#     # Combine all features into a single vector
#     features = np.hstack([mfccs, chroma, spectral_contrast, tonnetz, zcr, rmse])
    
#     return features

# # Main function to process and save features from a single audio file
# def process_audio_and_save_features(audio_path, output_dir, target_duration=10, sr=22050):
#     # Ensure the output directory exists
#     ensure_dir(output_dir)
    
#     # Extract features from the audio file
#     features = extract_audio_features(audio_path, target_duration=target_duration, sr=sr)
    
#     # Save the features to a file
#     output_file = os.path.join(output_dir, f"{os.path.basename(audio_path).split('.')[0]}_features.npy")
#     np.save(output_file, features)
    
#     print(f"Features saved at: {output_file}")
    
#     return features

# # Example usage
# audio_path = "./snany_processed.wav"
# output_dir = "./"
# process_audio_and_save_features(audio_path, output_dir)





import os
import librosa
import numpy as np
from tqdm import tqdm
import argparse

# Ensure the output directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to extract audio features
def extract_audio_features(audio_path, target_duration=10, sr=22050):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Trim or extend to the target duration (default: 10s)
    current_duration = librosa.get_duration(y=y, sr=sr)
    if current_duration > target_duration:
        y = y[:int(target_duration * sr)]
    elif current_duration < target_duration:
        # Extend by repeating if shorter than the target duration
        y = np.tile(y, int(np.ceil(target_duration / current_duration)))[:int(target_duration * sr)]

    # Extract features (MFCCs, chroma, spectral contrast, tonnetz, ZCR, RMSE)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr).mean(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    rmse = librosa.feature.rms(y=y).mean()

    # Combine all features into a single vector
    features = np.hstack([mfccs, chroma, spectral_contrast, tonnetz, zcr, rmse])
    
    return features

# # Main function to process and save features from a single audio file
# def process_audio_and_save_features(audio_path, output_dir, target_duration=10, sr=22050):
#     # Ensure the output directory exists
#     ensure_dir(output_dir)
    
#     # Extract features from the audio file
#     features = extract_audio_features(audio_path, target_duration=target_duration, sr=sr)
    
#     # Save the features to a file
#     output_file = os.path.join(output_dir, f"{os.path.basename(audio_path).split('.')[0]}_features.npy")
#     np.save(output_file, features)
    
#     print(f"Features saved at: {output_file}")
    
#     return features


# Main function to process and save features from a single audio file
def process_audio_and_save_features(audio_path, output_dir, target_duration=10, sr=22050):
    # Ensure the output directory exists
    ensure_dir(output_dir)
    
    # Extract features from the audio file
    features = extract_audio_features(audio_path, target_duration=target_duration, sr=sr)
    
    # Save the features to a file
    output_file = os.path.join(output_dir, f"{os.path.basename(audio_path).split('.')[0]}_features.npy")
    np.save(output_file, features)
    
    print(f"Features saved at: {output_file}")
    
    # Return the path to the saved features file instead of the features array
    return output_file



# Command-line argument parsing and script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract audio features (MFCCs, chroma, spectral contrast, etc.) and save them as .npy files.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input processed audio file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the extracted features.")
    parser.add_argument('--sr', type=int, default=22050, help="Sampling rate for the audio file. Default is 22050.")
    parser.add_argument('--target_duration', type=int, default=10, help="Target duration (in seconds) to trim or extend the audio file. Default is 10 seconds.")

    args = parser.parse_args()

    # Process the audio file and extract features
    process_audio_and_save_features(args.input, args.output_dir, sr=args.sr, target_duration=args.target_duration)
