# import os
# import librosa
# import soundfile as sf
# import numpy as np

# # Function to ensure the output directory exists
# def ensure_dir(directory):
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# # Function to save audio segments
# def save_audio_segment(y, sr, output_path):
#     sf.write(output_path, y, sr)

# # Function to remove silence from an audio file
# def remove_silence(y, sr):
#     y_trimmed, _ = librosa.effects.trim(y)
#     return y_trimmed

# # Function to normalize the audio signal
# def normalize_audio(y):
#     return librosa.util.normalize(y)

# # Function to trim or extend an audio file to a specific duration (e.g., 10 seconds)
# def trim_or_extend_audio(y, sr, target_duration=10):
#     current_duration = librosa.get_duration(y=y, sr=sr)
#     if current_duration >= target_duration:
#         y_segment = y[:int(target_duration * sr)]
#     else:
#         # Extend the audio by repeating itself if it's shorter than the target duration
#         y_segment = np.tile(y, int(np.ceil(target_duration / current_duration)))[:int(target_duration * sr)]
#     return y_segment

# # Main function to preprocess a single audio file
# def preprocess_audio(audio_path, output_dir, sr=22050):
#     # Load the audio file
#     y, sr = librosa.load(audio_path, sr=sr)

#     # Remove silence
#     y = remove_silence(y, sr)

#     # Normalize the audio
#     y = normalize_audio(y)

#     # Trim or extend to 10 seconds
#     y_segment = trim_or_extend_audio(y, sr, target_duration=10)

#     # Ensure the output directory exists
#     ensure_dir(output_dir)

#     # Save the processed audio file
#     output_path = os.path.join(output_dir, f"{os.path.basename(audio_path).split('.')[0]}_processed.wav")
#     save_audio_segment(y_segment, sr, output_path)

#     print(f"Processed audio saved at: {output_path}")

# # Example usage:
# # Provide the path to your audio file and the output directory
# audio_path = "./snany.wav"
# output_dir = "./"

# preprocess_audio(audio_path, output_dir)
import os
import librosa
import soundfile as sf
import numpy as np
import argparse

# Function to ensure the output directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to save audio segments
def save_audio_segment(y, sr, output_path):
    sf.write(output_path, y, sr)

# Function to remove silence from an audio file
def remove_silence(y, sr):
    y_trimmed, _ = librosa.effects.trim(y)
    return y_trimmed

# Function to normalize the audio signal
def normalize_audio(y):
    return librosa.util.normalize(y)

# Function to trim or extend an audio file to a specific duration (e.g., 10 seconds)
def trim_or_extend_audio(y, sr, target_duration=10):
    current_duration = librosa.get_duration(y=y, sr=sr)
    if current_duration >= target_duration:
        y_segment = y[:int(target_duration * sr)]
    else:
        # Extend the audio by repeating itself if it's shorter than the target duration
        y_segment = np.tile(y, int(np.ceil(target_duration / current_duration)))[:int(target_duration * sr)]
    return y_segment



# Main function to preprocess a single audio file
def preprocess_audio(audio_path, output_dir, sr=22050, target_duration=10):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr)

    # Remove silence
    y = remove_silence(y, sr)

    # Normalize the audio
    y = normalize_audio(y)

    # Trim or extend to the target duration
    y_segment = trim_or_extend_audio(y, sr, target_duration=target_duration)

    # Ensure the output directory exists
    ensure_dir(output_dir)

    # Save the processed audio file
    output_path = os.path.join(output_dir, f"{os.path.basename(audio_path).split('.')[0]}_processed.wav")
    save_audio_segment(y_segment, sr, output_path)

    print(f"Processed audio saved at: {output_path}")
    return output_path

    

# Command-line argument parsing and script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess an audio file by removing silence, normalizing, and trimming/extending to a target duration.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input audio file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the preprocessed audio file.")
    parser.add_argument('--sr', type=int, default=22050, help="Sampling rate for the audio file. Default is 22050.")
    parser.add_argument('--target_duration', type=int, default=10, help="Target duration (in seconds) to trim or extend the audio file. Default is 10 seconds.")

    args = parser.parse_args()

    # Call the preprocess_audio function with the parsed arguments
    preprocess_audio(args.input, args.output_dir, sr=args.sr, target_duration=args.target_duration)