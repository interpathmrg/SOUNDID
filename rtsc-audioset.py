import tensorflow_hub as hub
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import time
from datetime import datetime

# Load YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy().decode('utf-8')
class_names = pd.read_csv(class_map_path)['display_name'].tolist()

# Define relevant classes
TARGET_CLASSES = {'Motorcycle', 'Car', 'Truck', 'Music', 'Electric generator'}

# Excel file setup
EXCEL_FILE = "sound_classification_log.xlsx"


def record_audio(duration=1, sr=16000):
    """Records audio from the microphone."""
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)


def classify_audio(audio, sr=16000):
    """Classify the recorded audio using YAMNet."""
    # Ensure correct shape for YAMNet (flatten to 1D array)
    if len(audio.shape) > 1:
        audio = np.squeeze(audio)
    
    # Ensure the audio length is at least 16000 samples (1 second)
    if len(audio) < sr:
        pad = sr - len(audio)
        audio = np.pad(audio, (0, pad), mode='constant')
    
    scores, embeddings, spectrogram = model(audio)
    mean_scores = np.mean(scores.numpy(), axis=0)
    top_class_index = np.argmax(mean_scores)
    top_class_name = class_names[top_class_index]
    return top_class_name


def save_to_excel(data):
    """Save classification results to an Excel file."""
    df = pd.DataFrame(data, columns=['Timestamp', 'Classified Sound'])
    with pd.ExcelWriter(EXCEL_FILE, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="Sound Log")


def main():
    """Main function to run real-time sound classification."""
    print("Starting real-time sound classification...")
    log_data = []
    try:
        while True:
            # Record and classify sound
            audio = record_audio()
            label = classify_audio(audio)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Check if it belongs to target classes
            if label in TARGET_CLASSES:
                print(f"[{timestamp}] Detected: {label}")
                log_data.append([timestamp, label])
                
            # Save periodically
            if len(log_data) >= 10:
                save_to_excel(log_data)
                log_data = []
            
            time.sleep(1)  # Avoid excessive processing
    except KeyboardInterrupt:
        print("Stopping classification...")
        if log_data:
            save_to_excel(log_data)


if __name__ == "__main__":
    main()
