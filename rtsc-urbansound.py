"""
Filename: rtsc-urbansound.py
"""

import os  # Manejo de archivos y directorios
import numpy as np  # Manejo de arreglos numéricos
import pandas as pd  # Manipulación de datos en formato tabular
import librosa  # Procesamiento de audio
import librosa.display
import sounddevice as sd  # Captura de audio en tiempo real
import tensorflow as tf  # Biblioteca de Machine Learning
from tensorflow.keras.models import load_model  # Para cargar el modelo entrenado
import time  # Manejo de tiempos y pausas
from datetime import datetime  # Obtener fecha y hora actual
import joblib  # Guardar y cargar objetos de Python, como el codificador de etiquetas

# Cargar el modelo entrenado y el codificador de etiquetas
MODEL_PATH = "urban_sound_model.h5"
LABEL_ENCODER_PATH = "label_encoder.pkl"
model = load_model(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Definir las clases de sonidos relevantes para clasificar
TARGET_CLASSES = {'engine_idling', 'street_music', 'car_horn', 'siren', 'dog_bark'}

# Configuración del archivo de Excel donde se guardarán los resultados
EXCEL_FILE = "sound_classification_log.xlsx"


def extract_features(audio, sr=22050, n_mfcc=40):
    """Extrae características MFCC (Mel-Frequency Cepstral Coefficients) del audio."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs.T, axis=0)  # Se obtiene el promedio para reducir dimensiones
    return mfccs


def record_audio(duration=2, sr=22050):
    """Graba audio desde el micrófono por un tiempo determinado."""
    print("Grabando...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Espera a que termine la grabación
    return np.squeeze(audio)  # Se devuelve el audio en una sola dimensión


def classify_audio(audio, sr=22050):
    """Clasifica el audio grabado usando el modelo entrenado."""
    features = extract_features(audio, sr)  # Extrae las características del audio
    features = np.expand_dims(features, axis=0)  # Se le da la forma adecuada para el modelo
    features = np.expand_dims(features, axis=-1)  # Se añade otra dimensión para la CNN
    prediction = model.predict(features)  # Se obtiene la predicción del modelo
    predicted_label = np.argmax(prediction, axis=1)  # Se obtiene la clase con mayor probabilidad
    return label_encoder.inverse_transform(predicted_label)[0]  # Se convierte la predicción a la etiqueta original


def save_to_excel(data):
    """Guarda los resultados de clasificación en un archivo de Excel. Lo crea si no existe."""
    df = pd.DataFrame(data, columns=['Timestamp', 'Classified Sound'])
    
    if not os.path.exists(EXCEL_FILE):
        df.to_excel(EXCEL_FILE, index=False, sheet_name="Sound Log", engine='openpyxl')
    else:
        with pd.ExcelWriter(EXCEL_FILE, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="Sound Log")


def main():
    """Función principal para la clasificación de sonido en tiempo real."""
    print("Iniciando clasificación de sonidos en tiempo real...")
    log_data = []
    try:
        while True:
            # Grabar y clasificar sonido
            audio = record_audio()
            label = classify_audio(audio)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Verificar si el sonido detectado está en las clases de interés
            if label in TARGET_CLASSES:
                print(f"[{timestamp}] Detectado: {label}")
                log_data.append([timestamp, label])
                
            # Guardar en el archivo de Excel cada 10 clasificaciones
            if len(log_data) >= 10:
                save_to_excel(log_data)
                log_data = []
            
            time.sleep(1)  # Pequeña pausa para evitar sobrecarga de procesamiento
    except KeyboardInterrupt:
        print("Deteniendo clasificación...")
        if log_data:
            save_to_excel(log_data)  # Guardar cualquier dato pendiente antes de salir


if __name__ == "__main__":
    main()

