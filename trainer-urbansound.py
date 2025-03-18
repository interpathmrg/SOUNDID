"""
Filename: trainer-urbansound.py
By: Miguel González
"""

import os  # Manejo de archivos y directorios
import numpy as np  # Manejo de arreglos numéricos
import pandas as pd  # Manipulación de datos en formato tabular
import librosa  # Procesamiento de audio
import librosa.display
import tensorflow as tf  # Biblioteca de Machine Learning
from tensorflow.keras.models import Sequential  # Tipo de modelo en Keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D  # Capas para la CNN
from sklearn.preprocessing import LabelEncoder  # Para convertir etiquetas categóricas a numéricas
from sklearn.model_selection import train_test_split  # Para dividir el dataset en entrenamiento y prueba
import joblib  # Guardar y cargar objetos de Python, como el codificador de etiquetas

# Ruta del dataset
DATASET_PATH = "../data/UrbanSound8K/"
# Cargar el archivo de metadatos que contiene información sobre las clases y archivos de audio
metadata = pd.read_csv(os.path.join(DATASET_PATH, "metadata/UrbanSound8K.csv"))


def extract_features(file_path, sr=22050, n_mfcc=40):
    """Extrae características MFCC (Mel-Frequency Cepstral Coefficients) de un archivo de audio."""
    audio, sr = librosa.load(file_path, sr=sr)  # Cargar archivo de audio
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)  # Extraer MFCCs
    return np.mean(mfccs.T, axis=0)  # Promediar para reducir dimensiones

# Cargar los datos de audio y sus etiquetas
X, y = [], []
for index, row in metadata.iterrows():
    file_path = os.path.join(DATASET_PATH, "audio", f"fold{row['fold']}", row['slice_file_name'])
    if os.path.exists(file_path):
        features = extract_features(file_path)
        X.append(features)
        y.append(row['class'])

# Convertir a arreglos de NumPy
X = np.array(X)
y = np.array(y)

# Codificar las etiquetas en números
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Guardar el codificador de etiquetas
joblib.dump(label_encoder, "label_encoder.pkl")

# Dividir el conjunto de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construcción del modelo de red neuronal convolucional (CNN)
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(40, 1)),  # Primera capa convolucional
    MaxPooling1D(pool_size=2),  # Capa de agrupamiento para reducir dimensiones
    Conv1D(128, kernel_size=3, activation='relu'),  # Segunda capa convolucional
    MaxPooling1D(pool_size=2),  # Otra capa de agrupamiento
    Flatten(),  # Aplanar la salida para conectarla a capas densas
    Dense(256, activation='relu'),  # Capa completamente conectada
    Dropout(0.3),  # Regularización para evitar sobreajuste
    Dense(len(np.unique(y)), activation='softmax')  # Capa de salida con activación softmax
])

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Preparar los datos para que sean compatibles con la CNN
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Guardar el modelo entrenado
model.save("urban_sound_model.h5")

print("¡Entrenamiento del modelo completo y guardado!")
