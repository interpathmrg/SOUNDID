Prueba de concepto de clasificación de sonidos usando un modelo de red neuronal convolucional (CNN) 
Por:  Miguel González con ayuda de ChatGPT 4o

La muestra de sonidos para entrenamiento se descargó de aquí:
wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
tar -xvzf UrbanSound8K.tar.gz

Los audios se encuentran en el directorio de audios
La metadata se encuentra en metadata/UrbanSound8K.csv

El archivo: trainer-urbansound.py debe entrenar el modelo y producir dos archivos:
1)- El modelo entrenado = "urban_sound_model.h5"
2)- El codificador de las etiquetas = "label_encoder.pkl"

Estos dos archivos son el insumo para el script rtsc-urbansound.py
que graba los sonidos del ambiente desde el mic, los analiza, los etiqueta 
de acuerdo a nuestro requerimiento {'engine_idling', 'street_music', 'car_horn', 'siren', 'dog_bark'}
y los va registrando en el archivo sound_classification_log.xlsx