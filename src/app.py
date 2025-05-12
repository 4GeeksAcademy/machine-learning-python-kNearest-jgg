from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# Cargar el modelo y el scaler
modelo = load('models/modelo_final_knn.sav')
scaler = load('models/scaler_knn.sav')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convertir los inputs en float y escalar
        features = [float(request.form[f'feature{i}']) for i in range(1, 12)]
        features_scaled = scaler.transform([features])
        pred = modelo.predict(features_scaled)[0]

        if pred == 0:
            resultado = "Este vino probablemente sea de baja calidad 🍷"
        elif pred == 1:
            resultado = "Este vino probablemente sea de calidad media 🍷"
        else:
            resultado = "Este vino probablemente sea de alta calidad 🍷"
    except Exception as e:
        resultado = f"Error en la predicción: {e}"

    return render_template('index.html', resultado=resultado)