from flask import Flask, request, render_template
from joblib import load
import os

app = Flask(__name__)

# Cargar el modelo y el scaler
modelo_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'modelo_final_knn.sav')
scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler_knn.sav')

modelo = load(modelo_path)
scaler = load(scaler_path)

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))