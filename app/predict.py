# app/predict.py

import pickle
import numpy as np
from flask import Flask, request, jsonify

# Create the web app
app = Flask(__name__)

# Load our saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Door 1: Health check
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

# Door 2: Prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("features")
    prediction = model.predict([data])
    flower_names = ["Setosa", "Versicolor", "Virginica"]
    return jsonify({
        "prediction": int(prediction[0]),
        "flower": flower_names[int(prediction[0])]
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)