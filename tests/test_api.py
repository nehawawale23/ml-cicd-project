# tests/test_api.py

import sys, pickle
import numpy as np
sys.path.insert(0, "app")

# Make a quick model for testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
iris = load_iris()
model = RandomForestClassifier().fit(iris.data, iris.target)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

from predict import app

def test_health_check():
    """Check: does the /health door respond?"""
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"
    print("✅ Health check passed!")

def test_prediction_works():
    """Check: does the /predict door give valid answers?"""
    client = app.test_client()
    response = client.post("/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 200
    assert "flower" in response.json
    print(f"✅ Prediction check passed: {response.json['flower']}")