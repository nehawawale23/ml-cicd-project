# tests/test_model.py

import sys
sys.path.insert(0, "app")
from model import train_model

def test_model_accuracy():
    """Check: is the model accurate enough?"""
    _, accuracy = train_model()
    assert accuracy >= 0.85, f"❌ Accuracy too low: {accuracy}"
    print(f"✅ Accuracy check passed: {accuracy:.2f}")

def test_model_output():
    """Check: does the model give valid predictions?"""
    import pickle, numpy as np
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample)

    assert prediction[0] in [0, 1, 2], "❌ Invalid prediction!"
    print(f"✅ Output check passed: predicted flower {prediction[0]}")