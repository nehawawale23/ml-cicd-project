# app/model.py

import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_model():
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Model trained! Accuracy: {accuracy:.2f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model, accuracy

if __name__ == "__main__":
    train_model()