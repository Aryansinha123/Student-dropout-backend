from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_ann(X_train, y_train):
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=600,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_ann(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def save_model(model, scaler):
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, "saved_models/ann_model.pkl")
    joblib.dump(scaler, "saved_models/scaler.pkl")