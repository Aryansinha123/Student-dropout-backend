import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()

    target_column = "target"

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return X, y

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def save_metadata(feature_columns, mean_values):
    os.makedirs("saved_models", exist_ok=True)

    joblib.dump(feature_columns, "saved_models/feature_columns.pkl")
    joblib.dump(mean_values, "saved_models/feature_means.pkl")