from utils.preprocessing import load_data, split_and_scale, save_metadata
from models.ann_model import train_ann, evaluate_ann, save_model
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Load dataset
X, y = load_data("data/dropout.csv")

# Save feature order
feature_columns = X.columns.tolist()

# Save mean values for optional filling
mean_values = X.mean().to_dict()

# Split + scale
X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = split_and_scale(X, y)

# Train
model = train_ann(X_train_scaled, y_train)

# Evaluate
accuracy = evaluate_ann(model, X_test_scaled, y_test)

print("ANN Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, model.predict(X_test_scaled)))
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test_scaled)))

# Save everything
save_model(model, scaler)
save_metadata(feature_columns, mean_values)