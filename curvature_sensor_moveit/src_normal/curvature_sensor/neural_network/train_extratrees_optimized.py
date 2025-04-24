import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
import json
import matplotlib.pyplot as plt

# Load dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "csv_data", "preprocessed", "preprocessed_training_dataset.csv")
df = pd.read_csv(dataset_path)

# Use only active curvature rows
df = df[df["Curvature_Active"] == 1]

# Select features and targets
feature_cols = [col for col in df.columns if col.startswith("FFT_") or "Band" in col or "Ratio" in col]
X = df[feature_cols]
y = df[["Position_cm", "Curvature_Label"]]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Existing code continues here (assuming the original code is below)

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_dist = {
    'estimator__n_estimators': [50, 100, 200],
    'estimator__max_features': ['auto', 'sqrt', 'log2'],
    'estimator__max_depth': [None, 10, 20, 30],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__min_samples_leaf': [1, 2, 4],
    'estimator__bootstrap': [True, False]
}

base_model = ExtraTreesRegressor(random_state=42)
model = MultiOutputRegressor(base_model)

search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=3, random_state=42, n_jobs=-1, verbose=2)
search.fit(X_train, y_train)

best_model = search.best_estimator_

# Evaluate best model
y_pred = best_model.predict(X_test)
rmse_pos = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]) ** 0.5
rmse_curv = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]) ** 0.5
r2_pos = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
r2_curv = r2_score(y_test.iloc[:, 1], y_pred[:, 1])

metrics = {
    "model": "ExtraTrees",
    "rmse_position": rmse_pos,
    "rmse_curvature": rmse_curv,
    "r2_position": r2_pos,
    "r2_curvature": r2_curv,
    "best_params": search.best_params_
}
model_output_dir = os.path.join(base_dir, "models", "extratrees_optimized")
os.makedirs(model_output_dir, exist_ok=True)
with open(os.path.join(model_output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

plt.figure(figsize=(6, 6))
plt.scatter(y_test["Position_cm"], y_pred[:, 0], alpha=0.4)
plt.xlabel("True Position (cm)")
plt.ylabel("Predicted Position (cm)")
plt.title("Position Prediction (ExtraTrees)")
plt.grid(True)
plt.savefig(os.path.join(model_output_dir, "position_plot.png"))
plt.clf()

plt.figure(figsize=(6, 6))
plt.scatter(y_test["Curvature_Label"], y_pred[:, 1], alpha=0.4)
plt.xlabel("True Curvature")
plt.ylabel("Predicted Curvature")
plt.title("Curvature Prediction (ExtraTrees)")
plt.grid(True)
plt.savefig(os.path.join(model_output_dir, "curvature_plot.png"))
plt.clf()

import joblib
joblib.dump(best_model, os.path.join(model_output_dir, "extratrees_model.pkl"))
