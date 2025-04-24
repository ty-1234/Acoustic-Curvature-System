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
dataset_path = os.path.join(base_dir, "csv_data", "preprocessed", "withpos_preprocessed_training_dataset.csv")
df = pd.read_csv(dataset_path)


# Use only active curvature rows
df = df[df["Curvature_Active"] == 1]

# FFT peak index feature
import re
fft_cols = [col for col in df.columns if col.startswith("FFT_")]
df["FFT_Peak_Index"] = (
    df[fft_cols]
    .idxmax(axis=1)
    .str.extract(r"(\d+)")
    .astype(float)
    .fillna(-1)
    .astype(int)
)

# Select features
fft_cols = [col for col in df.columns if col.startswith("FFT_")]
pos_cols = ["PosX", "PosY", "PosZ"]

# Curvature is predicted from FFT features
curvature_features = df[fft_cols + [col for col in df.columns if "Band" in col or "Ratio" in col or col == "FFT_Peak_Index"]]

# Position is predicted from PosX, PosY, PosZ
position_features = df[pos_cols]

# Scale features
scaler_pos = MinMaxScaler()
scaler_curv = MinMaxScaler()
X_pos_scaled = scaler_pos.fit_transform(position_features)
X_curv_scaled = scaler_curv.fit_transform(curvature_features)

# Define separate outputs
y_pos = df[["Position_cm"]]
y_curv = df[["Curvature_Label"]]

# GroupKFold split using RunID to avoid temporal leakage
from sklearn.model_selection import GroupKFold

groups = df["RunID"]
gkf = GroupKFold(n_splits=2)

train_idx, test_idx = next(gkf.split(X_pos_scaled, y_pos, groups))

X_train_pos, X_test_pos = X_pos_scaled[train_idx], X_pos_scaled[test_idx]
y_train_pos, y_test_pos = y_pos.iloc[train_idx], y_pos.iloc[test_idx]

X_train_curv, X_test_curv = X_curv_scaled[train_idx], X_curv_scaled[test_idx]
y_train_curv, y_test_curv = y_curv.iloc[train_idx], y_curv.iloc[test_idx]

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

param_dist = {
    'estimator__n_estimators': [100, 200, 300, 500, 800, 1000],
    'estimator__max_features': ['sqrt', 'log2', None],
    'estimator__max_depth': [10, 20, 30, 40, 50, None],
    'estimator__min_samples_split': [2, 4, 6, 8, 10],
    'estimator__min_samples_leaf': [1, 2, 3, 4, 5],
    'estimator__bootstrap': [True, False]
}

# Train separate models for curvature and position
model_curv = RandomizedSearchCV(
    MultiOutputRegressor(ExtraTreesRegressor(random_state=42)),
    param_distributions=param_dist,
    n_iter=60,
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=2
)
model_curv.fit(X_train_curv, y_train_curv)

model_pos = RandomizedSearchCV(
    MultiOutputRegressor(ExtraTreesRegressor(random_state=42)),
    param_distributions=param_dist,
    n_iter=60,
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=2
)
model_pos.fit(X_train_pos, y_train_pos)

# Predictions
y_pred_pos = model_pos.predict(X_test_pos)
y_pred_curv = model_curv.predict(X_test_curv)

# Metrics
rmse_pos = mean_squared_error(y_test_pos, y_pred_pos) ** 0.5
rmse_curv = mean_squared_error(y_test_curv, y_pred_curv) ** 0.5
r2_pos = r2_score(y_test_pos, y_pred_pos)
r2_curv = r2_score(y_test_curv, y_pred_curv)

metrics = {
    "model_position": "ExtraTrees",
    "rmse_position": rmse_pos,
    "r2_position": r2_pos,
    "best_params_position": model_pos.best_params_,
    "model_curvature": "ExtraTrees",
    "rmse_curvature": rmse_curv,
    "r2_curvature": r2_curv,
    "best_params_curvature": model_curv.best_params_
}
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_output_dir = os.path.join(base_dir, "models", f"WithPosextratrees_optimized_{timestamp}")
os.makedirs(model_output_dir, exist_ok=True)
with open(os.path.join(model_output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

plt.figure(figsize=(6, 6))
plt.scatter(y_test_pos["Position_cm"], y_pred_pos[:, 0], alpha=0.4)
plt.xlabel("True Position (cm)")
plt.ylabel("Predicted Position (cm)")
plt.title("Position Prediction (ExtraTrees)")
plt.grid(True)
plt.savefig(os.path.join(model_output_dir, "position_plot.png"))
plt.clf()

plt.figure(figsize=(6, 6))
plt.scatter(y_test_curv["Curvature_Label"], y_pred_curv[:, 0], alpha=0.4)
plt.xlabel("True Curvature")
plt.ylabel("Predicted Curvature")
plt.title("Curvature Prediction (ExtraTrees)")
plt.grid(True)
plt.savefig(os.path.join(model_output_dir, "curvature_plot.png"))
plt.clf()

import joblib
joblib.dump(model_pos.best_estimator_, os.path.join(model_output_dir, "position_model.pkl"))
joblib.dump(model_curv.best_estimator_, os.path.join(model_output_dir, "curvature_model.pkl"))
print("✅ Training complete for both Position and Curvature models")
print("Best Parameters for Position Model:", model_pos.best_params_)
print(f"Position RMSE: {rmse_pos:.2f} cm, R²: {r2_pos:.3f}")
print("Best Parameters for Curvature Model:", model_curv.best_params_)
print(f"Curvature RMSE: {rmse_curv:.5f}, R²: {r2_curv:.3f}")
