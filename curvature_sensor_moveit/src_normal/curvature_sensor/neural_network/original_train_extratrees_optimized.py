import os
import sys
import json
import time
import joblib
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor

# Load dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "csv_data", "preprocessed", "preprocessed_training_dataset.csv")
df = pd.read_csv(dataset_path)

# Drop rows where Position_cm is missing while Curvature_Active == 1
df = df[~(df["Position_cm"].isna() & (df["Curvature_Active"] == 1))]

# Fill missing Position_cm with a placeholder (-1) to retain rows for curvature learning
df["Position_cm"] = df["Position_cm"].fillna(-1)

# FFT peak index feature
fft_cols = [col for col in df.columns if col.startswith("FFT_")]
df["FFT_Peak_Index"] = (
    df[fft_cols]
    .idxmax(axis=1)
    .str.extract(r"(\d+)")
    .astype(float)
    .fillna(-1)
    .astype(int)
)

# Select features and targets
feature_cols = [col for col in df.columns if col.startswith("FFT_") or "Band" in col or "Ratio" in col or col == "FFT_Peak_Index" or col == "Curvature_Active"]
X = df[feature_cols]
y = pd.DataFrame({
    "Position_cm": df["Position_cm"],
    "Curvature_Label": df["Curvature_Label"]
})

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# GroupKFold split using RunID to avoid temporal leakage
groups = df["RunID"]
gkf = GroupKFold(n_splits=2)
train_idx, test_idx = next(gkf.split(X_scaled, y, groups))
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Hyperparameter tuning
param_dist = {
    'estimator__n_estimators': [100, 200, 300, 500, 800, 1000],
    'estimator__max_features': ['sqrt', 'log2', None],
    'estimator__max_depth': [10, 20, 30, 40, 50, None],
    'estimator__min_samples_split': [2, 4, 6, 8, 10],
    'estimator__min_samples_leaf': [1, 2, 3, 4, 5],
    'estimator__bootstrap': [True, False]
}
base_model = ExtraTreesRegressor(random_state=42)
model = MultiOutputRegressor(base_model)
search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=60, cv=3, random_state=42, n_jobs=-1, verbose=2)
search.fit(X_train, y_train)
best_model = search.best_estimator_

# Evaluate best model
y_pred = best_model.predict(X_test)
rmse_pos = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]) ** 0.5
rmse_curv = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]) ** 0.5
r2_pos = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
r2_curv = r2_score(y_test.iloc[:, 1], y_pred[:, 1])

# Save results
metrics = {
    "model": "ExtraTrees",
    "rmse_position": rmse_pos,
    "rmse_curvature": rmse_curv,
    "r2_position": r2_pos,
    "r2_curvature": r2_curv,
    "best_params": search.best_params_
}
timestamp = datetime.now().strftime("%-I:%M%p").lower()
model_output_dir = os.path.join(base_dir, "models", f"extratrees_optimized_{timestamp}")
os.makedirs(model_output_dir, exist_ok=True)
with open(os.path.join(model_output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Plotting
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

# Save model
joblib.dump(best_model, os.path.join(model_output_dir, "extratrees_model.pkl"))

# Done
print("✅ Training complete")
print("Best Parameters:", search.best_params_)
print(f"Curvature RMSE: {rmse_curv:.5f}, R²: {r2_curv:.3f}")
print(f"Position RMSE: {rmse_pos:.2f} cm, R²: {r2_pos:.3f}")