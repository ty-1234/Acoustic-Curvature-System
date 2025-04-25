import os
import sys
import json
import time
import argparse
import joblib
import pandas as pd
import numpy as np
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
df["Position_cm"] = df["Position_cm"].fillna(-1)

# FFT peak index feature - only include true frequency columns
fft_cols = [col for col in df.columns if col.startswith("FFT_") and "Hz" in col]
df["FFT_Peak_Index"] = (
    df[fft_cols]
    .idxmax(axis=1)
    .str.extract(r"(\d+)")
    .astype(float)
    .fillna(-1)
    .astype(int)
)

# Additional FFT-based features - fix the extraction of frequency values
freqs = np.array([int(col.split('_')[1].split('Hz')[0]) for col in fft_cols])
df["FFT_Centroid"] = df[fft_cols].dot(freqs) / df[fft_cols].sum(axis=1)

# Fix extraction of frequency values in band calculations
low_freq_cols = [col for col in fft_cols if 200 <= int(col.split('_')[1].split('Hz')[0]) <= 800]
mid_freq_cols = [col for col in fft_cols if 800 < int(col.split('_')[1].split('Hz')[0]) <= 1400]
high_freq_cols = [col for col in fft_cols if int(col.split('_')[1].split('Hz')[0]) > 1400]

df["low_band_power"] = df[low_freq_cols].mean(axis=1)
df["mid_band_power"] = df[mid_freq_cols].mean(axis=1)
df["high_band_power"] = df[high_freq_cols].mean(axis=1)

# Feature and target selection
feature_cols = [col for col in df.columns if col.startswith("FFT_") or "Band" in col or "Ratio" in col or col in ["Curvature_Active", "FFT_Peak_Index", "FFT_Centroid", "low_band_power", "mid_band_power", "high_band_power"]]
X = df[feature_cols]
y = df[["Position_cm", "Curvature_Label"]]

# Updated sample weight with 0.01 for inactive points
sample_weight = np.where(y["Position_cm"] == -1, 0.01, 1.0)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Grouped cross-validation
groups = df["RunID"]
gkf = GroupKFold(n_splits=2)
train_idx, test_idx = next(gkf.split(X_scaled, y, groups))
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Hyperparameter tuning for ExtraTrees inside MultiOutputRegressor
param_dist = {
    'estimator__n_estimators': [100, 200, 300, 500],
    'estimator__max_features': ['sqrt', 'log2', None],
    'estimator__max_depth': [10, 20, 30, 40, 50, None],
    'estimator__min_samples_split': [2, 4, 6, 8, 10],
    'estimator__min_samples_leaf': [1, 2, 3, 4],
    'estimator__bootstrap': [True, False]
}

base_model = ExtraTreesRegressor(random_state=42)
model = MultiOutputRegressor(base_model)

search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=3, random_state=42, n_jobs=-1, verbose=2)
search.fit(X_train, y_train, sample_weight=sample_weight[train_idx])
best_model = search.best_estimator_

# Evaluate best model on test fold
y_pred = best_model.predict(X_test)
rmse_pos = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]) ** 0.5
rmse_curv = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]) ** 0.5
r2_pos = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
r2_curv = r2_score(y_test.iloc[:, 1], y_pred[:, 1])

# Save metrics
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

# Save final model
joblib.dump(best_model, os.path.join(model_output_dir, "extratrees_model.pkl"))

print("✅ Training complete")
print("Best Parameters:", search.best_params_)
print(f"Curvature RMSE: {rmse_curv:.5f}, R²: {r2_curv:.3f}")
print(f"Position RMSE: {rmse_pos:.2f} cm, R²: {r2_pos:.3f}")
