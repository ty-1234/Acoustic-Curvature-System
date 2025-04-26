import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# === Setup paths ===
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(parent_dir, "csv_data", "combined_dataset.csv")
output_dir = os.path.join(parent_dir, "neural_network", "model_outputs", "og_gpr")
os.makedirs(output_dir, exist_ok=True)

# === Load dataset and filter only active curvature segments ===
print("\n📥 Loading dataset...")
df = pd.read_csv(dataset_path)
# === Apply logic: only use rows where curvature is actively applied ===
df["Curvature_Active"] = df[["Section", "PosX", "PosY", "PosZ"]].notna().all(axis=1).astype(int)
df = df[df["Curvature_Active"] == 1]
# Optionally, drop only on Curvature_Label now
df = df.dropna(subset=["Curvature_Label"])
fft_cols = [col for col in df.columns if col.startswith("FFT_")]
X = df[fft_cols]
y = df["Curvature_Label"]
groups = df["RunID"]

# === Scale FFT features ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(output_dir, "fft_scaler.pkl"))

# === Define GPR model with Rational Quadratic kernel ===
kernel = RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=1e-5)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

# === Cross-validation ===
gkf = GroupKFold(n_splits=2)
rmse_list = []
r2_list = []

print("\n📊 Evaluating GPR with GroupKFold...")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    # Updated RMSE calculation for compatibility
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    rmse_list.append(rmse)
    r2_list.append(r2)

    print(f"Fold {fold+1}: RMSE = {rmse:.5f}, R² = {r2:.3f}")

# === Train on full dataset ===
gpr.fit(X_scaled, y)
joblib.dump(gpr, os.path.join(output_dir, "gpr_curvature_model.pkl"))

# === Save metrics ===
metrics = {
    "rmse_mean": float(np.mean(rmse_list)),
    "r2_mean": float(np.mean(r2_list)),
    "kernel": str(gpr.kernel_)
}
with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# === Plot: True vs Predicted Curvature ===
y_pred_all = gpr.predict(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred_all, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("True Curvature")
plt.ylabel("Predicted Curvature")
plt.title("GPR (Rational Quadratic) - Curvature Prediction")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "curvature_plot.png"))
plt.clf()

print(f"\n✅ GPR training complete. Files saved to: {output_dir}")
