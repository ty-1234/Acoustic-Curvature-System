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
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Set seed for reproducibility
np.random.seed(42)

# === Setup paths ===
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(parent_dir, "csv_data", "preprocessed", "preprocessed_training_dataset.csv")
output_dir = os.path.join(parent_dir, "neural_network", "model_outputs", "gpr_curvature")
os.makedirs(output_dir, exist_ok=True)

# === Load dataset and filter only active curvature segments ===
print("\n📥 Loading dataset...")
df = pd.read_csv(dataset_path)
df = df[df["Curvature_Active"] == 1]
print(f"Dataset shape after filtering active curvature segments: {df.shape}")

# === Features and Targets ===
feature_cols = [col for col in df.columns if col.startswith("FFT_")]
print(f"Number of FFT features: {len(feature_cols)}")
X = df[feature_cols]
y = df["Curvature_Label"]
groups = df["RunID"]
unique_groups = groups.unique()
print(f"Number of unique runs (groups): {len(unique_groups)}")

# === Scale features ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

# === Define GPR model ===
kernel = RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=1e-5)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)

# === Grouped Cross-validation ===
# Adjust n_splits to match the number of unique groups
n_splits = min(3, len(unique_groups))  # Use either 3 or the number of groups, whichever is smaller
gkf = GroupKFold(n_splits=n_splits)
rmse_list, mae_list, maxerr_list, r2_list = [], [], [], []

print(f"\n📊 Evaluating GPR with GroupKFold (n_splits={n_splits})...")
start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
    fold_start = time.time()
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Fold {fold+1}: Training samples={len(X_train)}, Test samples={len(X_test)}")
    print(f"Training groups: {np.unique(groups.iloc[train_idx])}")
    print(f"Testing groups: {np.unique(groups.iloc[test_idx])}")
    
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    maxerr = max_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse_list.append(rmse)
    mae_list.append(mae)
    maxerr_list.append(maxerr)
    r2_list.append(r2)

    fold_time = time.time() - fold_start
    print(f"Fold {fold+1}: RMSE={rmse:.5f}, MAE={mae:.5f}, MaxError={maxerr:.5f}, R²={r2:.3f}")
    print(f"Fold {fold+1} completed in {fold_time:.2f} seconds\n")

cv_time = time.time() - start_time
print(f"Cross-validation completed in {cv_time:.2f} seconds")

# === Print mean performance metrics ===
print("\n=== Mean Cross-Validation Metrics ===")
print(f"Mean RMSE: {np.mean(rmse_list):.5f} ± {np.std(rmse_list):.5f}")
print(f"Mean MAE: {np.mean(mae_list):.5f} ± {np.std(mae_list):.5f}")
print(f"Mean MaxError: {np.mean(maxerr_list):.5f} ± {np.std(maxerr_list):.5f}")
print(f"Mean R²: {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")

# === Retrain on full dataset ===
print("\n🔄 Retraining on full dataset...")
full_train_start = time.time()
gpr.fit(X_scaled, y)
print(f"Full dataset training completed in {time.time() - full_train_start:.2f} seconds")

# Save the model
model_path = os.path.join(output_dir, "gpr_curvature_model.pkl")
joblib.dump(gpr, model_path)
print(f"Model saved to: {model_path}")

# === Save metrics ===
metrics = {
    "rmse_mean": float(np.mean(rmse_list)),
    "rmse_std": float(np.std(rmse_list)),
    "mae_mean": float(np.mean(mae_list)),
    "mae_std": float(np.std(mae_list)),
    "max_error_mean": float(np.mean(maxerr_list)),
    "max_error_std": float(np.std(maxerr_list)),
    "r2_mean": float(np.mean(r2_list)),
    "r2_std": float(np.std(r2_list)),
    "kernel": str(gpr.kernel_),
    "training_points": len(X),
    "feature_count": len(feature_cols),
    "run_count": len(unique_groups),
    "n_splits_used": n_splits,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

metrics_df = pd.DataFrame({
    "Fold": list(range(1, len(rmse_list) + 1)),
    "RMSE": rmse_list,
    "MAE": mae_list,
    "MaxError": maxerr_list,
    "R2": r2_list
})

metrics_df.to_csv(os.path.join(output_dir, "per_fold_metrics.csv"), index=False)

with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# === Plot: True vs Predicted Curvature ===
y_pred_all = gpr.predict(X_scaled)
plt.figure(figsize=(10, 8))
plt.scatter(y, y_pred_all, alpha=0.4, edgecolors='k', linewidths=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect Prediction')

# Add regression line
z = np.polyfit(y, y_pred_all, 1)
p = np.poly1d(z)
plt.plot(np.sort(y), p(np.sort(y)), 'b-', linewidth=1.5, label=f'Best Fit: y={z[0]:.3f}x+{z[1]:.3f}')

# Add RMSE and R2 to plot
rmse_full = np.sqrt(mean_squared_error(y, y_pred_all))
r2_full = r2_score(y, y_pred_all)
plt.annotate(f'RMSE = {rmse_full:.5f}\nR² = {r2_full:.3f}', 
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle='round', fc='white', alpha=0.8),
             fontsize=12)

plt.xlabel("True Curvature (mm$^{-1}$)", fontsize=14)
plt.ylabel("Predicted Curvature (mm$^{-1}$)", fontsize=14)
plt.title("Gaussian Process Regression - Curvature Prediction", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "curvature_prediction_plot.png"), dpi=300)
plt.close()

# === Plot residuals ===
residuals = y_pred_all - y
plt.figure(figsize=(10, 6))
plt.scatter(y, residuals, alpha=0.4, edgecolors='k', linewidths=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel("True Curvature (mm$^{-1}$)", fontsize=14)
plt.ylabel("Residuals (Predicted - True)", fontsize=14)
plt.title("GPR Model Residuals", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residuals_plot.png"), dpi=300)
plt.close()

print(f"\n✅ GPR training complete. Files saved to: {output_dir}")
print(f"Mean cross-validation RMSE: {np.mean(rmse_list):.5f}")
print(f"Mean cross-validation R²: {np.mean(r2_list):.3f}")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")
