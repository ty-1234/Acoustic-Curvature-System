import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor

# === Setup paths ===
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(parent_dir, "csv_data", "combined_dataset.csv")

# Timestamp for output directory ---
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(parent_dir, "output", f"extratrees_optimized_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# === Load dataset ===
print("\n🚀 Training optimized ExtraTrees model")
print(f"📂 Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)
df = df.dropna(subset=["Curvature_Label", "Section"])
df["Position_cm"] = df["Section"].str.replace("cm", "").astype(float)

# === Prepare features ===
fft_cols = [col for col in df.columns if col.startswith("FFT_")]
X = df[fft_cols]
y = df[["Position_cm", "Curvature_Label"]]
groups = df["RunID"]

# === Scale features ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
scaler_path = os.path.join(output_dir, "fft_scaler.pkl")
joblib.dump(scaler, scaler_path)

# === Define base model and hyperparameter grid ===
base_model = ExtraTreesRegressor(random_state=42)
param_grid = {
    "estimator__n_estimators": [300, 500, 700, 1000, 1500],
    "estimator__max_depth": [5, 10, 15, 20, None],
    "estimator__min_samples_split": [2, 5, 10, 20],
    "estimator__min_samples_leaf": [1, 2, 4, 8],
    "estimator__max_features": ["sqrt", "log2", None]
}
model = MultiOutputRegressor(base_model)

# === Hyperparameter tuning ===
print("\n🔧 Running RandomizedSearchCV with extensive grid...")
# Increased n_iter to 40 for better coverage of the expanded parameter space
search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=40, random_state=42, n_jobs=-1, cv=3)
search.fit(X_scaled, y)
best_model = search.best_estimator_
print(f"\n✅ Best Parameters: {search.best_params_}")

# === Evaluation ===
gkf = GroupKFold(n_splits=2)
rmse_pos_list, rmse_curv_list = [], []
r2_pos_list, r2_curv_list = [], []

print("\n📊 Cross-validation Evaluation:")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    rmse_pos = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]) ** 0.5
    rmse_curv = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]) ** 0.5
    r2_pos = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
    r2_curv = r2_score(y_test.iloc[:, 1], y_pred[:, 1])
    rmse_pos_list.append(rmse_pos)
    rmse_curv_list.append(rmse_curv)
    r2_pos_list.append(r2_pos)
    r2_curv_list.append(r2_curv)
    print(f"Fold {fold+1}: RMSE Pos = {rmse_pos:.3f} cm, RMSE Curv = {rmse_curv:.5f}, R² Pos = {r2_pos:.3f}, R² Curv = {r2_curv:.3f}")

# === Final training on all data ===
best_model.fit(X_scaled, y)
model_path = os.path.join(output_dir, "multioutput_model.pkl")
joblib.dump(best_model, model_path)

# === Save metrics ===.
metrics = {
    "rmse_position_mean": np.mean(rmse_pos_list),
    "rmse_curvature_mean": np.mean(rmse_curv_list),
    "r2_position_mean": np.mean(r2_pos_list),
    "r2_curvature_mean": np.mean(r2_curv_list),
    "best_params": search.best_params_
}
with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# === Plotting ===
y_pred_all = best_model.predict(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(y["Position_cm"], y_pred_all[:, 0], alpha=0.3)
plt.plot([0, 5], [0, 5], 'r--')
plt.xlabel("True Position (cm)")
plt.ylabel("Predicted Position (cm)")
plt.title("Position: True vs Predicted (ExtraTrees Optimized)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "position_plot.png"))
plt.clf()

plt.figure(figsize=(8, 6))
plt.scatter(y["Curvature_Label"], y_pred_all[:, 1], alpha=0.3)
plt.plot([y["Curvature_Label"].min(), y["Curvature_Label"].max()],
         [y["Curvature_Label"].min(), y["Curvature_Label"].max()], 'r--')
plt.xlabel("True Curvature")
plt.ylabel("Predicted Curvature")
plt.title("Curvature: True vs Predicted (ExtraTrees Optimized)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "curvature_plot.png"))
plt.clf()

print(f"\n✅ Optimized ExtraTrees model training complete. All files saved to: {output_dir}")
