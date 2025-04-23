import os
import sys
import json
import time
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

# === Setup output directory ===
output_dir = "curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/output"
os.makedirs(output_dir, exist_ok=True)

# === Argument parser ===
parser = argparse.ArgumentParser(description="Train final multi-output regressor")
parser.add_argument("--model", type=str, choices=["extratrees", "xgb"], default="extratrees",
                    help="Which model to train: 'extratrees' or 'xgb'")
args = parser.parse_args()

# === Setup ===
print(f"\n🚀 Training final model: {args.model}")
df = pd.read_csv("csv_data/combined_dataset.csv")
df = df.dropna(subset=["Curvature_Label", "Section"])
df["Position_cm"] = df["Section"].str.replace("cm", "").astype(float)

fft_cols = [col for col in df.columns if col.startswith("FFT_")]
X = df[fft_cols]
y = df[["Position_cm", "Curvature_Label"]]
groups = df["RunID"]

# === Scaling ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(output_dir, "fft_scaler.pkl"))

# === Model selection ===
if args.model == "extratrees":
    base_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
elif args.model == "xgb":
    base_model = XGBRegressor(n_estimators=100, random_state=42)
else:
    raise ValueError("Invalid model")

model = MultiOutputRegressor(base_model)

# === Evaluation ===
gkf = GroupKFold(n_splits=5)
rmse_pos_list = []
rmse_curv_list = []
r2_pos_list = []
r2_curv_list = []

print("\n📊 Evaluating with GroupKFold:")
for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

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
model.fit(X_scaled, y)
model_path = os.path.join(output_dir, f"{args.model}_multioutput_model.pkl")
joblib.dump(model, model_path)
print(f"\n✅ Model saved as: {model_path}")

# === Save metrics ===
metrics = {
    "model": args.model,
    "rmse_position_mean": np.mean(rmse_pos_list),
    "rmse_curvature_mean": np.mean(rmse_curv_list),
    "r2_position_mean": np.mean(r2_pos_list),
    "r2_curvature_mean": np.mean(r2_curv_list)
}
metrics_path = os.path.join(output_dir, f"{args.model}_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"📄 Metrics saved to: {metrics_path}")

# === Plotting ===
y_pred_all = model.predict(X_scaled)
plt.scatter(y["Position_cm"], y_pred_all[:, 0], alpha=0.3)
plt.plot([0, 5], [0, 5], 'r--')
plt.xlabel("True Position (cm)")
plt.ylabel("Predicted Position (cm)")
plt.title(f"Position: True vs Predicted ({args.model})")
plt.tight_layout()
plot_position_path = os.path.join(output_dir, f"position_true_vs_pred_{args.model}.png")
plt.savefig(plot_position_path)
plt.clf()

plt.scatter(y["Curvature_Label"], y_pred_all[:, 1], alpha=0.3)
plt.plot([y["Curvature_Label"].min(), y["Curvature_Label"].max()],
         [y["Curvature_Label"].min(), y["Curvature_Label"].max()], 'r--')
plt.xlabel("True Curvature")
plt.ylabel("Predicted Curvature")
plt.title(f"Curvature: True vs Predicted ({args.model})")
plt.tight_layout()
plot_curvature_path = os.path.join(output_dir, f"curvature_true_vs_pred_{args.model}.png")
plt.savefig(plot_curvature_path)
plt.clf()

print("📈 Plots saved!")