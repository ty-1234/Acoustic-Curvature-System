# File: curvature_sensor_moveit/src_normal/curvature_sensor/neural_network/analysis_tools/trainGPmodel_ARD.py

import os
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

# === Define features and targets ===
feature_cols = [
    'FFT_600Hz', 'FFT_800Hz', 'FFT_1000Hz', 'FFT_1200Hz', 'FFT_1400Hz',
    'FFT_1600Hz', 'FFT_1800Hz', 'FFT_2000Hz', 'Mid_Band_Mean', 'PC1'
]
target_cols = ["Curvature", "Position_cm"]

# === Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "..", "csv_data", "preprocessed", "merged_tests")
output_path = os.path.join(script_dir, "analysis_output", "gp_model_ARD_groupkfold.json")

# === Load and merge datasets ===
df1 = pd.read_csv(os.path.join(data_dir, "test_01.csv"))
df1["RunID"] = "test_01"
df2 = pd.read_csv(os.path.join(data_dir, "test_02.csv"))
df2["RunID"] = "test_02"
df3 = pd.read_csv(os.path.join(data_dir, "test_03.csv"))
df3["RunID"] = "test_03"

df = pd.concat([df1, df2, df3], ignore_index=True)
df = df[df["Curvature_Active"].isin([0, 1])].copy()
df["Position_cm"] = df["Position_cm"].fillna(-1)

X = df[feature_cols].copy()
y = df[target_cols].copy()
groups = df["RunID"]

# === Normalize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Use ARD kernel ===
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X_scaled.shape[1]), length_scale_bounds=(1e-5, 1e5)) + WhiteKernel()
model = MultiOutputRegressor(GaussianProcessRegressor(kernel=kernel, alpha=1e-4, normalize_y=True, random_state=42))

# === GroupKFold ===
gkf = GroupKFold(n_splits=3)
fold_r2 = []
fold_rmse = []
length_scales = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups)):
    print(f"\n🔁 Fold {fold + 1}")
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    mse = mean_squared_error(y_val, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred, multioutput='raw_values')

    fold_rmse.append(rmse)
    fold_r2.append(r2)

    for i, est in enumerate(model.estimators_):
        ls = est.kernel_.k1.k2.length_scale
        length_scales.append(ls)

    print(f"R2: Curvature={r2[0]:.3f}, Position_cm={r2[1]:.3f}")
    print(f"RMSE: Curvature={rmse[0]:.6f}, Position_cm={rmse[1]:.6f}")

# === Average metrics ===
r2_avg = np.mean(fold_r2, axis=0)
rmse_avg = np.mean(fold_rmse, axis=0)

# === Average length scales ===
length_scales = np.array(length_scales)
mean_ls_curvature = np.mean(length_scales[::2], axis=0)
mean_ls_position = np.mean(length_scales[1::2], axis=0)

# === Save results ===
metrics = {
    "model": "MultiOutput(GaussianProcessRegressor with ARD)",
    "mode": "GroupKFold (n_splits=3)",
    "features_used": feature_cols,
    "mean_r2": {
        "Curvature": float(r2_avg[0]),
        "Position_cm": float(r2_avg[1])
    },
    "mean_rmse": {
        "Curvature": float(rmse_avg[0]),
        "Position_cm": float(rmse_avg[1])
    },
    "length_scales": {
        "Curvature": dict(zip(feature_cols, mean_ls_curvature.round(5).tolist())),
        "Position_cm": dict(zip(feature_cols, mean_ls_position.round(5).tolist()))
    }
}

with open(output_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("\n✅ ARD GroupKFold training complete. Metrics and feature importances saved.")
