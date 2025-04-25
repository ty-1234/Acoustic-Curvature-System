import os
import sys
import json
import time
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# === Setup paths ===
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(parent_dir, "csv_data", "preprocessed", "preprocessed_training_dataset.csv")
output_dir = os.path.join(parent_dir, "multioutput_model_results")
os.makedirs(output_dir, exist_ok=True)

# === Load and preprocess dataset ===
df = pd.read_csv(dataset_path)
df = df[~(df["Position_cm"].isna() & (df["Curvature_Active"] == 1))]
df["Position_cm"] = df["Position_cm"].fillna(-1)
df["FFT_Peak_Index"] = (
    df[[col for col in df.columns if col.startswith("FFT_")]]
    .idxmax(axis=1)
    .str.extract(r"(\d+)")
    .astype(float)
    .fillna(-1)
    .astype(int)
)

print("\n📋 Unique RunIDs detected:", df["RunID"].unique())
print(f"Total unique RunIDs: {df['RunID'].nunique()}")

feature_cols = [col for col in df.columns if col.startswith("FFT_") or "Band" in col or "Ratio" in col or col == "Curvature_Active" or col == "FFT_Peak_Index"]
X = df[feature_cols]
y = df[["Position_cm", "Curvature_Label"]]
sample_weight = np.where(y["Position_cm"] == -1, 0.01, 1.0)
groups = df["RunID"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

models_to_run = ["extratrees", "xgb", "lgbm", "rf"]
all_metrics = []

for model_name in models_to_run:
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\n🚀 Training final model: {model_name}")

    if model_name == "extratrees":
        base_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [10, 20, 30, 50, None],
            "estimator__min_samples_split": [2, 4, 6, 8],
            "estimator__min_samples_leaf": [1, 2, 3],
            "estimator__max_features": ["sqrt", "log2", None],
            "estimator__bootstrap": [True, False]
        }
    elif model_name == "xgb":
        base_model = XGBRegressor(n_estimators=100, random_state=42)
        param_grid = {
            "estimator__max_depth": [3, 5, 7],
            "estimator__learning_rate": [0.05, 0.1, 0.2],
            "estimator__n_estimators": [100, 200, 300]
        }
    elif model_name == "lgbm":
        base_model = LGBMRegressor(random_state=42)
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__learning_rate": [0.05, 0.1, 0.2],
            "estimator__max_depth": [3, 5, 7, -1],
            "estimator__num_leaves": [31, 50, 100],
            "estimator__boosting_type": ["gbdt", "dart"]
        }
    elif model_name == "rf":
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [10, 20, 30, 50, None],
            "estimator__min_samples_split": [2, 4, 6, 8],
            "estimator__min_samples_leaf": [1, 2, 3],
            "estimator__max_features": ["sqrt", "log2", None],
            "estimator__bootstrap": [True, False]
        }

    model = MultiOutputRegressor(base_model)
    print(f"\n🔧 Performing RandomizedSearchCV for {model_name}...")
    search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, random_state=42, n_jobs=-1, cv=3)
    search.fit(X_scaled, y)
    best_model = search.best_estimator_

    gkf = GroupKFold(n_splits=2)
    rmse_pos_list, rmse_curv_list, r2_pos_list, r2_curv_list = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        best_model.fit(X_train, y_train, sample_weight=sample_weight[train_idx])
        y_pred = best_model.predict(X_test)

        rmse_pos = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]) ** 0.5
        rmse_curv = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]) ** 0.5
        r2_pos = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
        r2_curv = r2_score(y_test.iloc[:, 1], y_pred[:, 1])

        rmse_pos_list.append(rmse_pos)
        rmse_curv_list.append(rmse_curv)
        r2_pos_list.append(r2_pos)
        r2_curv_list.append(r2_curv)

    metrics = {
        "model": model_name,
        "rmse_position_mean": np.mean(rmse_pos_list),
        "rmse_curvature_mean": np.mean(rmse_curv_list),
        "r2_position_mean": np.mean(r2_pos_list),
        "r2_curvature_mean": np.mean(r2_curv_list)
    }
    all_metrics.append(metrics)

    print(f"\n📌 Final Evaluation Metrics for {model_name}:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

# === Save and plot summary ===
metrics_df = pd.DataFrame(all_metrics)
metrics_df.sort_values("rmse_curvature_mean", inplace=True)
metrics_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

plt.figure(figsize=(10, 6))
plt.bar(metrics_df["model"], metrics_df["rmse_curvature_mean"], color='skyblue')
plt.title("Model Comparison - RMSE Curvature")
plt.ylabel("RMSE (Curvature)")
plt.xlabel("Model")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rmse_curvature_comparison.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(metrics_df["model"], metrics_df["rmse_position_mean"], color='lightgreen')
plt.title("Model Comparison - RMSE Position")
plt.ylabel("RMSE (Position cm)")
plt.xlabel("Model")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rmse_position_comparison.png"))
plt.close()

print("\n✅ All models evaluated and summary saved.")
