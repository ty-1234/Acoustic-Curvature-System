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
from sklearn.model_selection import ParameterGrid

# === Model selection via interactive prompt ===
def select_model_interactively():
    models = {
        "1": "extratrees",
        "2": "xgb",
        "3": "gb",
        "4": "rf"
    }
    
    print("\n=== Model Selection ===")
    print("Please select a model to train:")
    print("1. ExtraTrees Regressor")
    print("2. XGBoost Regressor")
    print("3. Gradient Boosting Regressor")
    print("4. Random Forest Regressor")
    
    while True:
        choice = input("\nEnter your choice (1-4) or 'q' to quit: ")
        if choice == 'q':
            print("Exiting...")
            sys.exit(0)
        if choice in models:
            return models[choice]
        print("Invalid choice. Please try again.")

# === Parse command line arguments if provided, otherwise use interactive mode ===
parser = argparse.ArgumentParser(description="Train final multi-output regressor with tuning")
parser.add_argument("--model", type=str, choices=["extratrees", "xgb", "gb", "rf"],
                    help="Which model to train: 'extratrees', 'xgb', 'gb', 'rf'")
args = parser.parse_args()

# If model not provided via command line, use interactive selection
if args.model is None:
    args.model = select_model_interactively()

# === Setup paths ===
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(parent_dir, "csv_data", "combined_dataset.csv")
output_dir = os.path.join(parent_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Create model-specific subfolder
model_output_dir = os.path.join(output_dir, args.model)
os.makedirs(model_output_dir, exist_ok=True)

# === Setup ===
print(f"\n🚀 Training final model: {args.model}")
print(f"📂 Loading dataset from: {dataset_path}")
print(f"📂 Outputs will be saved to: {model_output_dir}")

df = pd.read_csv(dataset_path)
df = df.dropna(subset=["Curvature_Label", "Section"])
df["Position_cm"] = df["Section"].str.replace("cm", "").astype(float)

# Check the unique RunIDs
print("📋 Unique RunIDs detected:", df["RunID"].unique())
print(f"Total unique RunIDs: {df['RunID'].nunique()}")

fft_cols = [col for col in df.columns if col.startswith("FFT_")]
X = df[fft_cols]
y = df[["Position_cm", "Curvature_Label"]]
groups = df["RunID"]

# === Scaling ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(model_output_dir, "fft_scaler.pkl"))

# === Model selection and hyperparameter tuning ===
if args.model == "extratrees":
    base_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
    param_grid = {
        "estimator__max_depth": [5, 10, 15, None], 
        "estimator__n_estimators": [100, 200, 300]
    }
elif args.model == "xgb":
    base_model = XGBRegressor(n_estimators=100, random_state=42)
    param_grid = {
        "estimator__max_depth": [3, 5, 7], 
        "estimator__learning_rate": [0.05, 0.1, 0.2], 
        "estimator__n_estimators": [100, 200, 300]
    }
elif args.model == "gb":
    base_model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        "estimator__n_estimators": [100, 200], 
        "estimator__learning_rate": [0.05, 0.1], 
        "estimator__max_depth": [3, 5, 7]
    }
elif args.model == "rf":
    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    param_grid = {
        "estimator__max_depth": [5, 10, None], 
        "estimator__n_estimators": [100, 200, 300]
    }

# Multi-output regression
model = MultiOutputRegressor(base_model)

# === Hyperparameter tuning using RandomizedSearchCV ===
print(f"\n🔧 Performing RandomizedSearchCV for {args.model}...")
search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, random_state=42, n_jobs=-1, cv=3)
search.fit(X_scaled, y)

best_model = search.best_estimator_
print(f"\n🚀 Best parameters: {search.best_params_}")

# === Evaluation ===
gkf = GroupKFold(n_splits=2)  # Use 2 splits for your dataset
rmse_pos_list = []
rmse_curv_list = []
r2_pos_list = []
r2_curv_list = []

print("\n📊 Evaluating with GroupKFold:")
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
model_path = os.path.join(model_output_dir, "multioutput_model.pkl")
joblib.dump(best_model, model_path)
print(f"\n✅ Model saved as: {model_path}")

# === Save metrics ===
metrics = {
    "model": args.model,
    "rmse_position_mean": np.mean(rmse_pos_list),
    "rmse_curvature_mean": np.mean(rmse_curv_list),
    "r2_position_mean": np.mean(r2_pos_list),
    "r2_curvature_mean": np.mean(r2_curv_list)
}
metrics_path = os.path.join(model_output_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"📄 Metrics saved to: {metrics_path}")

# === Plotting ===
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Position plot
y_pred_all = best_model.predict(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(y["Position_cm"], y_pred_all[:, 0], alpha=0.3)
plt.plot([0, 5], [0, 5], 'r--')
plt.xlabel("True Position (cm)")
plt.ylabel("Predicted Position (cm)")
plt.title(f"Position: True vs Predicted ({args.model})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_position_path = os.path.join(model_output_dir, "position_plot.png")
plt.savefig(plot_position_path)
plt.clf()

# Curvature plot
plt.figure(figsize=(8, 6))
plt.scatter(y["Curvature_Label"], y_pred_all[:, 1], alpha=0.3)
plt.plot([y["Curvature_Label"].min(), y["Curvature_Label"].max()],
         [y["Curvature_Label"].min(), y["Curvature_Label"].max()], 'r--')
plt.xlabel("True Curvature")
plt.ylabel("Predicted Curvature")
plt.title(f"Curvature: True vs Predicted ({args.model})")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plot_curvature_path = os.path.join(model_output_dir, "curvature_plot.png")
plt.savefig(plot_curvature_path)
plt.clf()

print(f"📈 Plots saved to: {model_output_dir}")
print(f"📊 All files for {args.model} model saved to: {model_output_dir}")
