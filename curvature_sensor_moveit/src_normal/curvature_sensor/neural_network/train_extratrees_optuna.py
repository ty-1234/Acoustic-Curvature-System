import os
import sys
import json
import time
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from datetime import datetime
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
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

# Additional FFT-based features - extraction of frequency values
freqs = np.array([int(col.split('_')[1].split('Hz')[0]) for col in fft_cols])
df["FFT_Centroid"] = df[fft_cols].dot(freqs) / df[fft_cols].sum(axis=1)

# Frequency band calculations
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

# Define objective function for Optuna
def objective(trial):
    # Define the hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42
    }

    # Create model with suggested hyperparameters
    base_model = ExtraTreesRegressor(**params)
    model = MultiOutputRegressor(base_model)
    
    try:
        # Create a cross-validation object for evaluation
        cv = GroupKFold(n_splits=2)
        cv_splits = list(cv.split(X_train, y_train, groups=groups[train_idx]))
        
        # Custom CV scoring to handle multioutput and sample weights
        scores = []
        for train_cv_idx, val_cv_idx in cv_splits:
            X_train_cv, X_val_cv = X_train[train_cv_idx], X_train[val_cv_idx]
            y_train_cv, y_val_cv = y_train.iloc[train_cv_idx], y_train.iloc[val_cv_idx]
            sw_train_cv = sample_weight[train_idx][train_cv_idx]
            
            model.fit(X_train_cv, y_train_cv, sample_weight=sw_train_cv)
            y_pred_cv = model.predict(X_val_cv)
            
            # Calculate combined RMSE for both outputs (weighted)
            rmse_pos = mean_squared_error(y_val_cv.iloc[:, 0], y_pred_cv[:, 0]) ** 0.5
            rmse_curv = mean_squared_error(y_val_cv.iloc[:, 1], y_pred_cv[:, 1]) ** 0.5
            combined_score = (rmse_pos + rmse_curv * 100) / 2  # Weight curvature error higher
            scores.append(combined_score)
        
        return np.mean(scores)
    
    except ValueError as e:
        # Handle the case where GroupKFold cannot split due to insufficient groups
        if "Cannot have number of splits" in str(e):
            print(f"Skipping trial {trial.number} due to insufficient groups for GroupKFold splitting.")
        else:
            print(f"Unexpected error in trial {trial.number}: {str(e)}")
        
        # Return infinity to mark this trial as failed but allow optimization to continue
        return float('inf')

# Create Optuna study and optimize with fixed random state
print("Starting Bayesian optimization with Optuna...")
study = optuna.create_study(
    direction='minimize', 
    sampler=optuna.samplers.TPESampler(seed=42)  # Added fixed random state
)
study.optimize(objective, n_trials=400, n_jobs=-1)  # Increased from 200 to 400 trials

print("Best hyperparameters:", study.best_params)
best_params = study.best_params

# Train final model with the best parameters
base_model = ExtraTreesRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    bootstrap=best_params['bootstrap'],
    random_state=42
)
best_model = MultiOutputRegressor(base_model)
best_model.fit(X_train, y_train, sample_weight=sample_weight[train_idx])

# Evaluate best model on test fold
y_pred = best_model.predict(X_test)
rmse_pos = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]) ** 0.5
rmse_curv = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]) ** 0.5
r2_pos = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
r2_curv = r2_score(y_test.iloc[:, 1], y_pred[:, 1])

# Calculate additional metrics
mae_pos = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
mae_curv = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
maxerr_pos = max_error(y_test.iloc[:, 0], y_pred[:, 0])
maxerr_curv = max_error(y_test.iloc[:, 1], y_pred[:, 1])

# Create timestamped output directory
timestamp = datetime.now().strftime("%-I:%M%p").lower()
model_output_dir = os.path.join(base_dir, "neural_network", "model_outputs", "extratrees", f"{timestamp}")
os.makedirs(model_output_dir, exist_ok=True)

# Save metrics to JSON
metrics = {
    "model": "ExtraTrees_Optuna",
    "rmse_position": rmse_pos,
    "rmse_curvature": rmse_curv,
    "r2_position": r2_pos,
    "r2_curvature": r2_curv,
    "mae_position": mae_pos,
    "mae_curvature": mae_curv,
    "maxerr_position": maxerr_pos,
    "maxerr_curvature": maxerr_curv,
    "best_params": best_params,
    "optimization_history": [
        {"trial": t.number, "value": t.value, "params": t.params}
        for t in study.trials
    ]
}

with open(os.path.join(model_output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Save model
joblib.dump(best_model, os.path.join(model_output_dir, "extratrees_optuna_model.pkl"))

# Create visualizations

# 1. Bar chart for MAE and Max Error
labels = ['Position', 'Curvature']
mae_vals = [mae_pos, mae_curv]
maxerr_vals = [maxerr_pos, maxerr_curv]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, mae_vals, width, label='MAE')
plt.bar(x + width/2, maxerr_vals, width, label='Max Error')
plt.ylabel('Error Value')
plt.title('MAE and Max Error for ExtraTrees (Optuna)')
plt.xticks(x, labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(model_output_dir, "extratree_error_summary.png"))
plt.close()

# 2. Histogram of Absolute Position Error
abs_pos_error = np.abs(y_test.iloc[:, 0] - y_pred[:, 0])
plt.figure(figsize=(8, 4))
plt.hist(abs_pos_error, bins=20, edgecolor='black')
plt.xlabel('Absolute Position Error (cm)')
plt.ylabel('Frequency')
plt.title('Histogram of Position Errors')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(model_output_dir, "extratree_position_error_hist.png"))
plt.close()

# 3. Position Predicted vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test["Position_cm"], y_pred[:, 0], alpha=0.4)
plt.plot([-1, 5.5], [-1, 5.5], 'r--', label='Ideal')
plt.legend()
plt.xlim([-1, 5.5])
plt.ylim([-1, 5.5])
plt.title(f"Position Prediction (ExtraTrees+Optuna)\nRMSE = {rmse_pos:.2f} cm, R² = {r2_pos:.2f}")
plt.xlabel("True Position (cm)")
plt.ylabel("Predicted Position (cm)")
plt.grid(True)
plt.savefig(os.path.join(model_output_dir, "position_plot.png"))
plt.close()

# 4. Curvature Predicted vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test["Curvature_Label"], y_pred[:, 1], alpha=0.4)
plt.plot([0, 0.055], [0, 0.055], 'r--', label='Ideal')
plt.legend()
plt.xlim([0, 0.055])
plt.ylim([0, 0.055])
plt.title(f"Curvature Prediction (ExtraTrees+Optuna)\nRMSE = {rmse_curv:.5f}, R² = {r2_curv:.2f}")
plt.xlabel("True Curvature")
plt.ylabel("Predicted Curvature")
plt.grid(True)
plt.savefig(os.path.join(model_output_dir, "curvature_plot.png"))
plt.close()

# 5. Optuna optimization history
plt.figure(figsize=(10, 6))
optimization_values = [t.value for t in study.trials]
plt.plot(optimization_values, 'o-')
plt.axhline(y=study.best_value, color='r', linestyle='--', label=f'Best Score: {study.best_value:.4f}')
plt.title('Optuna Optimization History')
plt.xlabel('Trial')
plt.ylabel('Objective Value (Combined RMSE)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(model_output_dir, "optimization_history.png"))
plt.close()

# Print summary
print("✅ Training complete with Optuna optimization")
print("Best Parameters:", best_params)
print(f"Curvature RMSE: {rmse_curv:.5f}, R²: {r2_curv:.3f}")
print(f"Position RMSE: {rmse_pos:.2f} cm, R²: {r2_pos:.3f}")