import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import time 
import json
import lightgbm as lgbm
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

print("Starting the LightGBM Regression training process...")
start_time = time.time()

# Define the directory containing the merged files
base_dir = os.path.dirname(os.path.abspath(__file__))
merged_files_dir = os.path.join(base_dir, "..", "csv_data", "merged2")

print(f"Reading CSV files from directory: {merged_files_dir}")
all_files = glob.glob(os.path.join(merged_files_dir, "*.csv"))
print(f"Found {len(all_files)} CSV files to process")

dfs = []
for file in all_files:
    print(f"Reading file: {os.path.basename(file)}")
    dfs.append(pd.read_csv(file))

df = pd.concat(dfs, axis=0, ignore_index=True)
print(f"Total rows after merging: {df.shape[0]}")
initial_rows = df.shape[0]
df = df.drop_duplicates(subset=['Timestamp'])
print(f"Removed {initial_rows - df.shape[0]} duplicate rows based on Timestamp")
df.loc[(df['Clamp'] == 'open') | (df['Clamp'].isna()), 'Curvature'] = 0
df['Position'] = df['Section'].astype(str).str.replace('cm', '').astype(float)  # Renamed here
df = df.drop(columns=['Section'])  # Remove old column if you want
# Save merged CSV file
merged_csv_path = os.path.join(merged_files_dir, "merged_all.csv")
df.to_csv(merged_csv_path, index=False)
print(f"Merged CSV file saved to: {merged_csv_path}")

# Define FFT columns
fft_columns = [f'FFT_{i}Hz' for i in range(100, 16001, 100)]

# Calculate baseline only from Curvature == 0, Section is null, Clamp is null, and only before 0 cm
print("Step 2: Calculating baseline averages from Curvature == 0, Section is null, Clamp is null, and only before 0 cm")


# Find all indices where Position == 0 and Position == 5
zero_indices = df.index[df['Position'] == 0].tolist()
five_indices = df.index[df['Position'] == 5].tolist()

baseline_indices = []

# For each zero index, find the previous five index (or start of df)
for zero_idx in zero_indices:
    prev_five = [idx for idx in five_indices if idx < zero_idx]
    start_idx = prev_five[-1] + 1 if prev_five else 0
    # Add indices between start_idx and zero_idx (exclusive)
    baseline_indices.extend(range(start_idx, zero_idx))

baseline_region = df.loc[baseline_indices]
baseline_region = baseline_region[
    (baseline_region['Curvature'] == 0) &
    (baseline_region['Position'].isna()) &
    (baseline_region['Clamp'].isna())
]

print(f"Baseline region contains {baseline_region.shape[0]} rows.")

baseline_averages = {}
for col in fft_columns:
    if not baseline_region.empty:
        baseline_averages[col] = baseline_region[col].mean()
    else:
        baseline_averages[col] = 0

print("\nCalculated Baseline FFT Averages (Curvature == 0, Section is null):")
for col, avg in baseline_averages.items():
    print(f"{col}: {avg}")
print("\nStep 3: Normalizing the FFT values using the baseline")
def normalize_fft(row):
    normalized_values = {}
    for col in fft_columns:
        baseline = baseline_averages.get(col, 0)
        normalized_values[f'Normalized_{col}'] = row[col] - baseline
    return pd.Series(normalized_values)

normalized_fft = df.apply(normalize_fft, axis=1)
normalized_fft = normalized_fft.fillna(0)
df = pd.concat([df, normalized_fft], axis=1)
df['Position'] = df['Position'].astype(str).str.replace('cm', '').astype(float)
normalized_fft_columns = [f'Normalized_{col}' for col in fft_columns]
print(f"Normalized FFT Columns: {normalized_fft_columns}")

print("Filtering data where Clamp = 1")
initial_rows = df.shape[0]
if 'Clamp' in df.columns: 
    df['Clamp'] = df['Clamp'].map({'open': 0, 'close': 1}) 
    # Set Curvature to zero where Clamp is zero
    df.loc[df['Clamp'] == 0, 'Curvature'] = 0
df = df[df["Clamp"] == 1]
print(f"Rows after filtering Clamp == 1: {df.shape[0]} (removed {initial_rows - df.shape[0]} rows)")

print("Handling missing and infinite values")
df = df.replace([np.inf, -np.inf], np.nan)
num_nan_before = df.isna().sum().sum()
df = df.fillna(method='ffill').fillna(method='bfill')
num_nan_after = df.isna().sum().sum()
print(f"Filled {num_nan_before - num_nan_after} NaN values")

print("Converting curvature to radius of curvature")


print("Step 5: Splitting the data into training and testing sets before scaling")
X = df[normalized_fft_columns]
y = df[["Position", "Curvature"]]  # Predict both Position and Curvature

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

print("Step 6: Scaling the normalized FFT columns (features)")
scaler_features = MinMaxScaler()
X_train_scaled = scaler_features.fit_transform(X_train)
X_test_scaled = scaler_features.transform(X_test)
print(f"Features scaled: training shape = {X_train_scaled.shape}, testing shape = {X_test_scaled.shape}")

print("Step 7: Scaling the target columns (Section, Curvature)")
scaler_target = MinMaxScaler()
y_train_scaled = scaler_target.fit_transform(y_train)
y_test_scaled = scaler_target.transform(y_test)
print(f"Target scaled: training shape = {y_train_scaled.shape}, testing shape = {y_test_scaled.shape}")

assert not np.isnan(X_train_scaled).any(), "Training features contain NaN values!"
assert not np.isnan(X_test_scaled).any(), "Testing features contain NaN values!"
assert not np.isnan(y_train_scaled).any(), "Training target contains NaN values!"
assert not np.isnan(y_test_scaled).any(), "Testing target contains NaN values!"

print("\nStep 8: Defining the LightGBM model")
lgbm_model = lgbm.LGBMRegressor(random_state=42)
multi_lgbm = MultiOutputRegressor(lgbm_model)
print("MultiOutputRegressor with LGBMRegressor initialized.")

print("\nStep 9: Performing hyperparameter search using RandomizedSearchCV")
param_distributions = {
    "estimator__num_leaves": [31, 50, 100, 150],
    "estimator__max_depth": [-1, 5, 10, 15, 20],
    "estimator__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "estimator__n_estimators": [50, 100, 200, 300],
    "estimator__min_child_samples": [5, 10, 20, 50],
    "estimator__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "estimator__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "estimator__reg_alpha": [0, 0.1, 0.5, 1.0],
    "estimator__reg_lambda": [0, 0.1, 0.5, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=multi_lgbm,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="neg_mean_squared_error",
    cv=10,
    random_state=42,
    n_jobs=-1
)

random_search_start_time = time.time()
random_search.fit(X_train_scaled, y_train_scaled)
print(f"RandomizedSearchCV completed in {time.time() - random_search_start_time:.2f} seconds")

best_params = random_search.best_params_
print(f"Best hyperparameters: {best_params}")

print("\nStep 10: Training the final MultiOutputRegressor model on the full training set")
final_model_start_time = time.time()
multi_lgbm = MultiOutputRegressor(
    lgbm.LGBMRegressor(
        num_leaves=best_params["estimator__num_leaves"],
        max_depth=best_params["estimator__max_depth"],
        learning_rate=best_params["estimator__learning_rate"],
        n_estimators=best_params["estimator__n_estimators"],
        min_child_samples=best_params["estimator__min_child_samples"],
        subsample=best_params["estimator__subsample"],
        colsample_bytree=best_params["estimator__colsample_bytree"],
        reg_alpha=best_params["estimator__reg_alpha"],
        reg_lambda=best_params["estimator__reg_lambda"],
        random_state=42,
        verbose=-1
    )
)
multi_lgbm.fit(X_train_scaled, y_train_scaled)
print(f"Final model trained in {time.time() - final_model_start_time:.2f} seconds")

print("\nStep 11: Making predictions on the test set")
y_pred_scaled = multi_lgbm.predict(X_test_scaled)
y_pred_original = scaler_target.inverse_transform(y_pred_scaled)
y_test_original = scaler_target.inverse_transform(y_test_scaled)

# Metrics for both outputs
rmse_position = np.sqrt(mean_squared_error(y_test_original[:, 0], y_pred_original[:, 0]))
mae_position = mean_absolute_error(y_test_original[:, 0], y_pred_original[:, 0])
r2_position = r2_score(y_test_original[:, 0], y_pred_original[:, 0])

rmse_curvature = np.sqrt(mean_squared_error(y_test_original[:, 1], y_pred_original[:, 1]))
mae_curvature = mean_absolute_error(y_test_original[:, 1], y_pred_original[:, 1])
r2_curvature = r2_score(y_test_original[:, 1], y_pred_original[:, 1])

print(f"Position - RMSE: {rmse_position}")
print(f"Position - MAE: {mae_position}")
print(f"Position - R² Score: {r2_position}")
print(f"Curvature - RMSE: {rmse_curvature}")
print(f"Curvature - MAE: {mae_curvature}")
print(f"Curvature - R² Score: {r2_curvature}")

metrics = {
    "Position": {
        "RMSE": rmse_position,
        "MAE": mae_position,
        "R² Score": r2_position
    },
    "Curvature": {
        "RMSE": rmse_curvature,
        "MAE": mae_curvature,
        "R² Score": r2_curvature
    },
    "Best Hyperparameters": best_params
}

model_name = "lightgbm_multioutput_model"
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(base_dir, "models_output", f"{model_name}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

metrics_path = os.path.join(output_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {metrics_path}")

# Save only the optimal hyperparameters found by RandomizedSearchCV
best_hyperparams_path = os.path.join(output_dir, "best_hyperparameters.json")
with open(best_hyperparams_path, "w") as f:
    json.dump(best_params, f, indent=2)
print(f"Optimal hyperparameters saved to {best_hyperparams_path}")

# Optionally, also save model parameters with additional info as before
model_params = best_params.copy()
model_params.update({
    "test_size": 0.2,
    "random_state": 42
})

params_path = os.path.join(output_dir, "model_params.json")
with open(params_path, "w") as f:
    json.dump(model_params, f, indent=2)
print(f"Model parameters saved to {params_path}")

# Scatter plots
plt.figure(figsize=(8, 6))
plt.scatter(y_test_original[:, 0], y_pred_original[:, 0], alpha=0.5)
plt.plot([y_test_original[:, 0].min(), y_test_original[:, 0].max()],
         [y_test_original[:, 0].min(), y_test_original[:, 0].max()],
         'r--', label='Reference (y=x)')  # Reference line
plt.xlabel("True Position")
plt.ylabel("Predicted Position")
plt.title("Position: True vs Predicted")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "scatter_position.png"))
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(y_test_original[:, 1], y_pred_original[:, 1], alpha=0.5, color='orange')
plt.plot([y_test_original[:, 1].min(), y_test_original[:, 1].max()],
         [y_test_original[:, 1].min(), y_test_original[:, 1].max()],
         'r--', label='Reference (y=x)')  # Reference line
plt.xlabel("True Curvature")
plt.ylabel("Predicted Curvature")
plt.title("Curvature: True vs Predicted")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "scatter_curvature.png"))
plt.close()

# Feature importance plots (LightGBM specific)
print("\nStep 12: Plotting feature importances")

# Since we're using MultiOutputRegressor, we need to get feature importance for each target
for i, target_name in enumerate(["Position", "Curvature"]):
    estimator = multi_lgbm.estimators_[i]
    importance = estimator.feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importance)[::-1]
    feature_names = np.array(normalized_fft_columns)[indices]
    
    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    plt.title(f"Feature Importance for {target_name}")
    plt.barh(range(min(20, len(indices))), importance[indices][:20], align='center')
    plt.yticks(range(min(20, len(indices))), feature_names[:20])
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"feature_importance_{target_name.lower()}.png"))
    plt.close()

print(f"Feature importance plots saved to {output_dir}")
print(f"All outputs saved to {output_dir}")

total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print("LightGBM Multi-output Regression training process completed!")