import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import time 
import json
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

print("Starting the MLP Regression training process...")
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
df['Position'] = df['Section'].astype(str).str.replace('cm', '').astype(float)
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

# Remove the radius of curvature conversion
# print("Converting curvature to radius of curvature")
# def convert_to_radius_of_curvature(df, curvature_column='Curvature', new_column='Radius_of_Curvature'):
#     df[new_column] = df[curvature_column].replace(0, float('inf')).apply(lambda x: 1 / x if x != float('inf') else float('inf'))
#     return df
# df = convert_to_radius_of_curvature(df)

print("Step 5: Splitting the data into training and testing sets before scaling")
X = df[normalized_fft_columns]
y = df[["Position", "Curvature"]]  # Position in cm, Curvature in mm^-1

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

print("\nStep 8: Defining the MLPRegressor model")
mlp = MLPRegressor(max_iter=1000, random_state=42)
multi_mlp = MultiOutputRegressor(mlp)
print("MultiOutputRegressor with MLPRegressor initialized.")

print("\nStep 9: Performing hyperparameter search using RandomizedSearchCV")
assert not np.isnan(y_test_scaled).any(), "Testing target contains NaN values!"

print("\nStep 8: Defining the MLPRegressor model")
mlp = MLPRegressor(max_iter=1000, random_state=42)
multi_mlp = MultiOutputRegressor(mlp)
print("MultiOutputRegressor with MLPRegressor initialized.")

print("\nStep 9: Performing hyperparameter search using RandomizedSearchCV")
param_distributions = {
    "estimator__hidden_layer_sizes": [(100,), (50, 50), (100, 50, 25)],
    "estimator__activation": ["relu", "tanh", "logistic"],
    "estimator__solver": ["adam", "lbfgs"],
    "estimator__alpha": [0.0001, 0.001, 0.01],
    "estimator__learning_rate": ["constant", "adaptive"]
}

random_search = RandomizedSearchCV(
    estimator=multi_mlp,
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
multi_mlp = MultiOutputRegressor(
    MLPRegressor(
        hidden_layer_sizes=best_params["estimator__hidden_layer_sizes"],
        activation=best_params["estimator__activation"],
        solver=best_params["estimator__solver"],
        alpha=best_params["estimator__alpha"],
        learning_rate=best_params["estimator__learning_rate"],
        max_iter=1000,
        random_state=42
    )
)
multi_mlp.fit(X_train_scaled, y_train_scaled)
print(f"Final model trained in {time.time() - final_model_start_time:.2f} seconds")

print("\nStep 11: Making predictions on the test set")
y_pred_scaled = multi_mlp.predict(X_test_scaled)
y_pred_original = scaler_target.inverse_transform(y_pred_scaled)
y_test_original = scaler_target.inverse_transform(y_test_scaled)

# Metrics for both outputs
rmse_position = np.sqrt(mean_squared_error(y_test_original[:, 0], y_pred_original[:, 0]))
mae_position = mean_absolute_error(y_test_original[:, 0], y_pred_original[:, 0])
r2_position = r2_score(y_test_original[:, 0], y_pred_original[:, 0])

rmse_curvature = np.sqrt(mean_squared_error(y_test_original[:, 1], y_pred_original[:, 1]))
mae_curvature = mean_absolute_error(y_test_original[:, 1], y_pred_original[:, 1])
r2_curvature = r2_score(y_test_original[:, 1], y_pred_original[:, 1])

print(f"Position (cm) - RMSE: {rmse_position}")
print(f"Position (cm) - MAE: {mae_position}")
print(f"Position (cm) - R² Score: {r2_position}")
print(f"Curvature (mm^-1) - RMSE: {rmse_curvature}")
print(f"Curvature (mm^-1) - MAE: {mae_curvature}")
print(f"Curvature (mm^-1) - R² Score: {r2_curvature}")

metrics = {
    "Position (cm)": {
        "RMSE": rmse_position,
        "MAE": mae_position,
        "R² Score": r2_position
    },
    "Curvature (mm^-1)": {
        "RMSE": rmse_curvature,
        "MAE": mae_curvature,
        "R² Score": r2_curvature
    },
    "Best Hyperparameters": best_params
}

model_name = "mlp_multioutput_model"
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(base_dir, "models_output", f"{model_name}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

metrics_path = os.path.join(output_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {metrics_path}")

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
         'r--', label='Reference (y=x)')
plt.xlabel("True Position (cm)")
plt.ylabel("Predicted Position (cm)")
plt.title("Position (cm): True vs Predicted")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "scatter_position.png"))
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(y_test_original[:, 1], y_pred_original[:, 1], alpha=0.5, color='orange')
plt.plot([y_test_original[:, 1].min(), y_test_original[:, 1].max()],
         [y_test_original[:, 1].min(), y_test_original[:, 1].max()],
         'r--', label='Reference (y=x)')
plt.xlabel("True Curvature (mm^-1)")
plt.ylabel("Predicted Curvature (mm^-1)")
plt.title("Curvature (mm^-1): True vs Predicted")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "scatter_curvature.png"))
plt.close()

print(f"Scatter plots saved to {output_dir}")

print(f"All outputs saved to {output_dir}")

total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print("MLP Multi-output Regression training process completed!")
