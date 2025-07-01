import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import time
import json
import matplotlib.pyplot as plt

print("Starting the GPR Regression training process...")
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
df['Position'] = df['Section'].astype(str).str.replace('cm', '').astype(float)  # Changed 'Section' to 'Position'
df = df.drop(columns=['Section'])  # Remove old 'Section' column
# Save merged CSV file
merged_csv_path = os.path.join(merged_files_dir, "merged_all.csv")
df.to_csv(merged_csv_path, index=False)
print(f"Merged CSV file saved to: {merged_csv_path}")

# Define FFT columns
fft_columns = [f'FFT_{i}Hz' for i in range(100, 16001, 100)]

# Calculate baseline
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

# Normalize FFT
print("\nStep 3: Normalizing the FFT values using the baseline")
def normalize_fft(row):
    return pd.Series({f'Normalized_{col}': row[col] - baseline_averages.get(col, 0) for col in fft_columns})

normalized_fft = df.apply(normalize_fft, axis=1).fillna(0)
df = pd.concat([df, normalized_fft], axis=1)

normalized_fft_columns = [f'Normalized_{col}' for col in fft_columns]

# Clamp filtering
print("Filtering data where Clamp = 1")
if 'Clamp' in df.columns:
    df['Clamp'] = df['Clamp'].map({'open': 0, 'close': 1})
    df.loc[df['Clamp'] == 0, 'Curvature'] = 0
df = df[df["Clamp"] == 1]
print(f"Remaining rows: {df.shape[0]}")

# Handle missing/infinite values
print("Handling missing and infinite values")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(method='ffill').fillna(method='bfill')

# Split data
print("Step 5: Splitting the data into training and testing sets before scaling")
X = df[normalized_fft_columns]
y = df[["Position", "Curvature"]]  # Changed 'Section' to 'Position'
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Scale features
print("Step 6: Scaling the normalized FFT columns (features)")
scaler_features = MinMaxScaler()
X_train_scaled = scaler_features.fit_transform(X_train)
X_test_scaled = scaler_features.transform(X_test)

# Step 8: Define and train GPR with RandomizedSearchCV and 10-fold CV
print("\nStep 8: Defining the GaussianProcessRegressor model with RandomizedSearchCV and 10-fold cross-validation")

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-2, normalize_y=True)

param_dist = {
    "estimator__alpha": [1e-2, 1e-3, 1e-4],
    "estimator__kernel": [
        C(1.0, (1e-3, 1e3)) * RBF(length_scale=l, length_scale_bounds=(1e-2, 1e2))
        for l in [0.1, 1.0, 10.0]
    ],
    "estimator__n_restarts_optimizer": [0, 2, 5]
}

multi_gpr = MultiOutputRegressor(gpr)

cv = KFold(n_splits=10, shuffle=True, random_state=42)

random_search = RandomizedSearchCV(
    multi_gpr,
    param_distributions=param_dist,
    n_iter=10,
    cv=cv,
    verbose=2,
    n_jobs=-1,
    scoring='neg_mean_squared_error',
    random_state=42
)

print("Fitting RandomizedSearchCV with MultiOutputRegressor and GaussianProcessRegressor")
model_start = time.time()
random_search.fit(X_train_scaled, y_train)
print(f"Best parameters found: {random_search.best_params_}")
print(f"Model trained in {time.time() - model_start:.2f} seconds")

# Use the best estimator for prediction
best_model = random_search.best_estimator_

# Step 11: Prediction
print("\nStep 11: Making predictions on the test set")
y_pred = best_model.predict(X_test_scaled)

# Metrics
rmse_position = np.sqrt(mean_squared_error(y_test["Position"], y_pred[:, 0]))
mae_position = mean_absolute_error(y_test["Position"], y_pred[:, 0])
r2_position = r2_score(y_test["Position"], y_pred[:, 0])

rmse_curvature = np.sqrt(mean_squared_error(y_test["Curvature"], y_pred[:, 1]))
mae_curvature = mean_absolute_error(y_test["Curvature"], y_pred[:, 1])
r2_curvature = r2_score(y_test["Curvature"], y_pred[:, 1])

print(f"Position - RMSE: {rmse_position}")
print(f"Position - MAE: {mae_position}")
print(f"Position - R² Score: {r2_position}")
print(f"Curvature - RMSE: {rmse_curvature}")
print(f"Curvature - MAE: {mae_curvature}")
print(f"Curvature - R² Score: {r2_curvature}")

# Save metrics
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
    "Model Type": "GaussianProcessRegressor"
}

timestamp = time.strftime("%Y%m%d_%H%M%S")
model_name = "gpr_multioutput_model"
output_dir = os.path.join(base_dir, "models_output", f"{model_name}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {output_dir}")

# Save best hyperparameters
def make_json_serializable(params):
    serializable = {}
    for k, v in params.items():
        try:
            json.dumps(v)
            serializable[k] = v
        except TypeError:
            serializable[k] = str(v)
    return serializable

best_params_serializable = make_json_serializable(random_search.best_params_)
with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
    json.dump(best_params_serializable, f, indent=2)
print(f"Best hyperparameters saved to {output_dir}")

# Save scatter plots
plt.figure(figsize=(8, 6))
plt.scatter(y_test["Position"], y_pred[:, 0], alpha=0.5)
min_pos = min(y_test["Position"].min(), y_pred[:, 0].min())
max_pos = max(y_test["Position"].max(), y_pred[:, 0].max())
plt.plot([min_pos, max_pos], [min_pos, max_pos], 'r--', label='y = x')  # Reference line
plt.xlabel("True Position")
plt.ylabel("Predicted Position")
plt.title("Position: True vs Predicted")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "scatter_position.png"))
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(y_test["Curvature"], y_pred[:, 1], alpha=0.5, color='orange')
min_curv = min(y_test["Curvature"].min(), y_pred[:, 1].min())
max_curv = max(y_test["Curvature"].max(), y_pred[:, 1].max())
plt.plot([min_curv, max_curv], [min_curv, max_curv], 'r--', label='y = x')  # Reference line
plt.xlabel("True Curvature")
plt.ylabel("Predicted Curvature")
plt.title("Curvature: True vs Predicted")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir , "scatter_curvature.png"))
plt.close()

print(f"Scatter plots saved to {output_dir}")

total_time = time.time() - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print("GPR Multi-output Regression training process completed!")
