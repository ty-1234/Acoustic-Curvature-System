import pandas as pd
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import time
import json
import matplotlib.pyplot as plt

print("Starting the SVR Regression training process...")
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
df = df.drop(columns=['Section'])  # Remove old 'Section' column
merged_csv_path = os.path.join(merged_files_dir, "merged_all.csv")
df.to_csv(merged_csv_path, index=False)
print(f"Merged CSV file saved to: {merged_csv_path}")

fft_columns = [f'FFT_{i}Hz' for i in range(100, 16001, 100)]
print("Step 2: Calculating baseline averages...")

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
def normalize_fft(row):
    return pd.Series({f'Normalized_{col}': row[col] - baseline_averages[col] for col in fft_columns})

normalized_fft = df.apply(normalize_fft, axis=1).fillna(0)
df = pd.concat([df, normalized_fft], axis=1)


normalized_fft_columns = [f'Normalized_{col}' for col in fft_columns]

print("Filtering data where Clamp = 1")
df['Clamp'] = df['Clamp'].map({'open': 0, 'close': 1})
df.loc[df['Clamp'] == 0, 'Curvature'] = 0
df = df[df["Clamp"] == 1]

print("Handling missing and infinite values")
df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

print("Step 5: Splitting the data")
X = df[normalized_fft_columns]
y = df[["Position", "Curvature"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Step 6: Scaling features and targets")
scaler_features = MinMaxScaler()
X_train_scaled = scaler_features.fit_transform(X_train)
X_test_scaled = scaler_features.transform(X_test)

scaler_target = MinMaxScaler()
y_train_scaled = scaler_target.fit_transform(y_train)
y_test_scaled = scaler_target.transform(y_test)

print("Step 8: Defining the SVR model")
svr = SVR()
multi_svr = MultiOutputRegressor(svr)

print("Step 9: Performing RandomizedSearchCV")
param_distributions = {
    "estimator__kernel": ["rbf", "linear", "poly"],
    "estimator__C": [0.1, 1, 10, 100],
    "estimator__epsilon": [0.01, 0.1, 0.2, 0.5],
    "estimator__gamma": ["scale", "auto"]
}

random_search = RandomizedSearchCV(
    estimator=multi_svr,
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

print("Step 10: Training final model")
multi_svr = MultiOutputRegressor(
    SVR(
        kernel=best_params["estimator__kernel"],
        C=best_params["estimator__C"],
        epsilon=best_params["estimator__epsilon"],
        gamma=best_params["estimator__gamma"]
    )
)
multi_svr.fit(X_train_scaled, y_train_scaled)

print("Step 11: Making predictions")
y_pred_scaled = multi_svr.predict(X_test_scaled)
y_pred_original = scaler_target.inverse_transform(y_pred_scaled)
y_test_original = scaler_target.inverse_transform(y_test_scaled)

# Calculate metrics for Position (cm)
rmse_position = np.sqrt(mean_squared_error(y_test_original[:, 0], y_pred_original[:, 0]))
mae_position = mean_absolute_error(y_test_original[:, 0], y_pred_original[:, 0])
r2_position = r2_score(y_test_original[:, 0], y_pred_original[:, 0])

# Calculate metrics for Curvature (mm^-1)
rmse_curvature = np.sqrt(mean_squared_error(y_test_original[:, 1], y_pred_original[:, 1]))
mae_curvature = mean_absolute_error(y_test_original[:, 1], y_pred_original[:, 1])
r2_curvature = r2_score(y_test_original[:, 1], y_pred_original[:, 1])

print(f"Position (cm) - RMSE: {rmse_position:.4f}, MAE: {mae_position:.4f}, R²: {r2_position:.4f}")
print(f"Curvature (mm^-1) - RMSE: {rmse_curvature:.4f}, MAE: {mae_curvature:.4f}, R²: {r2_curvature:.4f}")

metrics = {
    "Position (cm)": {"RMSE": rmse_position, "MAE": mae_position, "R²": r2_position},
    "Curvature (mm^-1)": {"RMSE": rmse_curvature, "MAE": mae_curvature, "R²": r2_curvature},
    "Best Hyperparameters": best_params
}

model_name = "svr_multioutput_model"
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(base_dir, "models_output", f"{model_name}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

model_params = best_params.copy()
model_params.update({"test_size": 0.2, "random_state": 42})
with open(os.path.join(output_dir, "model_params.json"), "w") as f:
    json.dump(model_params, f, indent=2)

plt.figure(figsize=(8, 6))
plt.scatter(y_test_original[:, 0], y_pred_original[:, 0], alpha=0.5)
plt.plot([y_test_original[:, 0].min(), y_test_original[:, 0].max()],
         [y_test_original[:, 0].min(), y_test_original[:, 0].max()], 'r--')
plt.xlabel("True Position (cm)")
plt.ylabel("Predicted Position (cm)")
plt.title("Position (cm): True vs Predicted")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "scatter_position.png"))
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(y_test_original[:, 1], y_pred_original[:, 1], alpha=0.5, color='orange')
plt.plot([y_test_original[:, 1].min(), y_test_original[:, 1].max()],
         [y_test_original[:, 1].min(), y_test_original[:, 1].max()], 'r--')
plt.xlabel("True Curvature (mm^-1)")
plt.ylabel("Predicted Curvature (mm^-1)")
plt.title("Curvature (mm^-1): True vs Predicted")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "scatter_curvature.png"))
plt.close()

print(f"Outputs saved to {output_dir}")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")
