import os
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Setup ===
script_dir = os.path.dirname(os.path.abspath(__file__))
# Updated path to use merged_tests directory
data_dir = os.path.join(script_dir, "..", "..", "csv_data", "preprocessed", "merged_tests")
output_dir = os.path.join(script_dir, "analysis_output", "curvature")
os.makedirs(output_dir, exist_ok=True)

# === Ask for test set to use ===
print("Available datasets:")
# Get all CSV files in the merged_tests directory
available = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

# Display available options with numbers
for i, filename in enumerate(available, 1):
    print(f"{i}. {filename}")

# Get user selection by number
while True:
    try:
        chosen_idx = int(input("\nEnter the number of the test set to use (1-{}): ".format(len(available)))) - 1
        if 0 <= chosen_idx < len(available):
            chosen_file = available[chosen_idx]
            break
        else:
            print(f"Please enter a number between 1 and {len(available)}")
    except ValueError:
        print("Please enter a valid number")

# === Load the selected CSV file ===
print("Loading data file:")
df_path = os.path.join(data_dir, chosen_file)
df = pd.read_csv(df_path)
print(f"Loaded: {chosen_file} with {len(df)} rows")

print(f"\nTotal samples: {len(df)}")

# === Keep all rows for curvature training (including rest state) ===
# No filtering — curvature is 0 when Curvature_Active == 0

# === Feature and Target Selection ===
feature_cols = [
    'FFT_200Hz', 'FFT_400Hz', 'FFT_600Hz', 'FFT_800Hz', 'FFT_1000Hz',
    'FFT_1200Hz', 'FFT_1400Hz', 'FFT_1600Hz', 'FFT_1800Hz', 'FFT_2000Hz',
    'Low_Band_Mean', 'Mid_Band_Mean', 'High_Band_Mean',
    'Mid_to_Low_Band_Ratio', 'High_to_Mid_Band_Ratio',
    'FFT_Peak_Index', 'PC1', 'PC2'
]

X = df[feature_cols].values
y_curvature = df['Curvature'].values

# === Subsample for GPR (optional) ===
MAX_SAMPLES = 2000
if len(X) > MAX_SAMPLES:
    X, _, y_curvature, _ = train_test_split(X, y_curvature, train_size=MAX_SAMPLES, random_state=42)

# === Train GPR with ARD for Curvature ===
print("Beginning GPR training (this may take some time)...")
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

# Create progress bar for GPR training
pbar = tqdm(total=100, desc="GPR Training")
gpr.fit(X, y_curvature)
pbar.update(100)  # Update to 100% when done
pbar.close()
print("GPR training complete.")

# === Extract ARD-based feature importances ===
length_scales = gpr.kernel_.k1.k2.length_scale
feature_importance = 1 / length_scales

feature_df = pd.DataFrame({
    'Feature': feature_cols,
    'Length_Scale': length_scales,
    'ARD_Importance': feature_importance
}).sort_values(by='ARD_Importance', ascending=False)

# Save feature importance to CSV - use chosen_file name without extension for output
test_id = os.path.splitext(chosen_file)[0]
importance_path = os.path.join(output_dir, f"gpr_ard_feature_importance_{test_id}.csv")
feature_df.to_csv(importance_path, index=False)
print(f"Saved ARD-based feature importance to: {importance_path}")

# === Predict on Training Data ===
print("Generating predictions...")
y_pred, y_std = gpr.predict(X, return_std=True)

# === Save prediction results to CSV ===
output_df = pd.DataFrame({
    'True_Curvature': y_curvature,
    'Predicted_Curvature': y_pred,
    'Uncertainty_1STD': y_std
})

output_filename = f"gpr_output_{test_id}_curvature.csv"
output_path = os.path.join(output_dir, output_filename)
output_df.to_csv(output_path, index=False)
print(f"\nSaved prediction CSV to: {output_path}")

# === Sort by true curvature for better visual tracking ===
sorted_df = output_df.sort_values(by='True_Curvature').reset_index(drop=True)

# === Plot Results ===
plt.figure(figsize=(10, 6))
plt.plot(sorted_df['True_Curvature'], label='True Curvature', alpha=0.6)
plt.plot(sorted_df['Predicted_Curvature'], label='Predicted Curvature', alpha=0.8)
plt.fill_between(
    np.arange(len(sorted_df)),
    sorted_df['Predicted_Curvature'] - sorted_df['Uncertainty_1STD'],
    sorted_df['Predicted_Curvature'] + sorted_df['Uncertainty_1STD'],
    color='gray', alpha=0.3, label='Uncertainty')

plt.title(f"GPR Prediction of Curvature ({test_id})")
plt.xlabel("Sorted Sample Index")
plt.ylabel("Curvature (m⁻¹)")
plt.legend()
plt.tight_layout()
plt.show()
