import os
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Setup paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "..", "csv_data", "preprocessed", "merged_tests")
output_dir = os.path.join(script_dir, "analysis_output", "position")
os.makedirs(output_dir, exist_ok=True)

# === Dataset selection ===
print("Available datasets:")
available = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
for i, filename in enumerate(available, 1):
    print(f"{i}. {filename}")

while True:
    try:
        idx = int(input(f"\nEnter the number of the test set to use (1-{len(available)}): ")) - 1
        if 0 <= idx < len(available):
            chosen_file = available[idx]
            break
        else:
            print(f"Please enter a number between 1 and {len(available)}")
    except ValueError:
        print("Please enter a valid number")

# === Load selected file ===
df_path = os.path.join(data_dir, chosen_file)
df = pd.read_csv(df_path)
print(f"Loaded {chosen_file} with {len(df)} rows")

# === Keep only rows where Position_cm is valid (i.e., Curvature_Active == 1) ===
df = df[df['Curvature_Active'] == 1].copy()
print(f"Filtered to {len(df)} active rows for position prediction")

# === Feature and Target Selection ===
feature_cols = [
    'FFT_200Hz', 'FFT_400Hz', 'FFT_600Hz', 'FFT_800Hz', 'FFT_1000Hz',
    'FFT_1200Hz', 'FFT_1400Hz', 'FFT_1600Hz', 'FFT_1800Hz', 'FFT_2000Hz',
    'Low_Band_Mean', 'Mid_Band_Mean', 'High_Band_Mean',
    'Mid_to_Low_Band_Ratio', 'High_to_Mid_Band_Ratio',
    'FFT_Peak_Index', 'PC1', 'PC2'
]

X = df[feature_cols].values
y_position = df['Position_cm'].values

# === Optional Subsampling ===
MAX_SAMPLES = 2000
if len(X) > MAX_SAMPLES:
    X, _, y_position, _ = train_test_split(X, y_position, train_size=MAX_SAMPLES, random_state=42)

# === GPR Model ===
print("Training GPR for position prediction...")
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
pbar = tqdm(total=100, desc="Training")
for _ in range(10):
    pbar.update(10)
    if _ == 0:
        gpr.fit(X, y_position)
pbar.close()
print("GPR training complete.")

# === Feature Importance ===
length_scales = gpr.kernel_.k1.k2.length_scale
feature_importance = 1 / length_scales
feature_df = pd.DataFrame({
    'Feature': feature_cols,
    'Length_Scale': length_scales,
    'ARD_Importance': feature_importance
}).sort_values(by='ARD_Importance', ascending=False)

basename = os.path.splitext(chosen_file)[0]
importance_path = os.path.join(output_dir, f"gpr_ard_feature_importance_{basename}_position.csv")
feature_df.to_csv(importance_path, index=False)
print(f"Saved feature importances to: {importance_path}")

# === Predict and Save Results ===
y_pred, y_std = gpr.predict(X, return_std=True)
output_df = pd.DataFrame({
    'True_Position_cm': y_position,
    'Predicted_Position_cm': y_pred,
    'Uncertainty_1STD': y_std
})

output_csv = os.path.join(output_dir, f"gpr_output_{basename}_position.csv")
output_df.to_csv(output_csv, index=False)
print(f"Saved prediction results to: {output_csv}")

# === Plot ===
sorted_df = output_df.sort_values(by='True_Position_cm').reset_index(drop=True)
plt.figure(figsize=(10, 6))
plt.plot(sorted_df['True_Position_cm'], label='True Position', alpha=0.6)
plt.plot(sorted_df['Predicted_Position_cm'], label='Predicted Position', alpha=0.8)
plt.fill_between(
    np.arange(len(sorted_df)),
    sorted_df['Predicted_Position_cm'] - sorted_df['Uncertainty_1STD'],
    sorted_df['Predicted_Position_cm'] + sorted_df['Uncertainty_1STD'],
    color='gray', alpha=0.3, label='Uncertainty')

plt.title(f"GPR Prediction of Position ({basename})")
plt.xlabel("Sorted Sample Index")
plt.ylabel("Position (cm)")
plt.legend()
plt.tight_layout()
plt.show()
