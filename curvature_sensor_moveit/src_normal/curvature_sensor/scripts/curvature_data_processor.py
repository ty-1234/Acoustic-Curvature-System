import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# This script is designed to perform feature normalization on FFT columns
# in the input CSV file. It supports two normalization methods:
# 1. Z-score normalization
# 2. Min-Max scaling

def process_curvature_data(
    curvature_value=0.01818, #That 0.01818 is just a placeholder default. It exists only so the function doesn’t crash if someone calls it without specifying a curvature_value.
    normalize_method='zscore'
):
    curvature_str = str(curvature_value).replace(".", "_")
    input_path = f'csv_data/raw/raw_{curvature_str}.csv'
    output_path = f'csv_data/preprocessed/processed_{curvature_str}.csv'

    # Check if file exists
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    # Normalize FFT columns
    fft_cols = [col for col in df.columns if 'FFT' in col]

    if normalize_method == 'zscore':
        scaler = StandardScaler()
        df[fft_cols] = scaler.fit_transform(df[fft_cols])
        print("📏 Normalized using Z-score")
    elif normalize_method == 'minmax':
        df[fft_cols] = (df[fft_cols] - df[fft_cols].min()) / (df[fft_cols].max() - df[fft_cols].min())
        print("📏 Normalized using Min-Max")
    else:
        print("⚠️ No normalization applied.")

    # Save processed data
    os.makedirs('csv_data/preprocessed', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to: {output_path}")

# ========== MAIN ==========
if __name__ == "__main__":
    # This script is executed as a standalone program to process curvature data
    curvature_value = float(input("Enter curvature value to process (e.g., 0.01818): "))
    process_curvature_data(curvature_value=curvature_value)
