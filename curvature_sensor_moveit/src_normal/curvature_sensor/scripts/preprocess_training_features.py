import os
import pandas as pd
import numpy as np

def process_curvature_data(curvature_value=None):
    """
    Process the curvature data to create engineered features.
    This function signature matches what main.py expects to call.

    Parameters:
    -----------
    curvature_value : float, optional
        Not used in this implementation but kept for compatibility with main.py
    """
    # === Path Config ===
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, "csv_data", "combined_dataset.csv")
    output_path = os.path.join(project_root, "csv_data", "preprocessed", "preprocessed_training_dataset.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # === Load Raw Dataset ===
    print("📥 Loading raw dataset from:", raw_data_path)
    df = pd.read_csv(raw_data_path)
    print("📄 Available columns:", df.columns.tolist())

    # Fill in Position_cm only for active pressing sections
    if "Section" in df.columns and df["Section"].notna().any():
        print("🔄 Filling Section labels and converting to Position_cm...")
        df["Section"] = df["Section"].fillna(method="ffill")
        df["Position_cm"] = df["Section"].str.replace("cm", "").astype(float, errors='ignore')
    else:
        print("⚠️ 'Section' column is missing or empty — 'Position_cm' will remain NaN")

    # === Extract FFT Columns ===
    fft_cols = [col for col in df.columns if col.startswith("FFT_")]
    fft_data = df[fft_cols]

    # === Feature Engineering ===
    features = pd.DataFrame()

    # 1. Original FFT features
    features[fft_cols] = fft_data

    # 2. Statistical features
    features["FFT_Mean"] = fft_data.mean(axis=1)
    features["FFT_Std"] = fft_data.std(axis=1)
    features["FFT_Min"] = fft_data.min(axis=1)
    features["FFT_Max"] = fft_data.max(axis=1)
    features["FFT_Range"] = features["FFT_Max"] - features["FFT_Min"]

    # 3. Band groupings
    low_band = fft_data.iloc[:, 0:3]   # 200–600 Hz
    mid_band = fft_data.iloc[:, 3:7]   # 800–1400 Hz
    high_band = fft_data.iloc[:, 7:]  # 1600–2000 Hz

    features["Low_Band_Mean"] = low_band.mean(axis=1)
    features["Mid_Band_Mean"] = mid_band.mean(axis=1)
    features["High_Band_Mean"] = high_band.mean(axis=1)

    # 4. Targets
    features["Curvature_Label"] = df["Curvature_Label"]
    features["RunID"] = df["RunID"]

    # Ensure Position_cm is included in output features
    if "Position_cm" in df.columns:
        features["Position_cm"] = df["Position_cm"]

    # === Save Output ===
    features.to_csv(output_path, index=False)
    print("✅ Preprocessed dataset saved to:", output_path)

    return True  # Return success


def main():
    """
    Main function for direct execution of the script.
    """
    process_curvature_data()


if __name__ == "__main__":
    main()