import os
import pandas as pd

# === Setup file paths ===
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(base_dir, "csv_data", "combined_dataset.csv")
output_path = os.path.join(base_dir, "csv_data", "preprocessed", "preprocessed_training_dataset.csv")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === Load dataset ===
df = pd.read_csv(input_path)
fft_cols = [col for col in df.columns if col.startswith("FFT_")]

# === Determine if curvature is being applied ===
df["Curvature_Active"] = df[["Section", "PosX", "PosY", "PosZ"]].notna().all(axis=1).astype(int)

# === Clean and convert Position ===
df.loc[df["Curvature_Active"] == 1, "Position_cm"] = df.loc[df["Curvature_Active"] == 1, "Section"].str.replace("cm", "").astype(float)

# === FFT Feature Engineering ===
feature_set = pd.DataFrame()
feature_set[fft_cols] = df[fft_cols]

# Add statistical features
feature_set["FFT_Mean"] = df[fft_cols].mean(axis=1)
feature_set["FFT_Std"] = df[fft_cols].std(axis=1)
feature_set["FFT_Min"] = df[fft_cols].min(axis=1)
feature_set["FFT_Max"] = df[fft_cols].max(axis=1)
feature_set["FFT_Range"] = feature_set["FFT_Max"] - feature_set["FFT_Min"]

# Band groups
low_band = df[fft_cols].iloc[:, 0:3]    # 200–600 Hz
mid_band = df[fft_cols].iloc[:, 3:7]    # 800–1400 Hz
high_band = df[fft_cols].iloc[:, 7:]   # 1600–2000 Hz

feature_set["Low_Band_Mean"] = low_band.mean(axis=1)
feature_set["Mid_Band_Mean"] = mid_band.mean(axis=1)
feature_set["High_Band_Mean"] = high_band.mean(axis=1)

# Add band ratio features
feature_set["Mid_to_Low_Band_Ratio"] = feature_set["Mid_Band_Mean"] / feature_set["Low_Band_Mean"].replace(0, 1e-6)
feature_set["High_to_Mid_Band_Ratio"] = feature_set["High_Band_Mean"] / feature_set["Mid_Band_Mean"].replace(0, 1e-6)

# FFT peak index feature
feature_set["FFT_Peak_Index"] = df[fft_cols].idxmax(axis=1).str.extract("(\d+)").astype(int)

# Add target and metadata columns
feature_set["Position_cm"] = df["Position_cm"]
feature_set["Curvature_Label"] = df["Curvature_Label"]
feature_set["RunID"] = df["RunID"]
feature_set["Curvature_Active"] = df["Curvature_Active"]

# === Save preprocessed dataset ===
feature_set.to_csv(output_path, index=False)
print(f"✅ Preprocessed dataset saved to: {output_path}")