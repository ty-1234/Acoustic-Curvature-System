"""
Module: preprocessor_indv.py
Author: [Your Name/Group Name]
Date: 9 May 2025

Description:
This script is designed for preprocessing individual CSV files containing sensor data,
specifically for a curvature sensor project. It performs several key operations:
1.  Reads sensor data from CSV files located in a specified input directory.
2.  Extracts and maps curvature values from filenames to provide context for the data.
3.  Computes various features from Fast Fourier Transform (FFT) data, including:
    - Mean values for predefined low, mid, and high frequency bands.
    - Ratios between these band means.
    - The frequency index corresponding to the peak FFT magnitude.
    - Principal Component Analysis (PCA) features to reduce dimensionality.
4.  Differentiates between 'active' (sensor interacting with a surface) and 'idle'
    (sensor not actively measuring) states based on data validity.
5.  Processes and saves active data grouped by sensor position, assigning a position class.
6.  Processes and saves idle data separately, assigning an 'idle' class.
7.  Generates new CSV files in a specified output directory, containing the
    original FFT data along with the newly computed features and class labels.

This preprocessing step is crucial for preparing the sensor data for subsequent
machine learning model training or data analysis tasks.
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import re

# === Setup file paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
# input_dir = os.path.join(script_dir, "..", "csv_data", "trimmed_data")
# output_dir = os.path.join(script_dir, "..", "csv_data", "preprocessed")

input_dir = os.path.join(script_dir, "..", "csv_data", "trimmed_downsampled")

output_dir = os.path.join(script_dir, "..", "csv_data", "downsized_preprocessed")
os.makedirs(output_dir, exist_ok=True)

# === Feature Functions ===
def compute_band_features(df):
    low_band = ['FFT_200Hz', 'FFT_400Hz', 'FFT_600Hz']
    mid_band = ['FFT_800Hz', 'FFT_1000Hz', 'FFT_1200Hz']
    high_band = ['FFT_1400Hz', 'FFT_1600Hz', 'FFT_1800Hz', 'FFT_2000Hz']
    df['Low_Band_Mean'] = df[low_band].mean(axis=1)
    df['Mid_Band_Mean'] = df[mid_band].mean(axis=1)
    df['High_Band_Mean'] = df[high_band].mean(axis=1)
    df['Mid_to_Low_Band_Ratio'] = df['Mid_Band_Mean'] / df['Low_Band_Mean']
    df['High_to_Mid_Band_Ratio'] = df['High_Band_Mean'] / df['Mid_Band_Mean']
    return df

def compute_fft_peak_index(df):
    fft_cols = [col for col in df.columns if col.startswith("FFT_")]
    df['FFT_Peak_Index'] = df[fft_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(float)
    return df

def compute_pca_features(df, fft_cols, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df[fft_cols])
    for i in range(n_components):
        df[f'PC{i+1}'] = principal_components[:, i]
    return df

# === Curvature Mapping Setup ===
curvature_map = {}
all_curvatures = []
filename_map = {}

for filename in os.listdir(input_dir):
    if filename.endswith(".csv") and "[test " in filename:
        match = re.search(r"trimmed_merged_(.*?)\[test", filename)
        if match:
            curvature_str = match.group(1)
            curvature_val = float(curvature_str.replace("_", "."))
            all_curvatures.append(curvature_val)
            filename_map[filename] = curvature_val

sorted_curvatures = sorted(set(all_curvatures))
curv_to_obj = {c: f"test_object_{chr(65+i)}" for i, c in enumerate(sorted_curvatures)}

print("\n=== Test Object to Curvature Mapping ===")
for curvature, label in sorted(curv_to_obj.items()):
    print(f"{label}: {curvature} mm^-1")
print()

# === Main Preprocessing Loop ===
for filename in os.listdir(input_dir):
    if filename.endswith(".csv") and filename in filename_map:
        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path)

        if 'Position_cm' not in df.columns or 'Curvature' not in df.columns:
            continue

        df['Curvature_Active'] = df[['Position_cm', 'Curvature']].notna().all(axis=1).astype(int)
        curvature_val = filename_map[filename]
        object_prefix = curv_to_obj[curvature_val]
        test_part = filename.split("[")[1]  # [test 1].csv

        # === Handle Active Rows (Curvature_Active == 1) ===
        df_active = df[df['Curvature_Active'] == 1]
        if not df_active.empty:
            for pos_val, group in df_active.groupby('Position_cm'):
                group = group.copy()
                fft_cols = [col for col in group.columns if col.startswith("FFT_")]

                group = compute_band_features(group)
                group = compute_fft_peak_index(group)
                group = compute_pca_features(group, fft_cols, n_components=2)

                group['Position_Class'] = str(int(pos_val))

                output_cols = fft_cols + [
                    'Low_Band_Mean', 'Mid_Band_Mean', 'High_Band_Mean',
                    'Mid_to_Low_Band_Ratio', 'High_to_Mid_Band_Ratio',
                    'FFT_Peak_Index', 'PC1', 'PC2',
                    'Curvature', 'Position_cm', 'Curvature_Active', 'Position_Class'
                ]
                df_final = group[output_cols]

                output_filename = f"{object_prefix}_pos{int(pos_val)}[{test_part}"
                output_path = os.path.join(output_dir, output_filename)
                df_final.to_csv(output_path, index=False)
                print(f"Saved: {output_filename}")

        # === Handle Idle Rows (Curvature_Active == 0) ===
        df_idle = df[df['Curvature_Active'] == 0].copy()
        if not df_idle.empty:
            fft_cols = [col for col in df_idle.columns if col.startswith("FFT_")]
            df_idle = compute_band_features(df_idle)
            df_idle = compute_fft_peak_index(df_idle)
            df_idle = compute_pca_features(df_idle, fft_cols, n_components=2)
            df_idle['Position_Class'] = 'rest'

            output_cols = fft_cols + [
                'Low_Band_Mean', 'Mid_Band_Mean', 'High_Band_Mean',
                'Mid_to_Low_Band_Ratio', 'High_to_Mid_Band_Ratio',
                'FFT_Peak_Index', 'PC1', 'PC2',
                'Curvature', 'Position_cm', 'Curvature_Active', 'Position_Class'
            ]
            df_final_idle = df_idle[output_cols]

            output_filename = f"{object_prefix}_IDLE[{test_part}"
            output_path = os.path.join(output_dir, output_filename)
            df_final_idle.to_csv(output_path, index=False)
            print(f"Saved: {output_filename}")

print("\n✅ All test object position + IDLE files created.")
