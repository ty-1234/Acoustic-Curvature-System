"""
Individual Test Object Preprocessor

This script processes trimmed sensor data files and generates feature-enriched test object files.
It adds derived features from FFT data and categorizes data points based on curvature activity.

Features added:
- Frequency band averages (low, mid, high)
- Band ratios (mid-to-low, high-to-mid)
- FFT peak frequency index
- Principal components (PCA) of FFT data
- Position classification

The script:
1. Loads each trimmed CSV file from the input directory
2. Extracts the curvature value from the filename
3. Maps each curvature value to a test object identifier (A, B, C, etc.)
4. Computes additional features from the FFT data
5. Saves the enriched data to individual test object files

Output files follow the naming convention: test_object_X[test N].csv
where X is a letter (A, B, C...) assigned based on curvature value.

Author: Bipindra Rai
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import re

# === Setup file paths relative to script location ===
script_dir = os.path.dirname(os.path.abspath(__file__))
# input_dir = os.path.join(script_dir, "..", "csv_data", "trimmed_data")
# output_dir = os.path.join(script_dir, "..", "csv_data", "preprocessed")

input_dir = os.path.join(script_dir, "..", "csv_data", "trimmed_downsampled")

output_dir = os.path.join(script_dir, "..", "csv_data", "downsized_preprocessed")
os.makedirs(output_dir, exist_ok=True)

# === Helper Functions ===
def compute_band_features(df):
    """
    Compute frequency band features from FFT columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing FFT_*Hz columns
        
    Returns:
        pd.DataFrame: DataFrame with added band features:
            - Low_Band_Mean (200-600Hz average)
            - Mid_Band_Mean (800-1200Hz average)
            - High_Band_Mean (1400-2000Hz average)
            - Mid_to_Low_Band_Ratio
            - High_to_Mid_Band_Ratio
    """
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
    """
    Compute the peak frequency from FFT columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing FFT_*Hz columns
        
    Returns:
        pd.DataFrame: DataFrame with added FFT_Peak_Index column
            containing the frequency (in Hz) with the highest amplitude
    """
    fft_cols = [col for col in df.columns if col.startswith("FFT_")]
    df['FFT_Peak_Index'] = df[fft_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(float)
    return df

def compute_pca_features(df, fft_cols, n_components=2):
    """
    Compute principal components from FFT data.
    
    Args:
        df (pd.DataFrame): DataFrame containing FFT data
        fft_cols (list): List of FFT column names
        n_components (int): Number of principal components to extract
        
    Returns:
        pd.DataFrame: DataFrame with added PCx columns (e.g., PC1, PC2)
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df[fft_cols])
    for i in range(n_components):
        df[f'PC{i+1}'] = principal_components[:, i]
    return df

# === Build curvature -> test_object_X mapping ===
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

# Print the mapping between test objects and curvature values
print("\n=== Test Object to Curvature Mapping ===")
for curvature, test_obj in sorted(curv_to_obj.items()):
    print(f"{test_obj}: {curvature} mm^-1")
print()

# === Process each CSV in the input directory ===
for filename in os.listdir(input_dir):
    if filename.endswith(".csv") and filename in filename_map:
        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path)

        fft_cols = [col for col in df.columns if col.startswith("FFT_")]

        df['Curvature_Active'] = df[['Position_cm', 'Curvature']].notna().all(axis=1).astype(int)

        first_active_idx = df[df['Curvature_Active'] == 1].index.min()
        if pd.notna(first_active_idx):
            df.loc[:first_active_idx - 1, 'Curvature'] = 0.0
            df = df[(df['Curvature_Active'] == 1) | (df.index < first_active_idx)]
        else:
            df['Curvature'] = 0.0

        df = compute_band_features(df)
        df = compute_fft_peak_index(df)
        df = compute_pca_features(df, fft_cols, n_components=2)

        df['Position_Class'] = df.apply(
            lambda row: str(int(row['Position_cm'])) if row['Curvature_Active'] == 1 else 'rest', axis=1
        )

        output_cols = fft_cols + [
            'Low_Band_Mean', 'Mid_Band_Mean', 'High_Band_Mean',
            'Mid_to_Low_Band_Ratio', 'High_to_Mid_Band_Ratio',
            'FFT_Peak_Index', 'PC1', 'PC2',
            'Curvature', 'Position_cm', 'Curvature_Active', 'Position_Class'
        ]
        df_final = df[output_cols]

        curvature_val = filename_map[filename]
        object_prefix = curv_to_obj[curvature_val]
        test_part = filename.split("[")[1]  # preserves [test N].csv
        output_filename = f"{object_prefix}[{test_part}"
        output_path = os.path.join(output_dir, output_filename)
        df_final.to_csv(output_path, index=False)
        print(f"Processed and saved: {output_filename}")

print("\n✅ Processing complete. Individual test object files created.")
