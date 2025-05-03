import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from collections import defaultdict

# === Setup file paths relative to script location ===
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "..", "csv_data", "normalised")
output_dir = os.path.join(script_dir, "..", "csv_data", "preprocessed")
os.makedirs(output_dir, exist_ok=True)

# === Helper Functions ===
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
    fft_cols = [col for col in df.columns if col.startswith("FFT_") and not col.endswith("_Norm")]
    if len(fft_cols) == 0:
        fft_cols = [col for col in df.columns if col.startswith("FFT_")]
    df['FFT_Peak_Index'] = df[fft_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(float)
    return df

def compute_pca_features(df, fft_cols, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df[fft_cols])
    for i in range(n_components):
        df[f'PC{i+1}'] = principal_components[:, i]
    return df

# === Process each CSV in the input directory ===
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path)

        # Identify normalized FFT columns
        fft_norm_cols = [col for col in df.columns if col.endswith("_Norm") and col.startswith("FFT_")]
        fft_cols = [col.replace("_Norm", "") for col in fft_norm_cols]
        df.rename(columns={f: t for f, t in zip(fft_norm_cols, fft_cols)}, inplace=True)

        # Only keep renamed normalized columns
        df = df[[col for col in df.columns if col in fft_cols or not col.startswith("FFT_")]]

        # Compute Curvature_Active BEFORE modifying curvature
        df['Curvature_Active'] = 0  # Set default value to 0
        df.loc[df[['Position_cm', 'Curvature']].notna().all(axis=1), 'Curvature_Active'] = 1

        # === Patch: Fix curvature labels ===
        first_active_idx = df[df['Curvature_Active'] == 1].index.min()
        if pd.notna(first_active_idx):
            # Before first press: set curvature = 0
            df.loc[:first_active_idx - 1, 'Curvature'] = 0.0

            # Mark transition rows (rows after first active press where Curvature_Active is 0)
            transition_mask = (df.index > first_active_idx) & (df['Curvature_Active'] == 0)
            df.loc[transition_mask, 'Curvature_Active'] = 2  # Changed from 3 to 2
            df.loc[df['Curvature_Active'] == 2, 'Curvature'] = 0.0  # Changed from 3 to 2
        else:
            # No active curvature found — fallback to setting all as 0
            df['Curvature'] = 0.0

        # === Continue feature engineering ===
        df = compute_band_features(df)
        df = compute_fft_peak_index(df)
        df = compute_pca_features(df, fft_cols, n_components=2)

        # Final feature selection
        output_cols = fft_cols + [
            'Low_Band_Mean', 'Mid_Band_Mean', 'High_Band_Mean',
            'Mid_to_Low_Band_Ratio', 'High_to_Mid_Band_Ratio',
            'FFT_Peak_Index', 'PC1', 'PC2',
            'Curvature', 'Position_cm', 'Curvature_Active'
        ]
        df_final = df[output_cols]

        # Save
        output_filename = filename.replace("normalized", "preprocessed").replace("merged_", "")
        output_path = os.path.join(output_dir, output_filename)
        df_final.to_csv(output_path, index=False)
        print(f"Processed and saved: {output_filename}")

# === Group and merge preprocessed files by test number ===
# Find all preprocessed files in output_dir
preprocessed_files = [f for f in os.listdir(output_dir) if f.endswith(".csv")]

# Group by test number using filename pattern: looks for [test X] in filename
test_groups = defaultdict(list)
for file in preprocessed_files:
    if "[test " in file:
        test_id = file.split("[test ")[1].split("]")[0]
        test_groups[test_id].append(file)

# Output directory for merged test files
merged_dir = os.path.join(output_dir, "merged_tests")
os.makedirs(merged_dir, exist_ok=True)

# Concatenate and save merged test files
for test_id, files in test_groups.items():
    dfs = []
    for file in sorted(files):  # sort for consistency
        df = pd.read_csv(os.path.join(output_dir, file))
        dfs.append(df)
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_filename = f"test_{int(test_id):02d}.csv"
    merged_path = os.path.join(merged_dir, merged_filename)
    merged_df.to_csv(merged_path, index=False)
    print(f"Merged test_{test_id} saved to: {merged_filename}")