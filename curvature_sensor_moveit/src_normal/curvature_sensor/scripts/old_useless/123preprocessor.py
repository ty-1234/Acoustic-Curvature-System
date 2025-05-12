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
def compute_band_features(df, prefix=""):
    low_band = [f'{prefix}FFT_200Hz', f'{prefix}FFT_400Hz', f'{prefix}FFT_600Hz']
    mid_band = [f'{prefix}FFT_800Hz', f'{prefix}FFT_1000Hz', f'{prefix}FFT_1200Hz']
    high_band = [f'{prefix}FFT_1400Hz', f'{prefix}FFT_1600Hz', f'{prefix}FFT_1800Hz', f'{prefix}FFT_2000Hz']
    df[f'{prefix}Low_Band_Mean'] = df[low_band].mean(axis=1)
    df[f'{prefix}Mid_Band_Mean'] = df[mid_band].mean(axis=1)
    df[f'{prefix}High_Band_Mean'] = df[high_band].mean(axis=1)
    df[f'{prefix}Mid_to_Low_Band_Ratio'] = df[f'{prefix}Mid_Band_Mean'] / df[f'{prefix}Low_Band_Mean']
    df[f'{prefix}High_to_Mid_Band_Ratio'] = df[f'{prefix}High_Band_Mean'] / df[f'{prefix}Mid_Band_Mean']
    return df

def compute_fft_peak_index(df, prefix=""):
    fft_cols = [col for col in df.columns if col.startswith(f"{prefix}FFT_")]
    df[f'{prefix}FFT_Peak_Index'] = df[fft_cols].idxmax(axis=1).str.extract(r'(\d+)').astype(float)
    return df

def compute_pca_features(df, fft_cols, prefix="Norm_", n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df[fft_cols])
    for i in range(n_components):
        df[f'{prefix}PC{i+1}'] = principal_components[:, i]
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

        print(f"\nProcessing: {filename}")
        df['Curvature_Active'] = df[['Position_cm', 'Curvature']].notna().all(axis=1).astype(int)
        curvature_val = filename_map[filename]
        object_prefix = curv_to_obj[curvature_val]
        
        # Add test object information
        df['Test_Object'] = object_prefix
        df['Test_Object_Curvature'] = curvature_val
        
        # Extract test number
        test_match = re.search(r'\[test (\d+)\]', filename)
        test_num = int(test_match.group(1)) if test_match else 0
        df['Test_Number'] = test_num
        
        # For rows where Position_cm exists, set Position_Class
        df.loc[df['Curvature_Active'] == 1, 'Position_Class'] = df.loc[df['Curvature_Active'] == 1, 'Position_cm'].astype(int).astype(str)
        # For idle rows, set Position_Class
        df.loc[df['Curvature_Active'] == 0, 'Position_Class'] = 'rest'
        
        # Compute features for all rows at once
        fft_cols = [col for col in df.columns if col.startswith("FFT_") and not col.startswith("Norm_")]
        norm_fft_cols = [col for col in df.columns if col.startswith("Norm_FFT_")]
        
        # Raw FFT-based features
        df = compute_band_features(df)
        df = compute_fft_peak_index(df)
        df = compute_pca_features(df, fft_cols, prefix="", n_components=2)
        
        # Norm FFT-based features
        df = compute_band_features(df, prefix="Norm_")
        df = compute_fft_peak_index(df, prefix="Norm_")
        df = compute_pca_features(df, norm_fft_cols, prefix="Norm_", n_components=2)
        
        # Select the final columns to include
        output_cols = fft_cols + norm_fft_cols + [
            'Low_Band_Mean', 'Mid_Band_Mean', 'High_Band_Mean',
            'Mid_to_Low_Band_Ratio', 'High_to_Mid_Band_Ratio',
            'FFT_Peak_Index', 'PC1', 'PC2',
            'Norm_Low_Band_Mean', 'Norm_Mid_Band_Mean', 'Norm_High_Band_Mean',
            'Norm_Mid_to_Low_Band_Ratio', 'Norm_High_to_Mid_Band_Ratio',
            'Norm_FFT_Peak_Index', 'Norm_PC1', 'Norm_PC2',
            'Curvature', 'Position_cm', 'Curvature_Active', 'Position_Class',
            'Test_Object', 'Test_Object_Curvature', 'Test_Number'
        ]
        
        # Make sure all columns exist
        for col in output_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Create the final dataframe
        df_final = df[output_cols]
        
        # Create output filename - keep similar to the input but prefix with "processed_"
        output_filename = f"processed_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the processed data
        df_final.to_csv(output_path, index=False)
        print(f"Saved processed file: {output_filename} with {len(df_final)} rows")

print("\n✅ All files processed and saved.")
