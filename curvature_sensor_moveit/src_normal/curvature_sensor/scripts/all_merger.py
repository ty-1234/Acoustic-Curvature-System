import os
import pandas as pd
import re
from glob import glob

# === Configuration ===
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_folder = os.path.join(base_dir, "csv_data/merged")
output_file = os.path.join(base_dir, "csv_data/combined_dataset.csv")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# === Helper Function: Extract curvature and run_id ===
def parse_filename(filename):
    base = os.path.basename(filename)
    # Updated regex to match the actual format with underscores
    match = re.match(r"merged_([0-9_eE\-]+)\[([^\]]+)\]\.csv", base)
    if match:
        # Convert the underscore-separated value back to a float
        curvature_str = match[1].replace("_", ".")
        curvature = float(curvature_str)
        run_id = match[2].strip()
        return curvature, run_id
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern.")

# === Gather and process files ===
all_files = glob(os.path.join(input_folder, "merged_*.csv"))
frames = []

for filepath in all_files:
    try:
        curvature, run_id = parse_filename(filepath)
        df = pd.read_csv(filepath)
        df['Curvature_Label'] = curvature
        df['RunID'] = run_id

        # Optional: Drop columns that are all NaN (e.g., PosX, Section)
        df = df.dropna(axis=1, how='all')

        frames.append(df)
    except Exception as e:
        print(f"Skipping {filepath}: {e}")

# === Merge and Save ===
if frames:
    merged_df = pd.concat(frames, ignore_index=True)
    merged_df.to_csv(output_file, index=False)
    print(f"✅ Combined dataset saved to: {output_file} ({len(merged_df)} rows)")
else:
    print("❌ No valid files found to merge.")
