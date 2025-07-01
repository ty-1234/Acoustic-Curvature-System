import pandas as pd
import os

def main():
    """
    Normalize FFT columns using trimmed baseline noise from Curvature_Active == 0.
    Trims first and last 2 seconds (~44 rows each) before Position_cm == 0.0 appears.
    """
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Input/output directories
    input_dir = os.path.join(base_dir, "..", "csv_data", "merged")
    output_dir = os.path.join(base_dir, "..", "csv_data", "normalised")
    os.makedirs(output_dir, exist_ok=True)

    # Constants
    rows_per_second = 1000 / 22050  # Each row ~0.04535 sec
    trim_duration_sec = 2.0
    rows_to_trim = int(trim_duration_sec / rows_per_second)  # ~44 rows

    # Process each file
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]

    for filepath in input_files:
        try:
            df = pd.read_csv(filepath)

            if "Position_cm" not in df.columns or "Curvature_Active" not in df.columns:
                print(f"⚠️ Skipping {filepath}: missing required columns!")
                continue

            fft_cols = [col for col in df.columns if col.startswith("FFT_")]

            # Identify first clamp point (Position_cm == 0.0)
            first_clamp_index = df[df["Position_cm"] == 0.0].index.min()
            if pd.isna(first_clamp_index):
                print(f"⚠️ Skipping {filepath}: no Position_cm == 0.0 found!")
                continue

            # Get all rows before clamp where Curvature_Active == 0
            baseline_mask = (df.index < first_clamp_index) & (df["Curvature_Active"] == 0)
            baseline_rows = df.loc[baseline_mask]

            if len(baseline_rows) < 2 * rows_to_trim:
                print(f"⚠️ Skipping {filepath}: not enough baseline rows to trim.")
                continue

            # Keep only middle (cleanest) portion of baseline noise
            trimmed_baseline = baseline_rows.iloc[rows_to_trim:-rows_to_trim]

            # Compute mean FFT values from trimmed baseline
            baseline_means = trimmed_baseline[fft_cols].mean()

            # Normalize original FFT columns using baseline means
            normalized = df[fft_cols] / baseline_means

            # Overwrite and rename FFT columns with _Norm
            for col in fft_cols:
                df[col] = normalized[col]
                df = df.rename(columns={col: f"{col}_Norm"})

            # Save output
            output_name = os.path.join(output_dir, f"normalized_{os.path.basename(filepath)}")
            df.to_csv(output_name, index=False)
            print(f"✅ Saved normalized file: {output_name}")

        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")

if __name__ == "__main__":
    main()