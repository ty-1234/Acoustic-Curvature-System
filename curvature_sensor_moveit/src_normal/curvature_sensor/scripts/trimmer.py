import pandas as pd
import os

def main():
    """
    Process sensor data by:
    1. Trimming the first and last 2 seconds (~44 rows each) from the initial inactive rows block
    2. Keeping only active rows (Curvature_Active != 0) after the first active transition
    3. Removing all other inactive rows (Curvature_Active = 0) that appear later in the file
    4. Computing mean FFT amplitudes from the trimmed idle block
    5. Adding normalized FFT columns (e.g., Norm_FFT_200Hz) by dividing each by the idle mean

    This creates a cleaned dataset with raw and normalized FFT features.
    """
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Input/output directories
    input_dir = os.path.join(base_dir, "..", "csv_data", "merged")
    output_dir = os.path.join(base_dir, "..", "csv_data", "trimmed_data")
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

            # Find the first row where Curvature_Active is not 0 (transition from inactive to active)
            first_active_index = df[df["Curvature_Active"] != 0].index.min()

            if pd.isna(first_active_index):
                print(f"⚠️ Skipping {filepath}: no active rows found (all Curvature_Active == 0)!")
                continue

            # Get all rows before the first active row (the first block of inactive rows)
            initial_inactive_rows = df.loc[:first_active_index - 1]

            if len(initial_inactive_rows) < 2 * rows_to_trim:
                print(f"⚠️ Skipping {filepath}: not enough initial inactive rows to trim (need {2 * rows_to_trim}, got {len(initial_inactive_rows)}).")
                continue

            # Keep only middle (cleanest) portion of initial inactive rows
            trimmed_initial_inactive = initial_inactive_rows.iloc[rows_to_trim:-rows_to_trim]

            # Get only the active rows after the first active transition
            active_rows = df.loc[first_active_index:][df.loc[first_active_index:]["Curvature_Active"] != 0]

            # Combine the trimmed initial inactive rows with only the active rows
            trimmed_df = pd.concat([trimmed_initial_inactive, active_rows], ignore_index=True)

            # === Add normalized FFT columns ===
            fft_cols = [col for col in df.columns if col.startswith("FFT_") and col.endswith("Hz")]
            idle_means = trimmed_initial_inactive[fft_cols].mean()

            for col in fft_cols:
                trimmed_df[f"Norm_{col}"] = trimmed_df[col] / idle_means[col]

            # Save the properly trimmed and normalized output
            output_name = os.path.join(output_dir, f"trimmed_{os.path.basename(filepath)}")
            trimmed_df.to_csv(output_name, index=False)

            # Calculate statistics for reporting
            total_inactive_dropped = len(df) - len(trimmed_df) - (len(initial_inactive_rows) - len(trimmed_initial_inactive))

            print(f"✅ Saved properly trimmed file: {output_name}")
            print(f"   - Trimmed {len(initial_inactive_rows) - len(trimmed_initial_inactive)} rows from initial inactive block")
            print(f"   - Removed {total_inactive_dropped} later inactive rows (Curvature_Active = 0)")
            print(f"   - Added {len(fft_cols)} normalized FFT columns")

        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")

if __name__ == "__main__":
    main()
