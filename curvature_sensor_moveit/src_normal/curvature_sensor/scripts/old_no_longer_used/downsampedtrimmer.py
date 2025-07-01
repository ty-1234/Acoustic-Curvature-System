import pandas as pd
import os

def main():
    """
    Process downsampled sensor data by:
    1. Trimming exactly 2 rows from the top and 2 rows from the bottom of the initial inactive rows
    2. Keeping only active rows (Curvature_Active != 0) after the first active transition
    3. Removing all other inactive rows (Curvature_Active = 0) that appear later in the file
    4. Computing mean FFT amplitudes from the trimmed idle block
    5. Adding normalized FFT columns (e.g., Norm_FFT_200Hz) by dividing each by the idle mean
    6. Setting Curvature = 0 for all rows where Curvature_Active = 0
    
    This fixed-row trimming approach is optimized for downsampled data.
    """
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Input/output directories
    input_dir = os.path.join(base_dir, "..", "csv_data", "merged", "downsampled_files")
    output_dir = os.path.join(base_dir, "..", "csv_data", "trimmed_downsampled")
    os.makedirs(output_dir, exist_ok=True)

    # Constants (fixed-row trimming now)
    TOP_TRIM_ROWS = 2        # rows to drop at the very start of the file
    BOTTOM_TRIM_ROWS = 2     # rows to drop just before the first active row

    # Process each file
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]

    for filepath in input_files:
        try:
            print(f"Processing: {os.path.basename(filepath)}")
            df = pd.read_csv(filepath)

            if "Position_cm" not in df.columns or "Curvature_Active" not in df.columns:
                print(f"⚠️ Skipping {filepath}: missing required columns!")
                continue

            # Find where active rows start
            first_active_index = df[df["Curvature_Active"] != 0].index.min()
            if pd.isna(first_active_index):
                print(f"⚠️ Skipping {filepath}: no active rows found!")
                continue

            # Entire block of idle rows before the first active row
            initial_inactive_rows = df.loc[:first_active_index - 1]

            # Require at least 4 rows so we can remove 2 top + 2 bottom
            if len(initial_inactive_rows) < (TOP_TRIM_ROWS + BOTTOM_TRIM_ROWS):
                print(f"⚠️ Skipping {filepath}: not enough pre-active rows to trim.")
                continue

            # Keep the middle portion
            trimmed_initial_inactive = initial_inactive_rows.iloc[
                TOP_TRIM_ROWS : -BOTTOM_TRIM_ROWS
            ]

            # Active rows (unchanged)
            active_rows = df.loc[first_active_index:][
                df.loc[first_active_index:]["Curvature_Active"] != 0
            ]

            # Combine the trimmed initial inactive rows with only the active rows
            trimmed_df = pd.concat([trimmed_initial_inactive, active_rows], ignore_index=True)

            # === Add normalized FFT columns ===
            fft_cols = [col for col in df.columns if col.startswith("FFT_") and col.endswith("Hz")]
            idle_means = trimmed_initial_inactive[fft_cols].mean()

            for col in fft_cols:
                trimmed_df[f"Norm_{col}"] = trimmed_df[col] / idle_means[col]
                
            # Set Curvature = 0 for all inactive rows (Curvature_Active = 0)
            if "Curvature" in trimmed_df.columns:
                trimmed_df.loc[trimmed_df["Curvature_Active"] == 0, "Curvature"] = 0
                print(f"   - Set Curvature = 0 for all inactive rows")

            # Save the properly trimmed and normalized output
            output_name = os.path.join(output_dir, f"trimmed_{os.path.basename(filepath)}")
            trimmed_df.to_csv(output_name, index=False)

            # Calculate statistics for reporting
            total_inactive_dropped = len(df) - len(trimmed_df) - (TOP_TRIM_ROWS + BOTTOM_TRIM_ROWS)

            print(f"✅ Saved properly trimmed file: {output_name}")
            print(f"   - Trimmed {TOP_TRIM_ROWS} rows from top of initial inactive block")
            print(f"   - Trimmed {BOTTOM_TRIM_ROWS} rows from bottom of initial inactive block")
            print(f"   - Removed {total_inactive_dropped} later inactive rows (Curvature_Active = 0)")
            print(f"   - Added {len(fft_cols)} normalized FFT columns")

        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")

if __name__ == "__main__":
    main()