import pandas as pd
import os

def main():
    """
    Entry point for normalizing FFT columns in merged CSV files.
    """
    # Base directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Directory containing the input CSV files (correct path)
    input_dir = os.path.join(base_dir, "..", "csv_data", "merged")

    # Directory to save the normalized CSV files (correct path with British spelling)
    output_dir = os.path.join(base_dir, "..", "csv_data", "normalised")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Collect all CSV files from the directory
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]

    # Normalize all FFT columns using baseline rows before Position_cm == 0.0
    for filepath in input_files:
        try:
            df = pd.read_csv(filepath)
            
            # Check if Position_cm exists
            if "Position_cm" not in df.columns:
                print(f"⚠️ Skipping {filepath}: missing Position_cm column!")
                continue
                
            # Identify FFT columns
            fft_cols = [col for col in df.columns if col.startswith("FFT_")]
            
            # Get baseline: all rows before first appearance of Position_cm == 0.0
            first_clamp_index = df[df["Position_cm"] == 0.0].index.min()
            
            # Safety check for files with no Position_cm == 0.0 rows
            if pd.isna(first_clamp_index):
                print(f"⚠️ Skipping {filepath}: no Position_cm == 0.0 found!")
                continue
                
            baseline_df = df.loc[:first_clamp_index - 1, fft_cols]
            
            # Compute mean FFT for baseline
            baseline_means = baseline_df.mean()
            
            # Normalize FFT columns and replace the original values
            normalized = df[fft_cols] / baseline_means
            
            # Update the FFT columns with normalized values and rename them with _Norm suffix
            for col in fft_cols:
                df[col] = normalized[col]
                # Rename the column to include _Norm suffix
                df = df.rename(columns={col: f"{col}_Norm"})
            
            # Save output to the normalisation directory
            output_name = os.path.join(output_dir, f"normalized_{os.path.basename(filepath)}")
            df.to_csv(output_name, index=False)
            print(f"✅ Saved normalized file: {output_name}")
            
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")

if __name__ == "__main__":
    main()
