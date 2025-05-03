import os
import pandas as pd
import glob

def process_curvature_data():
    """
    Preprocess the normalized dataset to create training features.
    
    Looks for files in the normalization directory and performs feature engineering 
    on FFT data, including statistical features, band grouping, and peak detection.
    """
    # === Setup file paths relative to script location ===
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "..", "csv_data", "normalised")
    output_dir = os.path.join(script_dir, "..", "csv_data", "preprocessed")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find normalized CSV files in the directory
    input_files = glob.glob(os.path.join(input_dir, "normalized_*.csv"))
    
    if not input_files:
        print(f"❌ No normalized CSV files found in {input_dir}")
        return None
    
    print(f"📊 Found {len(input_files)} normalized CSV files to process")
    
    # Process each file or combine them
    all_features = []
    
    for input_file in input_files:
        filename = os.path.basename(input_file)
        print(f"Processing {filename}...")
        
        # === Load dataset ===
        try:
            df = pd.read_csv(input_file)
            
            # Verify this is an FFT data file
            fft_cols = [col for col in df.columns if col.startswith("FFT_")]
            if not fft_cols or "Section" not in df.columns:
                print(f"⚠️ Skipping {filename}: Not a valid FFT data file")
                continue
                
            # === Determine if curvature is being applied ===
            df["Curvature_Active"] = df["Section"].notna().astype(int)
            
            # === Clean and convert Position ===
            # Extract numeric value from Section if it contains "cm"
            if df["Section"].dtype == 'object':
                df["Position_cm"] = df["Section"].str.extract(r"(\d+\.?\d*)").astype(float)
            else:
                df["Position_cm"] = df["Section"]
                
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
            high_band = df[fft_cols].iloc[:, 7:]    # 1600–2000 Hz
            
            feature_set["Low_Band_Mean"] = low_band.mean(axis=1)
            feature_set["Mid_Band_Mean"] = mid_band.mean(axis=1)
            feature_set["High_Band_Mean"] = high_band.mean(axis=1)
            
            # Add band ratio features
            feature_set["Mid_to_Low_Band_Ratio"] = feature_set["Mid_Band_Mean"] / feature_set["Low_Band_Mean"].replace(0, 1e-6)
            feature_set["High_to_Mid_Band_Ratio"] = feature_set["High_Band_Mean"] / feature_set["Mid_Band_Mean"].replace(0, 1e-6)
            
            # FFT peak index feature
            feature_set["FFT_Peak_Index"] = df[fft_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(int)
            
            # Add metadata and extract curvature information from filename
            feature_set["Position_cm"] = df["Position_cm"]
            feature_set["Timestamp"] = df["Timestamp"]
            feature_set["Section"] = df["Section"]
            feature_set["Curvature_Active"] = df["Curvature_Active"]
            
            # Extract curvature value from filename (assuming format like "merged_0_05[test 1].csv")
            curvature_value = 0.0
            if "merged_" in filename:
                try:
                    curvature_part = filename.split("merged_")[1].split("[")[0]
                    curvature_value = float(curvature_part.replace("_", "."))
                except Exception as e:
                    print(f"Warning: Could not extract curvature from filename {filename}: {e}")
            
            feature_set["Curvature_Label"] = curvature_value
            
            # Extract just the test name/number from between square brackets
            test_name = "unknown"
            if "[" in filename and "]" in filename:
                test_name = filename.split("[")[1].split("]")[0]
            
            feature_set["RunID"] = test_name
            
            all_features.append(feature_set)
            print(f"✓ Successfully processed {filename}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
    
    # Combine all processed features
    if all_features:
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # === Save preprocessed dataset ===
        output_path = os.path.join(output_dir, "preprocessed_training_dataset.csv")
        combined_features.to_csv(output_path, index=False)
        print(f"✅ Preprocessed dataset saved to: {output_path} with {len(combined_features)} rows from {len(all_features)} files")
        return output_path
    else:
        print("❌ No features were generated")
        return None

# Allow the script to be run directly
if __name__ == "__main__":
    process_curvature_data()