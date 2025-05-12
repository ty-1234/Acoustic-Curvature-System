import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # For demonstration of split
from sklearn.preprocessing import StandardScaler

def load_preprocessed_data(csv_path):
    """Loads the data preprocessed by your feature engineering script."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded preprocessed data from '{csv_path}'.")
        return df
    except FileNotFoundError:
        exit(f"ERROR: Preprocessed file '{csv_path}' not found.")
    except Exception as e:
        exit(f"ERROR loading preprocessed CSV '{csv_path}': {e}")

def get_first_row_fft_baseline(df_train_portion, fft_column_names):
    """
    Calculates the FFT baseline from the very first row of the training data portion.
    ASSUMPTION: The df_train_portion provided starts with a representative idle row.
    In a real CV loop, this would be more nuanced, averaging first rows of multiple original files
    that constitute the training set for that fold.
    """
    if df_train_portion.empty:
        raise ValueError("Training data portion is empty, cannot determine FFT baseline.")
    
    # For this demonstration, we take the first row of the provided training data.
    # In your actual implementation, ensure this row truly represents the idle state.
    first_row_ffts = df_train_portion.iloc[0][fft_column_names]
    
    # Check for zeros or very small values to avoid division issues later
    if (first_row_ffts <= 1e-9).any():
        print("Warning: Some FFT baseline values in the first row are zero or near-zero. This might cause issues with division-based normalization.")
        # Apply a floor to avoid division by zero, or consider subtraction-based normalization
        first_row_ffts = first_row_ffts.copy() # To avoid SettingWithCopyWarning
        first_row_ffts[first_row_ffts <= 1e-9] = 1e-9 # Replace zeros/small numbers with a tiny epsilon

    print(f"Using FFT baseline from first training row: \n{first_row_ffts}")
    return first_row_ffts


def scale_features_for_gpr(df, fft_columns, position_column, target_column, test_size=0.2, random_state=42):
    """
    Demonstrates scaling of input features and target variable for GPR.
    In a real scenario, scaling parameters are fit ONLY on the training set
    within each cross-validation fold.

    Args:
        df (pd.DataFrame): The input DataFrame (output from your previous preprocessing script).
        fft_columns (list): List of FFT column names.
        position_column (str): Name of the Position_cm column.
        target_column (str): Name of the Curvature column.
        test_size (float): Proportion of dataset to include in the test split for demo.
        random_state (int): Random seed for reproducibility of split.

    Returns:
        X_train_scaled, X_test_scaled, y_train_scaled, y_test (original scale), output_scaler
    """
    
    # 1. Define features (X) and target (y)
    X = df[fft_columns + [position_column]].copy()
    y = df[target_column].copy()

    # 2. Split data into training and testing sets (for demonstration)
    # In your real pipeline, this split would be handled by your CV strategy (e.g., LOOOCV)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False) # shuffle=False for time-series if order matters
    
    print(f"\nData split into training ({len(X_train)} rows) and test ({len(X_test)} rows).")

    # --- 3. FFT Feature Normalization (using first row of X_train as baseline) ---
    print("\nNormalizing FFT features...")
    # IMPORTANT: In a real LOOOCV, this baseline should be from the *first idle rows* of the *original files*
    # that constitute the current training fold. This example simplifies by using the first row of X_train.
    fft_baseline_values = get_first_row_fft_baseline(X_train, fft_columns)
    
    X_train_scaled_ffts = X_train[fft_columns].copy()
    X_test_scaled_ffts = X_test[fft_columns].copy()

    for col in fft_columns:
        baseline_val = fft_baseline_values[col]
        X_train_scaled_ffts[col] = X_train[fft_columns][col] / baseline_val
        X_test_scaled_ffts[col] = X_test[fft_columns][col] / baseline_val 
        # Alternative: Subtraction: X_train_scaled_ffts[col] = X_train[fft_columns][col] - baseline_val
    
    print("FFT features normalized against first-row baseline.")

    # --- 4. Position_cm Standardization ---
    print(f"\nStandardizing '{position_column}' feature...")
    pos_scaler = StandardScaler()
    
    # Fit ONLY on X_train
    X_train_scaled_pos = pos_scaler.fit_transform(X_train[[position_column]])
    # Transform X_test using the SAME scaler
    X_test_scaled_pos = pos_scaler.transform(X_test[[position_column]])
    
    print(f"'{position_column}' feature standardized.")

    # Combine scaled FFTs and scaled Position into final X_train_scaled, X_test_scaled
    X_train_scaled = pd.concat([
        X_train_scaled_ffts.reset_index(drop=True), 
        pd.DataFrame(X_train_scaled_pos, columns=[f"{position_column}_scaled"])
    ], axis=1)
    
    X_test_scaled = pd.concat([
        X_test_scaled_ffts.reset_index(drop=True),
        pd.DataFrame(X_test_scaled_pos, columns=[f"{position_column}_scaled"])
    ], axis=1)

    # --- 5. Target Variable (Curvature) Standardization ---
    print(f"\nStandardizing target variable '{target_column}'...")
    output_scaler = StandardScaler() # This will be needed to inverse_transform predictions
    
    # Fit ONLY on y_train
    y_train_scaled = output_scaler.fit_transform(y_train.values.reshape(-1, 1))
    # y_test is kept in its original scale for final evaluation after inverse_transforming predictions
    # If you needed y_test_scaled for some reason:
    # y_test_scaled = output_scaler.transform(y_test.values.reshape(-1, 1))
    
    print(f"Target variable '{target_column}' standardized for training.")
    print("Predictions will need to be inverse_transformed using the fitted output_scaler.")

    return X_train_scaled, X_test_scaled, y_train_scaled.ravel(), y_test, output_scaler


if __name__ == "__main__":
    import os
    import glob

    input_base_directory = "csv_data/merged/"
    output_base_directory = os.path.join(input_base_directory, "gpr_readydata")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_base_directory):
        os.makedirs(output_base_directory)
        print(f"Created output directory: '{output_base_directory}'")

    # --- Configuration ---
    fft_feature_columns = [f'FFT_{hz}Hz' for hz in range(200, 2001, 200)] # Example FFT column names
    position_feature_column = 'Position_cm'
    target_variable_column = 'Curvature'
    
    csv_files_to_process = glob.glob(os.path.join(input_base_directory, "*.csv"))

    if not csv_files_to_process:
        print(f"No CSV files found in '{input_base_directory}'. Exiting.")
        exit()
    
    print(f"Found {len(csv_files_to_process)} CSV files to process in '{input_base_directory}'.")

    for preprocessed_csv_path in csv_files_to_process:
        print(f"\n--- Processing file: {preprocessed_csv_path} ---")
        
        # --- Load Data ---
        master_df = load_preprocessed_data(preprocessed_csv_path)

        if master_df is None or master_df.empty:
            print(f"Skipping empty or unreadable file: {preprocessed_csv_path}")
            continue
            
        # Check if all required columns exist
        required_cols = fft_feature_columns + [position_feature_column, target_variable_column]
        missing_cols = [col for col in required_cols if col not in master_df.columns]
        if missing_cols:
            print(f"ERROR: File '{preprocessed_csv_path}' is missing required columns: {missing_cols}. Skipping.")
            continue

        print("\nOriginal columns:", master_df.columns.tolist())
        print("Shape of loaded data:", master_df.shape)

        # --- Scale and Prepare Data ---
        try:
            X_train_s, X_test_s, y_train_s_array, y_test_orig_series, fitted_output_scaler = scale_features_for_gpr(
                master_df,
                fft_columns=fft_feature_columns,
                position_column=position_feature_column,
                target_column=target_variable_column
                # test_size and random_state will use defaults from the function
            )
        except Exception as e:
            print(f"ERROR during scaling for file '{preprocessed_csv_path}': {e}. Skipping.")
            continue

        print("\n--- Data Scaling and Splitting Complete ---")
        print(f"Shape of X_train_scaled: {X_train_s.shape}")
        print(f"Shape of y_train_scaled: {y_train_s_array.shape}")
        print(f"Shape of X_test_scaled: {X_test_s.shape}")
        print(f"Shape of y_test (original scale): {y_test_orig_series.shape}")

        # --- Prepare DataFrames for Saving ---
        # Training data
        df_train_gpr = X_train_s.copy()
        df_train_gpr[target_variable_column + '_scaled'] = y_train_s_array
        
        # Test data
        df_test_gpr = X_test_s.copy()
        df_test_gpr[target_variable_column] = y_test_orig_series.values # Use .values to align if X_test_s index was reset

        # --- Save GPR-ready data ---
        base_filename = os.path.basename(preprocessed_csv_path)
        name_part, ext_part = os.path.splitext(base_filename)

        output_train_filename = f"{name_part}_train_gpr.csv"
        output_test_filename = f"{name_part}_test_gpr.csv"

        output_train_path = os.path.join(output_base_directory, output_train_filename)
        output_test_path = os.path.join(output_base_directory, output_test_filename)

        try:
            df_train_gpr.to_csv(output_train_path, index=False)
            print(f"Successfully saved scaled training data to: '{output_train_path}'")
            
            df_test_gpr.to_csv(output_test_path, index=False)
            print(f"Successfully saved scaled test data to: '{output_test_path}'")
        except Exception as e:
            print(f"ERROR saving GPR-ready data for '{base_filename}': {e}")
            
        print(f"--- Finished processing file: {preprocessed_csv_path} ---")

    print("\nAll specified CSV files processed.")
    print("GPR-ready training and test sets saved in:", output_base_directory)
    print("Remember that 'fitted_output_scaler' (not saved to CSV) is needed to inverse_transform GPR predictions on the target variable.")