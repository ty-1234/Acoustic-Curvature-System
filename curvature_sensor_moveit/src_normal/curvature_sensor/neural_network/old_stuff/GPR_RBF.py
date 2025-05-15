"""
gpr_pipeline_multi_file.py

A single, well-commented script to perform an end-to-end machine learning pipeline
for predicting sensor radius using Gaussian Process Regression (GPR).
This script is capable of processing multiple input CSV files from specified sessions,
combining the data, and then training and evaluating the GPR model.

Pipeline Steps:
1.  Configuration: Define paths, column names, model parameters, etc.
2.  File Discovery: Locate and filter relevant CSV files based on session numbers.
3.  Per-File Preprocessing Loop:
    a.  Load data from a single CSV file.
    b.  Apply FFT Baseline Normalization: Calculate a baseline from "idle" periods
        (Curvature_Active == 0, before Position_cm >= 0) specific to that file and
        subtract it from the FFT features of "active" rows (Curvature_Active == 1).
    c.  Convert Curvature to Radius: Transform the 'Curvature' column (e.g., in mm^-1)
        to 'Radius' (e.g., in mm), with specific handling for zero/near-zero curvature.
    d.  Collect all processed "active" DataFrames.
4.  Combine Data: Concatenate DataFrames from all processed files.
5.  Prepare Features (X) and Target (y): Select FFT columns for X and Radius column for y.
6.  Data Cleaning: Handle any remaining NaN values in the combined dataset.
7.  Train-Test Split: Divide the combined data into training and testing sets.
8.  Scaling:
    a.  Fit MinMaxScaler on the training data (X_train and y_train separately).
    b.  Apply the fitted scalers to transform X_train, X_test, and y_train to a 0-1 range.
9.  GPR Model Training: Train a GaussianProcessRegressor on the scaled training data.
10. Prediction: Make predictions on the scaled test data (X_test_scaled).
11. Descaling: Convert the scaled predictions (and true test target values if needed)
    back to their original physical units (e.g., radius in mm).
12. Evaluation: Calculate and print performance metrics (MAE, RMSE, R2 score)
    using the descaled, real-world values.
13. Optional: Convert final radius predictions back to curvature units.
"""

# Standard library imports
import pandas as pd
import numpy as np
import os
import glob # For finding files matching a pattern
import json # Added for JSON output
import matplotlib.pyplot as plt # Added for plotting

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Configuration Variables ---
# These variables control the script's behavior and can be adjusted as needed.

# --- File and Directory Settings ---
# Parent directory where the 'CLEANED_DATA_SUBFOLDER' is located.
# "." means the current directory where the script is run.
INPUT_DATA_PARENT_DIR = "csv_data"
# Subfolder within 'INPUT_DATA_PARENT_DIR' containing the cleaned CSV files.
CLEANED_DATA_SUBFOLDER = "cleaned"
# Directory for saving model outputs, relative to the script's location.
MODEL_OUTPUT_DIR = "model_outputs"
# List of session numbers (as strings) to process.
# The script will look for filenames containing "[test SESSION_NUM]" (case-insensitive).
# Example: ["1", "2", "4"] will process files related to test sessions 1, 2, and 4.
# If empty [], the script will attempt to process all "cleaned_*.csv" files in the subfolder.
SESSIONS_TO_PROCESS = ["1"] # Example: process only files from session 1

# --- Column Names (Must match your CSV files) ---
# List of columns representing FFT (Fast Fourier Transform) features.
FFT_COLUMNS = [f'FFT_{i}Hz' for i in range(200, 2001, 200)]
# Column indicating the sensor's position (e.g., in centimeters).
POSITION_COLUMN = 'Position_cm'
# Column indicating whether the sensor is in an active state (e.g., bending).
# Typically 0 for idle/inactive, 1 for active.
ACTIVITY_COLUMN = 'Curvature_Active'
# Column containing the original curvature measurement from the sensor.
# Ensure you know its units (e.g., mm^-1 or m^-1).
CURVATURE_COLUMN = 'Curvature'
# Name for the new column that will store the calculated radius.
# Units will be inverse of CURVATURE_COLUMN units (e.g., mm if curvature is mm^-1).
TARGET_RADIUS_COLUMN = 'Radius_mm'

# --- Radius Conversion Settings ---
# Method to handle curvature values of 0 or very close to 0 when converting to radius.
# Options:
#   'cap': Replace radius for zero/near-zero curvature with `RADIUS_CAP_VALUE`. This is often preferred
#          to avoid infinite values which can cause issues with scaling and modeling.
#   'placeholder': Replace with `RADIUS_PLACEHOLDER_VALUE` (e.g., np.nan or a specific number).
#   'infinity': Allow np.inf (1/0). May require careful handling in subsequent steps.
#   'warn_skip': Print a warning and set radius to np.nan for zero/near-zero curvature.
RADIUS_ZERO_HANDLING = 'cap'
# Value to use if `RADIUS_ZERO_HANDLING` is 'cap'. Represents a very large radius (effectively straight).
# Adjust based on expected radius range and curvature units. E.g., for mm^-1 curvature, 1e6 mm = 1 km.
RADIUS_CAP_VALUE = 1e6
# Value to use if `RADIUS_ZERO_HANDLING` is 'placeholder'.
RADIUS_PLACEHOLDER_VALUE = np.nan

# --- Gaussian Process Regression (GPR) Settings ---
# Kernel for the GPR model. This is a common choice combining an RBF kernel (for smoothness)
# with a ConstantKernel (for scaling) and a WhiteKernel (for noise).
# Length scale bounds and noise level bounds can be tuned.
GPR_KERNEL = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
             WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
# Value added to the diagonal of the kernel matrix during fitting.
# It can be interpreted as the variance of additional Gaussian noise on the observations
# or used for numerical stability.
GPR_ALPHA = 1e-10
# Number of times the optimizer is restarted when fitting the GPR kernel.
# More restarts can help find a better set of kernel hyperparameters but increases training time.
GPR_N_RESTARTS_OPTIMIZER = 10

# --- Train-Test Split Settings ---
# Proportion of the dataset to include in the test split (e.g., 0.2 for 20% test data).
TEST_SIZE = 0.2
# Seed for the random number generator to ensure reproducible train-test splits.
RANDOM_STATE = 42

# --- Script Control ---
# If True, print detailed messages about the script's progress. Set to False for less output.
VERBOSE = True

# --- 1. Data Loading Function (for a single file) ---
def load_single_file_data(file_path):
    """
    Loads data from a single CSV file into a pandas DataFrame.

    Args:
        file_path (str): The full path to the CSV file.

    Returns:
        pd.DataFrame or None: The loaded DataFrame, or None if an error occurs.
    """
    if VERBOSE:
        print(f"  Loading data from: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)
        if VERBOSE:
            print(f"    Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"    Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"    Error loading data from {file_path}: {e}")
        return None

# --- 2. FFT Baseline Normalization Function ---
def apply_fft_baseline_normalization(df_input, fft_cols, pos_col, act_col, filename=""):
    """
    Applies FFT baseline normalization to a DataFrame.
    The baseline is calculated from "idle" periods (activity == 0, before position changes significantly)
    specific to this DataFrame and then subtracted from "active" periods (activity == 1).

    Args:
        df_input (pd.DataFrame): The input DataFrame for a single file/session.
        fft_cols (list): List of FFT column names.
        pos_col (str): Name of the position column.
        act_col (str): Name of the activity column.
        filename (str, optional): Name of the file being processed (for logging). Defaults to "".

    Returns:
        tuple: (pd.DataFrame, pd.Series or None)
            - active_df: DataFrame containing only active rows with normalized FFTs.
            - idle_fft_baseline: Series containing the calculated baseline for each FFT column, or None.
    """
    df = df_input.copy() # Work on a copy to avoid modifying the original DataFrame
    if VERBOSE:
        print(f"  Applying FFT baseline normalization for {filename}...")

    # Convert position column to numeric, coercing errors to NaN for robustness
    numeric_positions = pd.to_numeric(df[pos_col], errors='coerce')

    # Find the index of the first row where position is >= 0 (start of potential movement)
    first_zero_or_positive_pos_idx = len(df) # Default if no such row is found
    if (numeric_positions >= 0).any(): # Check if any non-negative position exists
        first_zero_or_positive_pos_idx = (numeric_positions >= 0).idxmax() # Get index of first True

    # Find the index of the first row where activity is not 0 (i.e., activity becomes 1)
    first_active_event_idx = len(df) # Default if all activity is 0
    if (df[act_col] != 0).any(): # Check if any non-zero activity exists
       first_active_event_idx = (df[act_col] != 0).idxmax() # Get index of first True
    
    # The period for baseline calculation ends before the *earlier* of the two events above
    end_of_baseline_period_idx = min(first_zero_or_positive_pos_idx, first_active_event_idx)
    
    # Select rows for baseline: activity is 0 AND index is before the end_of_baseline_period_idx
    baseline_df = df.loc[(df.index < end_of_baseline_period_idx) & (df[act_col] == 0)]

    idle_fft_baseline = None # Initialize baseline as None
    if not baseline_df.empty:
        # Calculate the mean of each FFT column during these identified idle periods
        idle_fft_baseline = baseline_df[fft_cols].mean()
        if VERBOSE:
            print(f"    Calculated baseline from {len(baseline_df)} idle rows for {filename}.")
    elif VERBOSE:
        print(f"    Warning: No valid idle frames found for FFT baseline in {filename}.")

    # Get active parts of the DataFrame (where activity column is 1)
    active_df = df[df[act_col] == 1].copy() # Use .copy() to avoid SettingWithCopyWarning
    if VERBOSE:
        print(f"    Number of active rows in {filename} (where '{act_col}' == 1): {active_df.shape[0]}")

    # Subtract the calculated baseline from the active parts if baseline exists and there are active rows
    if idle_fft_baseline is not None and not active_df.empty:
        for col in fft_cols: # Iterate through each FFT column
            if col in idle_fft_baseline: # Check if baseline was calculated for this column
                active_df[col] = active_df[col] - idle_fft_baseline[col]
        if VERBOSE:
            print(f"    FFT baseline subtracted from active rows in {filename}.")
    elif idle_fft_baseline is None and not active_df.empty and VERBOSE:
        # If no baseline was calculated but there are active rows, these rows won't be normalized
        print(f"    FFT features for active rows in {filename} not normalized (no valid baseline).")
            
    return active_df, idle_fft_baseline

# --- 3. Curvature to Radius Conversion Function ---
def convert_curvature_to_radius(curvature_series, zero_handling, cap_value, placeholder_val):
    """
    Converts a pandas Series of curvature values to radius (1/curvature).
    Includes specified handling for zero or near-zero curvature values.

    Args:
        curvature_series (pd.Series): Series containing curvature values.
        zero_handling (str): Method to handle zero/near-zero curvature ('cap', 'placeholder', 'infinity', 'warn_skip').
        cap_value (float): Value to use if `zero_handling` is 'cap'.
        placeholder_val (float or np.nan): Value to use if `zero_handling` is 'placeholder'.

    Returns:
        pd.Series: Series containing radius values.
    """
    if VERBOSE:
        print(f"  Converting curvature to radius (method: {zero_handling})...")
    
    # Initialize an empty Series with the same index and float dtype for radius values
    radius_series = pd.Series(index=curvature_series.index, dtype=float)
    
    # Identify zero or very small (epsilon) curvature values to avoid division by zero
    # np.isclose is used for robust comparison with floating point numbers
    near_zero_mask = np.isclose(curvature_series, 0)

    # Calculate radius for non-zero/non-near-zero curvatures
    if (~near_zero_mask).any(): # Check if there are any non-zero curvatures
        non_zero_curvatures = curvature_series[~near_zero_mask]
        radius_series.loc[~near_zero_mask] = 1.0 / non_zero_curvatures
    
    # Handle zero/near-zero curvatures based on the chosen method
    if near_zero_mask.any(): # Check if there are any zero/near-zero curvatures
        if zero_handling == 'cap':
            radius_series.loc[near_zero_mask] = cap_value
        elif zero_handling == 'placeholder':
            radius_series.loc[near_zero_mask] = placeholder_val
        elif zero_handling == 'infinity':
            # Suppress division by zero warnings locally for this operation if infinity is desired
            with np.errstate(divide='ignore'):
                # For radius, typically positive infinity is desired for a straight sensor (zero curvature)
                radius_series.loc[near_zero_mask] = np.inf
        elif zero_handling == 'warn_skip':
            if VERBOSE:
                print("    Warning: Zero/near-zero curvature found. Setting radius to NaN.")
            radius_series.loc[near_zero_mask] = np.nan
        else:
            # Raise an error if an invalid zero_handling method is specified
            raise ValueError(f"Invalid 'zero_handling' method: {zero_handling}. "
                             "Choose from 'cap', 'placeholder', 'infinity', 'warn_skip'.")
    return radius_series

# --- 4. Feature Scaling Functions ---
def scale_data(data_to_scale, columns, existing_scalers=None):
    """
    Scales specified columns of data (DataFrame or Series) to the 0-1 range using MinMaxScaler.
    If `existing_scalers` is None, fits new scalers to `data_to_scale` and returns them.
    Otherwise, uses the `existing_scalers` to transform `data_to_scale`.

    Args:
        data_to_scale (pd.DataFrame or pd.Series): Data to be scaled.
        columns (list): List of column names to scale. If `data_to_scale` is a Series,
                        this should be a list with one element (the Series' name or target name).
        existing_scalers (dict, optional): Dictionary of pre-fitted scalers.
                                           Keys are column names, values are MinMaxScaler objects.
                                           Defaults to None (fit new scalers).

    Returns:
        tuple: (pd.DataFrame, dict)
            - scaled_data: DataFrame with specified columns scaled.
            - fitted_scalers: Dictionary of fitted/used scalers.
    """
    if VERBOSE and existing_scalers is None:
        print(f"  Fitting scalers and scaling data for columns: {columns}")
    elif VERBOSE and existing_scalers is not None:
        print(f"  Transforming data using existing scalers for columns: {columns}")

    scaled_data = data_to_scale.copy() # Work on a copy
    
    # If input is a Series (e.g., the target variable y), convert to DataFrame for consistent processing
    if isinstance(data_to_scale, pd.Series): 
        # Use Series name if available, otherwise use the first name from columns list
        col_name_for_series = data_to_scale.name if data_to_scale.name else columns[0]
        scaled_data = pd.DataFrame(scaled_data, columns=[col_name_for_series])

    # Initialize or use existing scalers dictionary
    fitted_scalers = {} if existing_scalers is None else existing_scalers

    for col in columns:
        if col not in scaled_data.columns:
            if VERBOSE: print(f"    Warning: Column '{col}' not in data for scaling. Skipping.")
            continue
        
        # MinMaxScaler cannot handle NaNs. Ensure data is clean before scaling.
        if scaled_data[col].isnull().any():
             raise ValueError(f"Column '{col}' contains NaN values. Handle NaNs (e.g., impute or drop) before scaling.")

        # Reshape data for scaler: expects a 2D array (n_samples, n_features=1)
        column_data = scaled_data[col].values.reshape(-1, 1)
        
        if existing_scalers is None: # Fit new scalers (typically for training data)
            scaler = MinMaxScaler() # Initialize a new scaler
            scaled_column_data = scaler.fit_transform(column_data) # Fit and then transform
            fitted_scalers[col] = scaler # Store the fitted scaler
        else: # Use existing scalers (typically for test data or new unseen data)
            if col in existing_scalers:
                scaler = existing_scalers[col]
                scaled_column_data = scaler.transform(column_data) # Only transform
            else:
                # This case should ideally not happen if columns are consistent
                if VERBOSE: print(f"    Warning: No existing scaler for column '{col}'. Skipping transform.")
                continue 
        
        # Update the DataFrame with the scaled column data (flattened back to 1D)
        scaled_data[col] = scaled_column_data.flatten()
        
    return scaled_data, fitted_scalers

def descale_data(data_to_descale, columns, scalers_dict):
    """
    Descales specified columns of data from the 0-1 range back to their original scale,
    using previously fitted MinMaxScaler objects.

    Args:
        data_to_descale (pd.DataFrame or pd.Series): Data with scaled columns (0-1 range).
        columns (list): List of column names to descale.
        scalers_dict (dict): Dictionary of fitted scalers (keys are column names).

    Returns:
        pd.DataFrame: DataFrame with specified columns descaled.
    """
    if VERBOSE:
        print(f"  Descaling data for columns: {columns}")
    
    descaled_data = data_to_descale.copy() # Work on a copy
    # If input is a Series, convert to DataFrame for consistent processing
    if isinstance(data_to_descale, pd.Series):
         col_name_for_series = data_to_descale.name if data_to_descale.name else columns[0]
         descaled_data = pd.DataFrame(descaled_data, columns=[col_name_for_series])

    for col in columns:
        if col not in descaled_data.columns:
            if VERBOSE: print(f"    Warning: Column '{col}' not in data for descaling. Skipping.")
            continue
        if col not in scalers_dict:
            if VERBOSE: print(f"    Warning: No scaler found for column '{col}' in scalers_dict. Skipping descaling.")
            continue

        scaler = scalers_dict[col]
        # Reshape data for inverse_transform: expects 2D array
        column_data = descaled_data[col].values.reshape(-1, 1)
        
        # Apply inverse transform to get data back to original scale
        original_scale_data = scaler.inverse_transform(column_data)
        
        # Update DataFrame with descaled data
        descaled_data[col] = original_scale_data.flatten()
        
    return descaled_data

# --- 5. Plotting Function ---
def generate_true_vs_predicted_plot(y_true, y_pred, output_dir, filename_prefix="gpr_80_20", target_unit="units"):
    """
    Generates and saves a scatter plot of true vs. predicted values.

    Args:
        y_true (pd.Series or np.array): True target values.
        y_pred (pd.Series or np.array): Predicted target values.
        output_dir (str): Directory to save the plot.
        filename_prefix (str): Prefix for the saved plot filename.
        target_unit (str): Unit of the target variable for plot labels.
    """
    if VERBOSE:
        print(f"\n  Generating True vs. Predicted plot...")
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Predictions")
    
    # Determine plot limits to include all data and the y=x line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal (y=x)")
    
    plt.xlabel(f"True Radius ({target_unit})")
    plt.ylabel(f"Predicted Radius ({target_unit})")
    plt.title(f"True vs. Predicted Radius ({filename_prefix})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_filename = f"{filename_prefix}_true_vs_predicted.png"
    plot_output_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_output_path)
        if VERBOSE:
            print(f"    Plot saved to: {plot_output_path}")
    except Exception as e:
        print(f"    Error saving plot: {e}")
    plt.close() # Close the plot figure to free up memory

# --- Main Execution Block ---
if __name__ == "__main__":
    # Construct the full path to the directory containing cleaned data files
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Get directory of the current script
    input_data_dir_path = os.path.join(script_dir, INPUT_DATA_PARENT_DIR, CLEANED_DATA_SUBFOLDER)

    if not os.path.isdir(input_data_dir_path):
        print(f"Error: Input data directory '{input_data_dir_path}' not found. "
              "Please check INPUT_DATA_PARENT_DIR and CLEANED_DATA_SUBFOLDER settings.")
        exit() # Exit if data directory doesn't exist

    # Construct the full path for the model output directory and create it if it doesn't exist
    model_output_full_path = os.path.join(script_dir, MODEL_OUTPUT_DIR)
    if not os.path.isdir(model_output_full_path):
        os.makedirs(model_output_full_path, exist_ok=True)
        if VERBOSE:
            print(f"Created output directory: {model_output_full_path}")

    # Get all 'cleaned_*.csv' files from the directory
    all_files_in_dir = glob.glob(os.path.join(input_data_dir_path, "cleaned_*.csv"))
    
    files_to_process = [] # List to store paths of files that will be processed
    if not SESSIONS_TO_PROCESS: # If SESSIONS_TO_PROCESS list is empty, process all found files
        if VERBOSE:
            print("No specific sessions defined in SESSIONS_TO_PROCESS. "
                  "Attempting to process all 'cleaned_*.csv' files found.")
        files_to_process = all_files_in_dir
    else: # Filter files based on specified session numbers
        for session_num in SESSIONS_TO_PROCESS:
            session_tag = f"[test {session_num}]".lower() # Create session tag, e.g., "[test 1]"
            found_for_session = False
            for f_path in all_files_in_dir:
                if session_tag in os.path.basename(f_path).lower(): # Case-insensitive check
                    files_to_process.append(f_path)
                    found_for_session = True
            if not found_for_session and VERBOSE:
                print(f"  Warning: No files found for session tag '{session_tag}' in '{input_data_dir_path}'.")
    
    if not files_to_process:
        print(f"No files found to process in '{input_data_dir_path}' matching specified criteria. Exiting.")
        exit()

    # Ensure unique list of files and sort them (optional, but good for reproducibility)
    files_to_process = sorted(list(set(files_to_process))) 
    if VERBOSE:
        print(f"Found {len(files_to_process)} files to process for GPR modeling:")
        for f_path_idx, f_path_val in enumerate(files_to_process):
            print(f"  {f_path_idx+1}. {os.path.basename(f_path_val)}")

    all_processed_active_dfs = [] # List to store processed DataFrames from each file

    # --- Stage 1: Loop through files for initial preprocessing ---
    if VERBOSE:
        print("\n--- Stage 1: Per-File Preprocessing ---")
    for file_path in files_to_process:
        # 1a. Load data from the current file
        df_raw_single = load_single_file_data(file_path)
        if df_raw_single is None: # Skip if file loading failed
            print(f"  Skipping file {os.path.basename(file_path)} due to loading error.")
            continue

        # 1b. Apply FFT Baseline Normalization (file-specific)
        df_active_single, _ = apply_fft_baseline_normalization(
            df_raw_single, FFT_COLUMNS, POSITION_COLUMN, ACTIVITY_COLUMN, filename=os.path.basename(file_path)
        )

        if df_active_single.empty: # Skip if no active rows after normalization
            if VERBOSE:
                print(f"  No active data rows in {os.path.basename(file_path)} after FFT normalization. Skipping.")
            continue
        
        # Check if the original curvature column exists before trying to convert it
        if CURVATURE_COLUMN not in df_active_single.columns:
            print(f"  Error: Curvature column '{CURVATURE_COLUMN}' not found in {os.path.basename(file_path)}. Skipping.")
            continue
            
        # 1c. Convert Curvature to Radius
        df_active_single[TARGET_RADIUS_COLUMN] = convert_curvature_to_radius(
            df_active_single[CURVATURE_COLUMN],
            RADIUS_ZERO_HANDLING,
            RADIUS_CAP_VALUE,
            RADIUS_PLACEHOLDER_VALUE
        )
        # 1d. Collect the processed DataFrame
        all_processed_active_dfs.append(df_active_single)

    if not all_processed_active_dfs: # Exit if no data was successfully processed from any file
        print("No data available after processing all specified files. Exiting.")
        exit()

    # --- Stage 2: Combine data from all files ---
    if VERBOSE:
        print("\n--- Stage 2: Combining Data from All Processed Files ---")
    combined_df = pd.concat(all_processed_active_dfs, ignore_index=True) # ignore_index resets DataFrame index
    if VERBOSE:
        print(f"Combined data from all files. Shape: {combined_df.shape}")

    # --- Stage 3: Prepare Features (X) and Target (y) from combined data ---
    X_combined = combined_df[FFT_COLUMNS]
    y_combined = combined_df[TARGET_RADIUS_COLUMN]

    # --- Stage 4: Data Cleaning on combined data ---
    # Drop any rows that might have NaN values after all processing (e.g., if radius conversion resulted in NaN)
    original_rows_combined = len(combined_df)
    # Create a temporary DataFrame for easy NaN dropping across X and y relevant columns
    df_for_nan_check = pd.concat([X_combined, y_combined], axis=1)
    df_final_processed = df_for_nan_check.dropna() # Drop rows with any NaN in these columns
    
    if VERBOSE and len(df_final_processed) < original_rows_combined:
        print(f"  Dropped {original_rows_combined - len(df_final_processed)} rows from combined data due to NaN values.")

    if df_final_processed.empty: # Exit if no data remains after cleaning
        print("No data remaining after NaN drop from combined data. Exiting.")
        exit()

    # Re-assign X and y from the cleaned, combined DataFrame
    X = df_final_processed[FFT_COLUMNS]
    y = df_final_processed[TARGET_RADIUS_COLUMN]

    # --- Stage 5: Train-Test Split on combined data ---
    if VERBOSE:
        print("\n--- Stage 5: Splitting Combined Data into Training and Test Sets ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    if VERBOSE:
        print(f"Combined data split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")

    # --- Stage 6: Scale Features and Target (Fit on TRAIN, transform train & test) ---
    if VERBOSE:
        print("\n--- Stage 6: Scaling Features and Target (0-1 Range) ---")
    # Scale X (FFT features)
    # Fit scalers on X_train and transform X_train
    X_train_scaled, x_scalers = scale_data(X_train, FFT_COLUMNS)
    # Transform X_test using the scalers fitted on X_train
    X_test_scaled, _ = scale_data(X_test, FFT_COLUMNS, existing_scalers=x_scalers)

    # Scale y (Radius target variable)
    # Convert y_train Series to DataFrame for the scale_data function
    y_train_df = pd.DataFrame(y_train, columns=[TARGET_RADIUS_COLUMN])
    # Fit scalers on y_train_df and transform y_train_df
    y_train_scaled_df, y_scalers = scale_data(y_train_df, [TARGET_RADIUS_COLUMN])
    # Convert scaled y_train back to Series for GPR fitting
    y_train_scaled = y_train_scaled_df[TARGET_RADIUS_COLUMN]

    # Note: y_test is not scaled at this stage. It's kept in its original (descaled) form
    # for direct comparison with descaled predictions later.

    # --- Stage 7: Gaussian Process Regression (GPR) Model Training ---
    if VERBOSE:
        print("\n--- Stage 7: Training Gaussian Process Regression (GPR) Model ---")
        print(f"  Using Kernel: {GPR_KERNEL}")
        
    gpr = GaussianProcessRegressor(
        kernel=GPR_KERNEL,
        alpha=GPR_ALPHA, # Regularization/noise term
        n_restarts_optimizer=GPR_N_RESTARTS_OPTIMIZER, # For robust hyperparameter optimization
        random_state=RANDOM_STATE # For reproducible results
    )
    # Fit the GPR model on the scaled training data
    gpr.fit(X_train_scaled, y_train_scaled)

    if VERBOSE:
        print("GPR model training complete.")
        print(f"  Optimized Kernel Parameters: {gpr.kernel_}")
        print(f"  Log-Marginal-Likelihood of trained model: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}")

    # --- Stage 8: Make Predictions on the Test Set ---
    if VERBOSE:
        print("\n--- Stage 8: Making Predictions on the Test Set ---")
    # Predict on the scaled test features. Returns mean prediction and standard deviation.
    y_pred_scaled, std_dev_scaled = gpr.predict(X_test_scaled, return_std=True)
    # y_pred_scaled is a numpy array of scaled radius predictions.

    # --- Stage 9: Descale Predictions ---
    if VERBOSE:
        print("\n--- Stage 9: Descaling Predictions to Original Radius Units ---")
    # Wrap y_pred_scaled (numpy array) into a DataFrame for the descale_data function.
    # Preserve index from X_test for potential alignment later if needed.
    y_pred_scaled_df = pd.DataFrame(y_pred_scaled, columns=[TARGET_RADIUS_COLUMN], index=X_test.index)
    
    # Descale the predictions using the scaler fitted for the target variable (y_scalers)
    y_pred_descaled_df = descale_data(y_pred_scaled_df, [TARGET_RADIUS_COLUMN], y_scalers)
    y_pred_descaled = y_pred_descaled_df[TARGET_RADIUS_COLUMN] # Get descaled predictions as a Series

    # y_test is already in original (descaled) radius units because it was split from 'y' before 'y_train' was scaled.
    y_true_descaled = y_test 

    # --- Stage 10: Evaluate Model Performance ---
    if VERBOSE:
        print("\n--- Stage 10: Evaluating Model Performance (on Descaled Radius Values) ---")
    
    # Calculate standard regression metrics
    mae = mean_absolute_error(y_true_descaled, y_pred_descaled)
    rmse = np.sqrt(mean_squared_error(y_true_descaled, y_pred_descaled))
    r2 = r2_score(y_true_descaled, y_pred_descaled)

    # Extract unit for printing, if TARGET_RADIUS_COLUMN has one (e.g., "Radius_mm" -> "mm")
    radius_unit = TARGET_RADIUS_COLUMN.split('_')[-1] if '_' in TARGET_RADIUS_COLUMN else 'units'
    
    print(f"  Mean Absolute Error (MAE):      {mae:.4f} {radius_unit}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} {radius_unit}")
    print(f"  R-squared (R2 Score):           {r2:.4f}")

    # Generate and save True vs. Predicted plot
    plot_filename_prefix = "gpr_80_20"
    if SESSIONS_TO_PROCESS:
        session_str = "_".join(SESSIONS_TO_PROCESS)
        plot_filename_prefix += f"_sessions_{session_str}"
    else:
        plot_filename_prefix += "_all_sessions"
    
    generate_true_vs_predicted_plot(
        y_true_descaled, 
        y_pred_descaled, 
        model_output_full_path, 
        filename_prefix=plot_filename_prefix,
        target_unit=radius_unit
    )

    # --- Prepare data for JSON output ---
    output_summary = {
        "model_type": "GaussianProcessRegressor",
        "script_variant": "80_20_split",
        "sessions_processed": SESSIONS_TO_PROCESS if SESSIONS_TO_PROCESS else "all",
        "input_data_directory": input_data_dir_path,
        "output_directory": model_output_full_path,
        "target_variable": TARGET_RADIUS_COLUMN,
        "radius_zero_handling": RADIUS_ZERO_HANDLING,
        "gpr_kernel_initial_config": str(GPR_KERNEL), # Store initial kernel as string
        "gpr_kernel_optimized_config": str(gpr.kernel_), # Store optimized kernel as string
        "gpr_alpha": GPR_ALPHA,
        "gpr_n_restarts_optimizer": GPR_N_RESTARTS_OPTIMIZER,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "performance_metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "units": radius_unit
        },
        "data_summary": {
            "total_files_processed": len(files_to_process),
            "combined_data_shape_before_nan_drop": combined_df.shape if 'combined_df' in locals() else "N/A",
            "combined_data_shape_after_nan_drop": df_final_processed.shape if 'df_final_processed' in locals() else "N/A",
            "training_set_shape": X_train.shape if 'X_train' in locals() else "N/A",
            "test_set_shape": X_test.shape if 'X_test' in locals() else "N/A"
        }
    }

    # Define JSON output filename
    # Create a unique filename based on sessions or a general name
    if SESSIONS_TO_PROCESS:
        session_str = "_".join(SESSIONS_TO_PROCESS)
        json_filename = f"gpr_80_20_metrics_config_sessions_{session_str}.json"
    else:
        json_filename = "gpr_80_20_metrics_config_all_sessions.json"
    
    json_output_path = os.path.join(model_output_full_path, json_filename)

    # Write the summary to a JSON file
    try:
        with open(json_output_path, 'w') as f:
            json.dump(output_summary, f, indent=4)
        if VERBOSE:
            print(f"\nMetrics and configuration saved to: {json_output_path}")
    except Exception as e:
        print(f"Error saving metrics and configuration to JSON: {e}")


    # --- Stage 11: Optional - Convert Predicted Radius back to Curvature ---
    if VERBOSE:
        print("\n--- Stage 11: Converting Descaled Radius Predictions back to Curvature ---")
    
    # Initialize an array for predicted curvatures with NaNs
    predicted_curvature = np.full_like(y_pred_descaled.values, fill_value=np.nan, dtype=float)
    
    # Handle "straight" predictions (where radius was capped) - set curvature to 0.0
    straight_mask = np.isclose(y_pred_descaled.values, RADIUS_CAP_VALUE)
    predicted_curvature[straight_mask] = 0.0 
    
    # Handle "bend" predictions (non-straight and non-zero radius) - calculate 1/radius
    # Ensure not to divide by zero if any predicted radius is exactly zero (unlikely for GPR with RBF)
    bend_mask = (~straight_mask) & (~np.isclose(y_pred_descaled.values, 0))
    if np.any(bend_mask): # Check if there are any bend values to convert
        predicted_curvature[bend_mask] = 1.0 / y_pred_descaled.values[bend_mask]

    # Retrieve original true curvature for the test samples from the combined_df
    # y_test.index holds the original indices from combined_df before shuffling by train_test_split
    true_curvature_for_test_samples = combined_df.loc[y_test.index, CURVATURE_COLUMN]

    # Create a DataFrame to display example results for easy comparison
    results_comparison_df = pd.DataFrame({
        'True_Radius_mm': y_true_descaled.values,
        'Predicted_Radius_mm': y_pred_descaled.values,
        'Predicted_Curvature': predicted_curvature, # Units inverse of radius_unit
        'True_Curvature': true_curvature_for_test_samples.values # Original curvature units
    }, index=y_test.index) # Use original index for clarity
    
    print("  Example Results (first 5 test samples from combined data):")
    print(results_comparison_df.head())

    print("\nScript finished successfully.")
