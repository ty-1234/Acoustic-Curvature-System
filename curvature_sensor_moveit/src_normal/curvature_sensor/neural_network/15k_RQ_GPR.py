"""
gpr_RQ_wide_fft.py (Adapted for WIDE FFT feature set from 100Hz to 16000Hz)

A single, well-commented script to perform an end-to-end machine learning pipeline
for predicting sensor radius using Gaussian Process Regression (GPR).
This version uses an ISOTROPIC Rational Quadratic kernel, includes Position_cm as an input feature,
predicts Radius directly, and includes detailed error analysis.
It's updated to use FFT features from 100Hz to 16,000Hz.

Pipeline Steps:
1.  Configuration.
2.  File Discovery (expects MERGED CSVs from your syncer script).
3.  Load and Combine Data from Merged Files.
4.  Filter for Active Rows and Prepare Target (Radius_mm).
5.  Prepare Features (X - new wide FFTs and Position) and Target (y - Radius).
6.  Data Cleaning.
7.  Train-Test Split.
8.  Scaling (Features and Target).
9.  GPR Model Training (with ISOTROPIC Rational Quadratic kernel).
10. Prediction.
11. Descaling.
12. Evaluation (MAE, RMSE, R2 on RADIUS values).
13. Detailed Error Analysis (Segmented errors, Residual plots on RADIUS).
14. Save outputs.
"""

# Standard library imports
import pandas as pd
import numpy as np
import os
import glob # For finding files matching a pattern
import json # For JSON output
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For enhanced plotting

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, ConstantKernel as C, Kernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Configuration Variables ---

# --- File and Directory Settings ---
INPUT_DATA_PARENT_DIR = "csv_data"
# This script now expects to load MERGED data (output of your syncer script)
CLEANED_DATA_SUBFOLDER = "merged" # MODIFIED: Point to your merged data directory
MODEL_OUTPUT_DIR = "model_outputs"
SESSIONS_TO_PROCESS = ["1"] # Or ["1", "2"] etc., or [] for all.
                            # This filters based on [test N] in merged filenames.

# --- Column Names ---
# UPDATED FFT_COLUMNS for the new dataset (100Hz to 16000Hz, 100Hz steps)
FFT_COLUMNS = [f'FFT_{i}Hz' for i in range(100, 16001, 100)]
POSITION_COLUMN = 'Position_cm'
INPUT_FEATURE_COLUMNS = FFT_COLUMNS + [POSITION_COLUMN]

# Columns expected in the MERGED CSV (output of your syncer script)
ACTIVITY_COLUMN = 'Curvature_Active' # Used for filtering active rows
CURVATURE_COLUMN_ORIGINAL = 'Curvature' # The curvature value from the syncer (0 for inactive)
TARGET_RADIUS_COLUMN = 'Radius_mm'    # This should be present in merged data or calculated

# --- Radius/Curvature Conversion Settings (used if TARGET_RADIUS_COLUMN needs to be calculated) ---
RADIUS_ZERO_HANDLING = 'cap'
RADIUS_CAP_VALUE = 1e6
RADIUS_PLACEHOLDER_VALUE = np.nan

# --- Gaussian Process Regression (GPR) Settings ---
# Using ISOTROPIC RationalQuadratic kernel (length_scale is a scalar)
GPR_KERNEL = C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0,
                                                      length_scale_bounds=(1e-2, 1e2),
                                                      alpha_bounds=(1e-2, 1e2)) + \
             WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
GPR_ALPHA = 1e-10
GPR_N_RESTARTS_OPTIMIZER = 10

# --- Train-Test Split Settings ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Script Control ---
VERBOSE = True
GENERATE_DETAILED_ERROR_ANALYSIS_PLOTS = True

# --- Helper Functions ---
def load_single_file_data(file_path):
    if VERBOSE: print(f"  Loading data from: {os.path.basename(file_path)}")
    try:
        df = pd.read_csv(file_path)
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True)
        if VERBOSE: print(f"    Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"    Error loading data from {file_path}: {e}")
        return None

def convert_curvature_to_radius(curvature_series, zero_handling, cap_value, placeholder_val):
    if VERBOSE: print(f"  Converting source curvature to radius (method: {zero_handling})...")
    radius_series = pd.Series(index=curvature_series.index, dtype=float)
    curvature_series_numeric = pd.to_numeric(curvature_series, errors='coerce')
    near_zero_mask = np.isclose(curvature_series_numeric, 0) | curvature_series_numeric.isnull()
    valid_curvature_mask = ~near_zero_mask & pd.notnull(curvature_series_numeric)
    if valid_curvature_mask.any():
        radius_series.loc[valid_curvature_mask] = 1.0 / curvature_series_numeric[valid_curvature_mask]
    zero_or_nan_mask = near_zero_mask
    if zero_or_nan_mask.any():
        if zero_handling == 'cap': radius_series.loc[zero_or_nan_mask] = cap_value
        elif zero_handling == 'placeholder': radius_series.loc[zero_or_nan_mask] = placeholder_val
        elif zero_handling == 'infinity':
            with np.errstate(divide='ignore'): radius_series.loc[zero_or_nan_mask] = np.inf
        elif zero_handling == 'warn_skip':
            if VERBOSE: print("    Warning: Zero/near-zero or NaN curvature. Setting radius to NaN.")
            radius_series.loc[zero_or_nan_mask] = np.nan
        else: raise ValueError(f"Invalid 'zero_handling' method: {zero_handling}.")
    return radius_series

def scale_data(data_to_scale, columns, existing_scalers=None):
    if VERBOSE and existing_scalers is None: print(f"  Fitting scalers and scaling data for columns: {columns}")
    elif VERBOSE and existing_scalers is not None: print(f"  Transforming data using existing scalers for columns: {columns}")
    scaled_data = data_to_scale.copy()
    if isinstance(data_to_scale, pd.Series): 
        col_name_for_series = data_to_scale.name if data_to_scale.name else columns[0]
        scaled_data = pd.DataFrame(scaled_data, columns=[col_name_for_series])
    fitted_scalers = {} if existing_scalers is None else existing_scalers
    for col in columns:
        if col not in scaled_data.columns:
            if VERBOSE: print(f"    Warning: Column '{col}' not in data for scaling. Skipping.")
            continue
        if scaled_data[col].isnull().any():
             mean_val = scaled_data[col].mean()
             if pd.isna(mean_val): mean_val = 0
             scaled_data[col].fillna(mean_val, inplace=True)
             if VERBOSE: print(f"    Filled NaN values in column '{col}' with mean ({mean_val:.4f}) before scaling.")
        column_data = scaled_data[col].values.reshape(-1, 1)
        if existing_scalers is None:
            scaler = MinMaxScaler()
            scaled_column_data = scaler.fit_transform(column_data)
            fitted_scalers[col] = scaler
        else:
            if col in existing_scalers:
                scaler = existing_scalers[col]
                scaled_column_data = scaler.transform(column_data)
            else:
                if VERBOSE: print(f"    Warning: No existing scaler for column '{col}'. Skipping transform.")
                continue 
        scaled_data[col] = scaled_column_data.flatten()
    return scaled_data, fitted_scalers

def descale_data(data_to_descale, columns, scalers_dict):
    if VERBOSE: print(f"  Descaling data for columns: {columns}")
    descaled_data = data_to_descale.copy()
    if isinstance(data_to_descale, pd.Series):
         col_name_for_series = data_to_descale.name if data_to_descale.name else columns[0]
         descaled_data = pd.DataFrame(descaled_data, columns=[col_name_for_series])
    for col in columns:
        if col not in descaled_data.columns:
            if VERBOSE: print(f"    Warning: Column '{col}' not in data for descaling. Skipping.")
            continue
        if col not in scalers_dict:
            if VERBOSE: print(f"    Warning: No scaler found for column '{col}'. Skipping descaling.")
            continue
        scaler = scalers_dict[col]
        column_data = descaled_data[col].values.reshape(-1, 1)
        original_scale_data = scaler.inverse_transform(column_data)
        descaled_data[col] = original_scale_data.flatten()
    return descaled_data

# --- Plotting and Error Analysis Functions ---
def generate_true_vs_predicted_plot(y_true, y_pred, output_dir, filename_prefix, target_unit):
    if VERBOSE: print(f"\n  Generating True vs. Predicted plot (Radius in {target_unit})...")
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Predictions", s=15, edgecolors='k', linewidths=0.5)
    min_val = min(np.nanmin(y_true), np.nanmin(y_pred)) if len(y_true) > 0 and len(y_pred) > 0 and not (np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred))) else 0
    max_val = max(np.nanmax(y_true), np.nanmax(y_pred)) if len(y_true) > 0 and len(y_pred) > 0 and not (np.all(np.isnan(y_true)) or np.all(np.isnan(y_pred))) else 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal (y=x)")
    plt.xlabel(f"True Radius ({target_unit})")
    plt.ylabel(f"Predicted Radius ({target_unit})")
    plt.title(f"True vs. Predicted Radius ({filename_prefix.replace('_', ' ').title()})")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_filename = f"{filename_prefix}_true_vs_predicted_radius.png" 
    plot_output_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_output_path); plt.close()
        if VERBOSE: print(f"    Plot saved to: {plot_output_path}")
    except Exception as e: print(f"    Error saving plot: {e}")

def calculate_segmented_errors(y_true, y_pred, segment_bins, target_name="Radius", target_unit="units"):
    if VERBOSE: print(f"\n  Calculating Segmented Errors for {target_name} ({target_unit}):")
    y_true_series = pd.Series(y_true, name="true")
    y_pred_series = pd.Series(y_pred, name="pred")
    df_eval = pd.concat([y_true_series, y_pred_series], axis=1).dropna()
    if df_eval.empty:
        logging.warning("  No valid data points for segmented error calculation after NaN drop.")
        return pd.DataFrame()
    df_eval['error'] = df_eval['true'] - df_eval['pred']
    segment_labels = [f"{segment_bins[i]:.1f}-{segment_bins[i+1]:.1f}" for i in range(len(segment_bins)-1)]
    df_eval['segment'] = pd.cut(df_eval['true'], bins=segment_bins, labels=segment_labels, right=False, include_lowest=True)
    segmented_results = []
    for segment in segment_labels:
        segment_df = df_eval[df_eval['segment'] == segment]
        if not segment_df.empty:
            mae = mean_absolute_error(segment_df['true'], segment_df['pred'])
            rmse = np.sqrt(mean_squared_error(segment_df['true'], segment_df['pred']))
            mean_err = segment_df['error'].mean()
            count = len(segment_df)
            segmented_results.append({
                "Segment": segment, "MAE": mae, "RMSE": rmse, "Mean_Error (Bias)": mean_err, "Count": count
            })
            if VERBOSE: print(f"    Segment {segment}: MAE={mae:.2f}, RMSE={rmse:.2f}, Bias={mean_err:.2f}, Count={count}")
        else:
            if VERBOSE: print(f"    Segment {segment}: No data points.")
    return pd.DataFrame(segmented_results)

def plot_residuals(residuals, x_values, x_label, y_label_base, title, output_dir, filename):
    if VERBOSE: print(f"  Generating Residual Plot: {title}...")
    residuals_np = np.array(residuals)
    x_values_np = np.array(x_values)
    valid_indices = ~np.isnan(residuals_np) & ~np.isnan(x_values_np)
    residuals_clean = residuals_np[valid_indices]
    x_values_clean = x_values_np[valid_indices]
    if len(residuals_clean) == 0:
        logging.warning(f"  No valid data points for residual plot: {title}. Skipping plot.")
        return
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values_clean, residuals_clean, alpha=0.5, s=15, edgecolors='k', linewidths=0.5)
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel(x_label)
    plt.ylabel(f"Residual ({y_label_base})")
    plt.title(title)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_output_path = os.path.join(output_dir, filename)
    try:
        plt.savefig(plot_output_path); plt.close()
        if VERBOSE: print(f"    Residual plot saved to: {plot_output_path}")
    except Exception as e: print(f"    Error saving residual plot: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    run_timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    session_str_for_dir = "_".join(SESSIONS_TO_PROCESS) if SESSIONS_TO_PROCESS else "all"
    run_specific_dir_name = f"run_{run_timestamp}_iso_rq_radius_pos_widefft_sessions_{session_str_for_dir}" # Added widefft
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_data_dir_path = os.path.join(script_dir, "..", INPUT_DATA_PARENT_DIR, CLEANED_DATA_SUBFOLDER) 
    model_output_full_path = os.path.join(script_dir, MODEL_OUTPUT_DIR, run_specific_dir_name)

    if not os.path.isdir(input_data_dir_path):
        print(f"Error: Input data directory for MERGED files '{input_data_dir_path}' not found."); exit()
    if not os.path.isdir(model_output_full_path):
        os.makedirs(model_output_full_path, exist_ok=True)
        if VERBOSE: print(f"Created run-specific output directory: {model_output_full_path}")

    all_files_in_dir = glob.glob(os.path.join(input_data_dir_path, "merged_*.csv"))
    files_to_process = []
    if not SESSIONS_TO_PROCESS:
        if VERBOSE: print("No specific sessions defined. Processing all 'merged_*.csv' files found.")
        files_to_process = all_files_in_dir
    else:
        for session_num in SESSIONS_TO_PROCESS:
            session_tag = f"[test {session_num}]".lower()
            for f_path in all_files_in_dir:
                if session_tag in os.path.basename(f_path).lower():
                    files_to_process.append(f_path)
    
    if not files_to_process: print(f"No 'merged_*.csv' files found to process in '{input_data_dir_path}' matching specified sessions. Exiting."); exit()
    files_to_process = sorted(list(set(files_to_process)))
    if VERBOSE:
        print(f"Found {len(files_to_process)} MERGED files to process for GPR modeling:")
        for i, f in enumerate(files_to_process): print(f"  {i+1}. {os.path.basename(f)}")

    all_dfs_from_merged_files = []
    if VERBOSE: print("\n--- Stage 1: Loading and Combining Data from Merged Files ---")
    for file_path in files_to_process:
        df_single_merged_file = load_single_file_data(file_path)
        if df_single_merged_file is not None and not df_single_merged_file.empty:
            for fft_col in FFT_COLUMNS: # Ensure all NEW FFT columns are present
                if fft_col not in df_single_merged_file.columns:
                    logging.warning(f"FFT column {fft_col} not found in {os.path.basename(file_path)}. Adding as zeros.")
                    df_single_merged_file[fft_col] = 0.0
            all_dfs_from_merged_files.append(df_single_merged_file)
        else:
            logging.warning(f"Skipping empty or unreadable file: {os.path.basename(file_path)}")

    if not all_dfs_from_merged_files: print("No data available after loading all merged files. Exiting."); exit()
    combined_df = pd.concat(all_dfs_from_merged_files, ignore_index=True)
    if VERBOSE: print(f"Combined data from all merged files. Shape: {combined_df.shape}")

    if VERBOSE: print("\n--- Stage 2: Filtering Active Rows and Preparing Target ---")
    if ACTIVITY_COLUMN not in combined_df.columns:
        print(f"Error: Activity column '{ACTIVITY_COLUMN}' not found in combined data. Exiting."); exit()
    active_df = combined_df[combined_df[ACTIVITY_COLUMN] == 1].copy()
    if active_df.empty:
        print("No active rows (Curvature_Active == 1) found. Exiting."); exit()
    if TARGET_RADIUS_COLUMN not in active_df.columns or active_df[TARGET_RADIUS_COLUMN].isnull().all():
        if CURVATURE_COLUMN_ORIGINAL not in active_df.columns:
            print(f"Error: Neither '{TARGET_RADIUS_COLUMN}' nor '{CURVATURE_COLUMN_ORIGINAL}' found. Exiting."); exit()
        if VERBOSE: print(f"  '{TARGET_RADIUS_COLUMN}' not found/all NaNs, calculating from '{CURVATURE_COLUMN_ORIGINAL}'.")
        active_df[TARGET_RADIUS_COLUMN] = convert_curvature_to_radius(
            active_df[CURVATURE_COLUMN_ORIGINAL], RADIUS_ZERO_HANDLING, RADIUS_CAP_VALUE, RADIUS_PLACEHOLDER_VALUE
        )
    if VERBOSE: print(f"  Active data shape: {active_df.shape}")

    required_cols_for_modeling = INPUT_FEATURE_COLUMNS + [TARGET_RADIUS_COLUMN]
    if not all(col in active_df.columns for col in required_cols_for_modeling):
        missing = [c for c in required_cols_for_modeling if c not in active_df.columns]
        # Check if missing columns are part of the new FFT_COLUMNS list
        missing_fft = [m for m in missing if m in FFT_COLUMNS]
        if len(missing_fft) == len(missing): # All missing are FFT columns
             print(f"Warning: Some FFT columns specified in FFT_COLUMNS list are not in the loaded data: {missing_fft}")
             print("         This might be okay if the loaded merged files are from an older FFT feature set.")
             print("         The script will proceed with available columns. Ensure FFT_COLUMNS matches your data.")
             # Update INPUT_FEATURE_COLUMNS to only use available columns
             INPUT_FEATURE_COLUMNS_AVAILABLE = [col for col in INPUT_FEATURE_COLUMNS if col in active_df.columns]
             if POSITION_COLUMN not in INPUT_FEATURE_COLUMNS_AVAILABLE and POSITION_COLUMN in active_df.columns: # Ensure pos is still there
                 INPUT_FEATURE_COLUMNS_AVAILABLE.append(POSITION_COLUMN)
             if not any(fft_c in INPUT_FEATURE_COLUMNS_AVAILABLE for fft_c in FFT_COLUMNS):
                 print("Error: NO FFT columns found in the data. Exiting.")
                 exit()
             print(f"Using available input features: {INPUT_FEATURE_COLUMNS_AVAILABLE}")
             X_all_active = active_df[INPUT_FEATURE_COLUMNS_AVAILABLE]
        else: # Other critical columns are missing
            print(f"Error: Missing critical columns for modeling in active_df: {missing}. Exiting."); exit()
    else:
        X_all_active = active_df[INPUT_FEATURE_COLUMNS]
        INPUT_FEATURE_COLUMNS_AVAILABLE = INPUT_FEATURE_COLUMNS # All are available

    y_all_active = active_df[TARGET_RADIUS_COLUMN]
    original_rows = len(active_df)
    df_for_nan_check = pd.concat([X_all_active, y_all_active], axis=1)
    df_for_nan_check.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final_processed = df_for_nan_check.dropna()
    if VERBOSE and len(df_final_processed) < original_rows:
        print(f"  Dropped {original_rows - len(df_final_processed)} active rows due to NaN/Inf values.")
    if df_final_processed.empty: print("No data remaining after NaN/Inf drop. Exiting."); exit()
    X = df_final_processed[INPUT_FEATURE_COLUMNS_AVAILABLE] # Use available features
    y = df_final_processed[TARGET_RADIUS_COLUMN] 

    if VERBOSE: print("\n--- Stage 4: Train-Test Split ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    if VERBOSE: print(f"Data split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")

    if VERBOSE: print("\n--- Stage 5: Scaling Features and Target ---")
    X_train_scaled, x_scalers = scale_data(X_train, INPUT_FEATURE_COLUMNS_AVAILABLE) # Use available features
    X_test_scaled, _ = scale_data(X_test, INPUT_FEATURE_COLUMNS_AVAILABLE, existing_scalers=x_scalers) # Use available
    y_train_df = pd.DataFrame(y_train, columns=[TARGET_RADIUS_COLUMN]) 
    y_train_scaled_df, y_target_scalers = scale_data(y_train_df, [TARGET_RADIUS_COLUMN])
    y_train_scaled = y_train_scaled_df[TARGET_RADIUS_COLUMN]

    if VERBOSE: print("\n--- Stage 6: Training GPR Model ---")
    print(f"  Predicting GPR Target: RADIUS (Column: {TARGET_RADIUS_COLUMN})")
    print(f"  Using Kernel (Isotropic Rational Quadratic): {GPR_KERNEL}")
    gpr = GaussianProcessRegressor(kernel=GPR_KERNEL, alpha=GPR_ALPHA, n_restarts_optimizer=GPR_N_RESTARTS_OPTIMIZER, random_state=RANDOM_STATE)
    gpr.fit(X_train_scaled, y_train_scaled)
    if VERBOSE:
        print("GPR model training complete.")
        print(f"  Optimized Kernel: {gpr.kernel_}") 
        print(f"  Log-Marginal-Likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}")

    if VERBOSE: print("\n--- Stage 7: Making Predictions ---") 
    y_pred_scaled, _ = gpr.predict(X_test_scaled, return_std=True)

    if VERBOSE: print("\n--- Stage 8: Descaling Predicted GPR Target ---")
    y_pred_scaled_df = pd.DataFrame(y_pred_scaled, columns=[TARGET_RADIUS_COLUMN], index=X_test.index)
    y_pred_descaled_df = descale_data(y_pred_scaled_df, [TARGET_RADIUS_COLUMN], y_target_scalers)
    y_pred_descaled = y_pred_descaled_df[TARGET_RADIUS_COLUMN]

    y_pred_evaluation_radius = y_pred_descaled 
    y_true_evaluation_radius = y_test 

    if VERBOSE: print(f"\n--- Stage 10: Evaluating Model Performance (on {TARGET_RADIUS_COLUMN}) ---")
    mae = mean_absolute_error(y_true_evaluation_radius, y_pred_evaluation_radius)
    rmse = np.sqrt(mean_squared_error(y_true_evaluation_radius, y_pred_evaluation_radius))
    r2 = r2_score(y_true_evaluation_radius, y_pred_evaluation_radius)
    radius_unit_eval = TARGET_RADIUS_COLUMN.split('_')[-1] if '_' in TARGET_RADIUS_COLUMN else 'mm'
    print(f"  Mean Absolute Error (MAE) on Radius:      {mae:.4f} {radius_unit_eval}")
    print(f"  Root Mean Squared Error (RMSE) on Radius: {rmse:.4f} {radius_unit_eval}")
    print(f"  R-squared (R2 Score) on Radius:           {r2:.4f}")

    base_filename_prefix = f"gpr_iso_rq_pos_radius_widefft_80_20" # Added widefft
    if SESSIONS_TO_PROCESS:
        session_str_for_file = "_".join(SESSIONS_TO_PROCESS)
        final_filename_prefix = f"{base_filename_prefix}_sessions_{session_str_for_file}"
    else:
        final_filename_prefix = f"{base_filename_prefix}_all_sessions"
    generate_true_vs_predicted_plot(
        y_true_evaluation_radius, y_pred_evaluation_radius, 
        model_output_full_path, filename_prefix=final_filename_prefix,
        target_unit=radius_unit_eval
    )
    
    if VERBOSE: print("\n--- Stage 11: Detailed Error Analysis (on Radius Predictions) ---")
    residuals_radius = y_true_evaluation_radius - y_pred_evaluation_radius
    min_true_rad = y_true_evaluation_radius.min()
    max_true_rad = y_true_evaluation_radius.max()
    if max_true_rad > min_true_rad :
        num_bins = 5
        if max_true_rad < RADIUS_CAP_VALUE * 0.9 :
             bins = np.linspace(min_true_rad, max_true_rad + 1e-9, num_bins + 1)
        else:
            pre_cap_max_val = y_true_evaluation_radius[y_true_evaluation_radius < RADIUS_CAP_VALUE * 0.9].max()
            if pd.isna(pre_cap_max_val) or pre_cap_max_val <= min_true_rad : 
                sensible_upper_bound = min(max_true_rad, 250) 
                pre_cap_max_val = min_true_rad + (sensible_upper_bound - min_true_rad) * (num_bins -1)/num_bins
            bins_up_to_pre_cap = np.linspace(min_true_rad, pre_cap_max_val, num_bins)
            bins = np.append(bins_up_to_pre_cap, [max_true_rad + 1e-9])
            bins = np.unique(bins) 
            if len(bins) < 2 : 
                 bins = np.linspace(min_true_rad, max_true_rad + 1e-9, 2)
    else: bins = [min_true_rad, max_true_rad + 1e-9] 
    if len(bins) < 2: 
        bins = np.array([min_true_rad, max_true_rad + 1e-9]) if min_true_rad < max_true_rad else np.array([min_true_rad - 0.5, min_true_rad + 0.5])
    segmented_error_df = calculate_segmented_errors(
        y_true_evaluation_radius, y_pred_evaluation_radius, segment_bins=bins,
        target_name="Radius", target_unit=radius_unit_eval
    )
    print("\nSegmented Error Analysis (Radius):\n", segmented_error_df)

    if GENERATE_DETAILED_ERROR_ANALYSIS_PLOTS:
        if VERBOSE: print("\n  Generating Residual Plots (on Radius)...")
        plot_residuals(residuals_radius, y_true_evaluation_radius, 
                       f"True Radius ({radius_unit_eval})", "Radius",
                       "Residuals vs. True Radius", model_output_full_path, 
                       f"{final_filename_prefix}_residuals_vs_true_radius.png")
        plot_residuals(residuals_radius, y_pred_evaluation_radius, 
                       f"Predicted Radius ({radius_unit_eval})", "Radius",
                       "Residuals vs. Predicted Radius", model_output_full_path, 
                       f"{final_filename_prefix}_residuals_vs_predicted_radius.png")
        for feature_col in INPUT_FEATURE_COLUMNS_AVAILABLE: # Use available features for plotting
            plot_residuals(residuals_radius, X_test_scaled[feature_col], 
                           f"Scaled {feature_col}", "Radius",
                           f"Residuals vs. Scaled {feature_col}", model_output_full_path, 
                           f"{final_filename_prefix}_residuals_vs_scaled_{feature_col}.png")
        plt.figure(figsize=(8,6)); sns.histplot(residuals_radius, kde=True)
        plt.title("Distribution of Residuals (Radius)"); plt.xlabel("Residual (Radius)"); plt.ylabel("Frequency")
        plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout()
        res_dist_path = os.path.join(model_output_full_path, f"{final_filename_prefix}_residuals_distribution.png")
        try: plt.savefig(res_dist_path); plt.close()
        except Exception as e: print(f"    Error saving residual distribution plot: {e}")

    output_summary = {
        "model_type": "GaussianProcessRegressor", "kernel_type": "Isotropic_RationalQuadratic", 
        "target_predicted_by_gpr": TARGET_RADIUS_COLUMN,
        "evaluation_always_on_radius": TARGET_RADIUS_COLUMN,
        "input_features_used": INPUT_FEATURE_COLUMNS_AVAILABLE, # Log available/used features
        "script_variant": f"80_20_iso_rq_pos_radius_widefft", # Added widefft
        "sessions_processed": SESSIONS_TO_PROCESS if SESSIONS_TO_PROCESS else "all",
        "input_data_directory": input_data_dir_path, "output_directory": model_output_full_path,
        "radius_zero_handling_for_source_data": RADIUS_ZERO_HANDLING,
        "gpr_kernel_initial_config": GPR_KERNEL.get_params(deep=True), 
        "gpr_kernel_optimized_config": gpr.kernel_.get_params(deep=True) if hasattr(gpr, 'kernel_') else str(gpr.kernel_),
        "gpr_alpha_value_added_to_diagonal": GPR_ALPHA, "gpr_n_restarts_optimizer": GPR_N_RESTARTS_OPTIMIZER,
        "test_size_split_ratio": TEST_SIZE, "random_state_seed": RANDOM_STATE,
        "performance_metrics_on_radius": {
            "mae": mae, "rmse": rmse, "r2_score": r2, "units": radius_unit_eval
        },
        "data_summary": {
            "total_files_processed": len(files_to_process),
            "combined_data_shape_before_processing_active_filter": combined_df.shape if 'combined_df' in locals() else "N/A",
            "active_data_shape_before_nan_drop": active_df.shape if 'active_df' in locals() else "N/A",
            "final_processed_data_shape_for_modeling": df_final_processed.shape if 'df_final_processed' in locals() else "N/A",
            "training_set_shape": X_train.shape if 'X_train' in locals() else "N/A",
            "test_set_shape": X_test.shape if 'X_test' in locals() else "N/A"
        },
        "segmented_radius_errors": segmented_error_df.to_dict(orient='records') if 'segmented_error_df' in locals() and not segmented_error_df.empty else "N/A"
    }
    json_filename = f"{final_filename_prefix}_metrics_config.json"
    json_output_path = os.path.join(model_output_full_path, json_filename)
    try:
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                if isinstance(obj, Kernel): 
                    params = obj.get_params(deep=False)
                    if 'length_scale' in params and isinstance(params['length_scale'], (list, np.ndarray)):
                        params['length_scale'] = f"list_of_{len(params['length_scale'])}_items" 
                    return params
                return super(NpEncoder, self).default(obj)
        with open(json_output_path, 'w') as f: json.dump(output_summary, f, indent=4, sort_keys=True, cls=NpEncoder)
        if VERBOSE: print(f"\nMetrics and configuration saved to: {json_output_path}")
    except Exception as e: print(f"Error saving metrics to JSON: {e}")

    if VERBOSE: print(f"\n--- Stage 13: Example Output Table ---")
    example_table_df = pd.DataFrame(index=y_true_evaluation_radius.index)
    example_table_df[f'True_{TARGET_RADIUS_COLUMN}'] = y_true_evaluation_radius
    example_table_df[f'Predicted_{TARGET_RADIUS_COLUMN}'] = y_pred_evaluation_radius
    pred_curv_from_rad = np.full_like(y_pred_evaluation_radius.values, np.nan)
    straight_mask = np.isclose(y_pred_evaluation_radius.values, RADIUS_CAP_VALUE)
    pred_curv_from_rad[straight_mask] = 0.0
    bend_mask = (~straight_mask) & (~np.isclose(y_pred_evaluation_radius.values, 0))
    if np.any(bend_mask): pred_curv_from_rad[bend_mask] = 1.0 / y_pred_evaluation_radius.values[bend_mask]
    example_table_df['Predicted_Curvature_from_Radius'] = pred_curv_from_rad
    example_table_df['True_Curvature'] = combined_df.loc[y_true_evaluation_radius.index, CURVATURE_COLUMN_ORIGINAL]
    print(f"  Example Results (first 5 test samples, GPR predicted RADIUS):")
    print(example_table_df.head())

    print("\nScript finished successfully.")
