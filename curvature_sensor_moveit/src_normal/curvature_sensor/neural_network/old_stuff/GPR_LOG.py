"""
gpr_pipeline_multi_file.py (Modified for Log-Radius Target, Error Analysis & Optional Curvature Prediction)

A single, well-commented script to perform an end-to-end machine learning pipeline
for predicting sensor radius (or curvature, or log-radius) using Gaussian Process Regression (GPR).
This version uses a Rational Quadratic kernel, includes Position_cm as an input feature,
allows choosing the prediction target, and includes detailed error analysis.

Pipeline Steps:
1.  Configuration.
2.  File Discovery.
3.  Per-File Preprocessing Loop (FFT Norm, Curvature to Radius).
4.  Target Transformation (if predicting LOG_RADIUS, apply log here).
5.  Combine Data.
6.  Prepare Features (X) and Target (y - Radius, Curvature, or Log-Radius).
7.  Data Cleaning.
8.  Train-Test Split.
9.  Scaling (Features and chosen Target).
10. GPR Model Training.
11. Prediction (scaled chosen target).
12. Descaling (predicted chosen target).
13. Inverse Target Transformation (if predicting LOG_RADIUS, apply exp here. If CURVATURE, convert to Radius).
14. Evaluation (MAE, RMSE, R2 on RADIUS values for consistent comparison).
15. Detailed Error Analysis (Segmented errors, Residual plots on RADIUS).
16. Save outputs.
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
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Configuration Variables ---

# --- File and Directory Settings ---
INPUT_DATA_PARENT_DIR = "csv_data"
CLEANED_DATA_SUBFOLDER = "cleaned"
MODEL_OUTPUT_DIR = "model_outputs"
# Update SESSIONS_TO_PROCESS as needed, e.g., ["1", "2"]
SESSIONS_TO_PROCESS = ["1"]

# --- Target Variable Choice ---
# Options: "RADIUS", "CURVATURE", "LOG_RADIUS"
PREDICT_TARGET = "LOG_RADIUS" # MODIFIED TO TEST LOG_RADIUS

# --- Column Names ---
FFT_COLUMNS = [f'FFT_{i}Hz' for i in range(200, 2001, 200)]
POSITION_COLUMN = 'Position_cm'
INPUT_FEATURE_COLUMNS = FFT_COLUMNS + [POSITION_COLUMN]

ACTIVITY_COLUMN = 'Curvature_Active'
CURVATURE_COLUMN = 'Curvature'
# Generic name for the column the GPR model will directly predict (can be Radius, Curvature, or LogRadius)
TARGET_COLUMN_FOR_MODEL = 'Target_For_GPR'
# Column name for the original radius values, used for consistent evaluation
EVALUATION_TARGET_RADIUS_COLUMN = 'Radius_mm'
# Column name for log-transformed radius if PREDICT_TARGET is "LOG_RADIUS"
LOG_RADIUS_COLUMN_NAME = 'Log_Radius_mm'


# --- Radius/Curvature Conversion Settings ---
RADIUS_ZERO_HANDLING = 'cap'
RADIUS_CAP_VALUE = 1e6
RADIUS_PLACEHOLDER_VALUE = np.nan
# Small epsilon to add before log transform if radius can be zero,
# though RADIUS_CAP_VALUE for Curvature=0 should make Radius_mm > 0.
# Using np.log directly as Radius_mm should be positive.
LOG_EPSILON = 1e-9 # Only used if np.log1p was chosen, not currently.

# --- Gaussian Process Regression (GPR) Settings ---
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
        if VERBOSE: print(f"    Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"    Error loading data from {file_path}: {e}")
        return None

def apply_fft_baseline_normalization(df_input, fft_cols, pos_col, act_col, filename=""):
    df = df_input.copy()
    if VERBOSE: print(f"  Applying FFT baseline normalization for {filename}...")
    numeric_positions = pd.to_numeric(df[pos_col], errors='coerce')
    first_zero_or_positive_pos_idx = len(df)
    if (numeric_positions >= 0).any():
        first_zero_or_positive_pos_idx = (numeric_positions >= 0).idxmax()
    first_active_event_idx = len(df)
    if (df[act_col] != 0).any():
       first_active_event_idx = (df[act_col] != 0).idxmax()
    end_of_baseline_period_idx = min(first_zero_or_positive_pos_idx, first_active_event_idx)
    baseline_df = df.loc[(df.index < end_of_baseline_period_idx) & (df[act_col] == 0)]
    idle_fft_baseline = None
    if not baseline_df.empty:
        idle_fft_baseline = baseline_df[fft_cols].mean()
        if VERBOSE: print(f"    Calculated baseline from {len(baseline_df)} idle rows for {filename}.")
    elif VERBOSE: print(f"    Warning: No valid idle frames found for FFT baseline in {filename}.")
    active_df = df[df[act_col] == 1].copy()
    if VERBOSE: print(f"    Number of active rows in {filename} (where '{act_col}' == 1): {active_df.shape[0]}")
    if idle_fft_baseline is not None and not active_df.empty:
        for col in fft_cols:
            if col in idle_fft_baseline: active_df[col] = active_df[col] - idle_fft_baseline[col]
        if VERBOSE: print(f"    FFT baseline subtracted from active rows in {filename}.")
    elif idle_fft_baseline is None and not active_df.empty and VERBOSE:
        print(f"    FFT features for active rows in {filename} not normalized (no valid baseline).")
    return active_df, idle_fft_baseline

def convert_curvature_to_radius(curvature_series, zero_handling, cap_value, placeholder_val):
    if VERBOSE: print(f"  Converting source curvature to radius (method: {zero_handling})...")
    radius_series = pd.Series(index=curvature_series.index, dtype=float)
    near_zero_mask = np.isclose(curvature_series, 0)
    if (~near_zero_mask).any():
        radius_series.loc[~near_zero_mask] = 1.0 / curvature_series[~near_zero_mask]
    if near_zero_mask.any():
        if zero_handling == 'cap': radius_series.loc[near_zero_mask] = cap_value
        elif zero_handling == 'placeholder': radius_series.loc[near_zero_mask] = placeholder_val
        elif zero_handling == 'infinity':
            with np.errstate(divide='ignore'): radius_series.loc[near_zero_mask] = np.inf
        elif zero_handling == 'warn_skip':
            if VERBOSE: print("    Warning: Zero/near-zero curvature. Setting radius to NaN.")
            radius_series.loc[near_zero_mask] = np.nan
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
             raise ValueError(f"Column '{col}' contains NaN values. Handle NaNs before scaling.")
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
def generate_true_vs_predicted_plot(y_true, y_pred, output_dir, filename_prefix, target_name_for_plot, target_unit):
    if VERBOSE: print(f"\n  Generating True vs. Predicted plot ({target_name_for_plot} in {target_unit})...")
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, label="Predictions", s=15, edgecolors='k', linewidths=0.5)
    min_val = min(np.min(y_true), np.min(y_pred)) if len(y_true) > 0 and len(y_pred) > 0 else 0
    max_val = max(np.max(y_true), np.max(y_pred)) if len(y_true) > 0 and len(y_pred) > 0 else 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal (y=x)")
    plt.xlabel(f"True {target_name_for_plot} ({target_unit})")
    plt.ylabel(f"Predicted {target_name_for_plot} ({target_unit})")
    plt.title(f"True vs. Predicted {target_name_for_plot} ({filename_prefix.replace('_', ' ').title()})")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plot_filename = f"{filename_prefix}_true_vs_predicted_{target_name_for_plot.lower().replace(' ', '_')}.png"
    plot_output_path = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_output_path); plt.close()
        if VERBOSE: print(f"    Plot saved to: {plot_output_path}")
    except Exception as e: print(f"    Error saving plot: {e}")

def calculate_segmented_errors(y_true, y_pred, segment_bins, target_name="Target", target_unit="units"):
    if VERBOSE: print(f"\n  Calculating Segmented Errors for {target_name} ({target_unit}):")
    y_true_series = pd.Series(y_true, name="true")
    y_pred_series = pd.Series(y_pred, name="pred")
    df_eval = pd.concat([y_true_series, y_pred_series], axis=1)
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
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, residuals, alpha=0.5, s=15, edgecolors='k', linewidths=0.5)
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
    # --- Setup Output Directory ---
    run_timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    session_str_for_dir = "_".join(SESSIONS_TO_PROCESS) if SESSIONS_TO_PROCESS else "all"
    run_specific_dir_name = f"run_{run_timestamp}_{PREDICT_TARGET.lower()}_sessions_{session_str_for_dir}"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_data_dir_path = os.path.join(script_dir, INPUT_DATA_PARENT_DIR, CLEANED_DATA_SUBFOLDER)
    model_output_full_path = os.path.join(script_dir, MODEL_OUTPUT_DIR, run_specific_dir_name)

    if not os.path.isdir(input_data_dir_path):
        print(f"Error: Input data directory '{input_data_dir_path}' not found."); exit()
    if not os.path.isdir(model_output_full_path):
        os.makedirs(model_output_full_path, exist_ok=True)
        if VERBOSE: print(f"Created run-specific output directory: {model_output_full_path}")

    # --- File Discovery ---
    all_files_in_dir = glob.glob(os.path.join(input_data_dir_path, "cleaned_*.csv"))
    files_to_process = []
    if not SESSIONS_TO_PROCESS:
        if VERBOSE: print("No specific sessions defined. Processing all 'cleaned_*.csv' files.")
        files_to_process = all_files_in_dir
    else:
        for session_num in SESSIONS_TO_PROCESS:
            session_tag = f"[test {session_num}]".lower()
            found_for_session = False
            for f_path in all_files_in_dir:
                if session_tag in os.path.basename(f_path).lower():
                    files_to_process.append(f_path); found_for_session = True
            if not found_for_session and VERBOSE: print(f"  Warning: No files for session tag '{session_tag}'.")
    if not files_to_process: print(f"No files found to process. Exiting."); exit()
    files_to_process = sorted(list(set(files_to_process)))
    if VERBOSE:
        print(f"Found {len(files_to_process)} files to process for GPR modeling:")
        for i, f in enumerate(files_to_process): print(f"  {i+1}. {os.path.basename(f)}")

    # --- Stage 1 & 2: Per-File Preprocessing & Target Transformation ---
    all_processed_active_dfs = []
    if VERBOSE: print("\n--- Stage 1 & 2: Per-File Preprocessing & Target Transformation ---")
    for file_path in files_to_process:
        df_raw_single = load_single_file_data(file_path)
        if df_raw_single is None: continue
        if POSITION_COLUMN not in df_raw_single.columns:
            print(f"  Error: Pos col '{POSITION_COLUMN}' not in {os.path.basename(file_path)}. Skip."); continue
        
        df_active_single, _ = apply_fft_baseline_normalization(
            df_raw_single, FFT_COLUMNS, POSITION_COLUMN, ACTIVITY_COLUMN, filename=os.path.basename(file_path)
        )
        if df_active_single.empty: continue
        if CURVATURE_COLUMN not in df_active_single.columns:
            print(f"  Error: Curvature col '{CURVATURE_COLUMN}' not in {os.path.basename(file_path)}. Skip."); continue

        # Always calculate Radius_mm for evaluation and for log transform if needed
        df_active_single[EVALUATION_TARGET_RADIUS_COLUMN] = convert_curvature_to_radius(
            df_active_single[CURVATURE_COLUMN], RADIUS_ZERO_HANDLING, RADIUS_CAP_VALUE, RADIUS_PLACEHOLDER_VALUE
        )
        
        # Prepare the specific target column for the GPR model
        if PREDICT_TARGET == "RADIUS":
            df_active_single[TARGET_COLUMN_FOR_MODEL] = df_active_single[EVALUATION_TARGET_RADIUS_COLUMN]
        elif PREDICT_TARGET == "CURVATURE":
            df_active_single[TARGET_COLUMN_FOR_MODEL] = df_active_single[CURVATURE_COLUMN]
        elif PREDICT_TARGET == "LOG_RADIUS":
            # Ensure radius is positive before log. convert_curvature_to_radius with 'cap' should ensure this.
            # If any radius values are <=0 here, log will produce NaN or error.
            if (df_active_single[EVALUATION_TARGET_RADIUS_COLUMN] <= 0).any():
                print(f"  Warning: Non-positive radius values found in {os.path.basename(file_path)} before log transform. This may lead to NaNs.")
            df_active_single[TARGET_COLUMN_FOR_MODEL] = np.log(df_active_single[EVALUATION_TARGET_RADIUS_COLUMN].replace(0, LOG_EPSILON)) # Apply log
            if VERBOSE: print(f"  Applied log transform to {EVALUATION_TARGET_RADIUS_COLUMN} for GPR target.")
        else:
            raise ValueError("PREDICT_TARGET must be 'RADIUS', 'CURVATURE', or 'LOG_RADIUS'")
        
        all_processed_active_dfs.append(df_active_single)

    if not all_processed_active_dfs: print("No data available after processing all files. Exiting."); exit()

    # --- Stage 3: Combining Data ---
    if VERBOSE: print("\n--- Stage 3: Combining Data ---")
    combined_df = pd.concat(all_processed_active_dfs, ignore_index=True)
    if VERBOSE: print(f"Combined data from all files. Shape: {combined_df.shape}")

    # --- Stage 4: Prepare Features (X) and Target (y) & Clean ---
    if not all(col in combined_df.columns for col in INPUT_FEATURE_COLUMNS + [TARGET_COLUMN_FOR_MODEL, EVALUATION_TARGET_RADIUS_COLUMN]):
        missing = [c for c in INPUT_FEATURE_COLUMNS + [TARGET_COLUMN_FOR_MODEL, EVALUATION_TARGET_RADIUS_COLUMN] if c not in combined_df.columns]
        print(f"Error: Missing required columns in combined_df: {missing}. Exiting."); exit()

    X_combined = combined_df[INPUT_FEATURE_COLUMNS]
    y_model_target_combined = combined_df[TARGET_COLUMN_FOR_MODEL]
    y_evaluation_radius_combined = combined_df[EVALUATION_TARGET_RADIUS_COLUMN]

    original_rows = len(combined_df)
    df_for_nan_check = pd.concat([X_combined, y_model_target_combined, y_evaluation_radius_combined], axis=1)
    # Check for inf values as well, which can arise from log(0) if not handled or extreme feature values
    df_for_nan_check.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_final_processed = df_for_nan_check.dropna()
    
    if VERBOSE and len(df_final_processed) < original_rows:
        print(f"  Dropped {original_rows - len(df_final_processed)} rows due to NaN/Inf values in features or targets.")
    if df_final_processed.empty: print("No data remaining after NaN/Inf drop. Exiting."); exit()
    
    X = df_final_processed[INPUT_FEATURE_COLUMNS]
    y_model_target = df_final_processed[TARGET_COLUMN_FOR_MODEL]
    y_evaluation_radius_true = df_final_processed[EVALUATION_TARGET_RADIUS_COLUMN]

    # --- Stage 5: Train-Test Split ---
    if VERBOSE: print("\n--- Stage 5: Train-Test Split ---")
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_model_target, y_test_model_target_true_unscaled = y_model_target.iloc[train_idx], y_model_target.iloc[test_idx]
    y_test_evaluation_radius_true = y_evaluation_radius_true.iloc[test_idx]
    if VERBOSE: print(f"Data split: X_train shape {X_train.shape}, X_test shape {X_test.shape}")

    # --- Stage 6: Scaling Features and Target ---
    if VERBOSE: print("\n--- Stage 6: Scaling Features and Target ---")
    X_train_scaled, x_scalers = scale_data(X_train, INPUT_FEATURE_COLUMNS)
    X_test_scaled, _ = scale_data(X_test, INPUT_FEATURE_COLUMNS, existing_scalers=x_scalers)
    y_train_model_target_df = pd.DataFrame(y_train_model_target, columns=[TARGET_COLUMN_FOR_MODEL])
    y_train_scaled_df, y_target_scalers = scale_data(y_train_model_target_df, [TARGET_COLUMN_FOR_MODEL])
    y_train_scaled_model_target = y_train_scaled_df[TARGET_COLUMN_FOR_MODEL]

    # --- Stage 7: Training GPR Model ---
    if VERBOSE: print("\n--- Stage 7: Training GPR Model ---")
    print(f"  Predicting GPR Target: {PREDICT_TARGET} (Column: {TARGET_COLUMN_FOR_MODEL})")
    print(f"  Using Kernel: {GPR_KERNEL}")
    gpr = GaussianProcessRegressor(kernel=GPR_KERNEL, alpha=GPR_ALPHA, n_restarts_optimizer=GPR_N_RESTARTS_OPTIMIZER, random_state=RANDOM_STATE)
    gpr.fit(X_train_scaled, y_train_scaled_model_target)
    if VERBOSE:
        print("GPR model training complete.")
        print(f"  Optimized Kernel: {gpr.kernel_}")
        print(f"  Log-Marginal-Likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}")

    # --- Stage 8: Making Predictions ---
    if VERBOSE: print("\n--- Stage 8: Making Predictions ---")
    y_pred_scaled_model_target, _ = gpr.predict(X_test_scaled, return_std=True)

    # --- Stage 9: Descaling Predicted Target ---
    if VERBOSE: print("\n--- Stage 9: Descaling Predicted GPR Target ---")
    y_pred_scaled_df = pd.DataFrame(y_pred_scaled_model_target, columns=[TARGET_COLUMN_FOR_MODEL], index=X_test.index)
    y_pred_descaled_model_target_df = descale_data(y_pred_scaled_df, [TARGET_COLUMN_FOR_MODEL], y_target_scalers)
    y_pred_descaled_model_target = y_pred_descaled_model_target_df[TARGET_COLUMN_FOR_MODEL]

    # --- Stage 10: Inverse Transform Predicted Target to Radius (if necessary) for Evaluation ---
    y_pred_evaluation_radius = None
    true_target_for_plot = None
    predicted_target_for_plot = None
    plot_target_name = PREDICT_TARGET.title()
    plot_target_unit = "units"

    if PREDICT_TARGET == "RADIUS":
        y_pred_evaluation_radius = y_pred_descaled_model_target
        true_target_for_plot = y_test_evaluation_radius_true
        predicted_target_for_plot = y_pred_evaluation_radius
        plot_target_unit = EVALUATION_TARGET_RADIUS_COLUMN.split('_')[-1] if '_' in EVALUATION_TARGET_RADIUS_COLUMN else 'mm'
    elif PREDICT_TARGET == "CURVATURE":
        if VERBOSE: print("\n--- Converting Predicted Curvature to Predicted Radius for Evaluation ---")
        pred_radius_temp = pd.Series(index=y_pred_descaled_model_target.index, dtype=float)
        near_zero_pred_curv_mask = np.isclose(y_pred_descaled_model_target, 0)
        if (~near_zero_pred_curv_mask).any():
            pred_radius_temp.loc[~near_zero_pred_curv_mask] = 1.0 / y_pred_descaled_model_target[~near_zero_pred_curv_mask]
        if near_zero_pred_curv_mask.any():
            pred_radius_temp.loc[near_zero_pred_curv_mask] = RADIUS_CAP_VALUE
        y_pred_evaluation_radius = pred_radius_temp
        true_target_for_plot = y_test_model_target_true_unscaled # True Curvature
        predicted_target_for_plot = y_pred_descaled_model_target # Predicted Curvature
        plot_target_name = "Curvature"
        plot_target_unit = CURVATURE_COLUMN.split('_')[-1] if '_' in CURVATURE_COLUMN else ('mm^-1' if 'mm' in EVALUATION_TARGET_RADIUS_COLUMN else 'm^-1')
    elif PREDICT_TARGET == "LOG_RADIUS":
        if VERBOSE: print("\n--- Converting Predicted Log-Radius to Predicted Radius for Evaluation ---")
        y_pred_evaluation_radius = np.exp(y_pred_descaled_model_target) # Inverse of np.log()
        true_target_for_plot = y_test_model_target_true_unscaled # True Log-Radius
        predicted_target_for_plot = y_pred_descaled_model_target # Predicted Log-Radius
        plot_target_name = "Log Radius"
        plot_target_unit = f"log({EVALUATION_TARGET_RADIUS_COLUMN.split('_')[-1]})" if '_' in EVALUATION_TARGET_RADIUS_COLUMN else 'log(units)'

    # --- Stage 11: Evaluating Model Performance (on EVALUATION_TARGET_RADIUS_COLUMN) ---
    if VERBOSE: print(f"\n--- Stage 11: Evaluating Model Performance (on {EVALUATION_TARGET_RADIUS_COLUMN}) ---")
    mae = mean_absolute_error(y_test_evaluation_radius_true, y_pred_evaluation_radius)
    rmse = np.sqrt(mean_squared_error(y_test_evaluation_radius_true, y_pred_evaluation_radius))
    r2 = r2_score(y_test_evaluation_radius_true, y_pred_evaluation_radius)
    radius_unit_eval = EVALUATION_TARGET_RADIUS_COLUMN.split('_')[-1] if '_' in EVALUATION_TARGET_RADIUS_COLUMN else 'mm'
    print(f"  Mean Absolute Error (MAE) on Radius:      {mae:.4f} {radius_unit_eval}")
    print(f"  Root Mean Squared Error (RMSE) on Radius: {rmse:.4f} {radius_unit_eval}")
    print(f"  R-squared (R2 Score) on Radius:           {r2:.4f}")

    # --- Plotting and JSON Output ---
    base_filename_prefix = f"gpr_rq_pos_{PREDICT_TARGET.lower()}_80_20"
    if SESSIONS_TO_PROCESS:
        session_str_for_file = "_".join(SESSIONS_TO_PROCESS)
        final_filename_prefix = f"{base_filename_prefix}_sessions_{session_str_for_file}"
    else:
        final_filename_prefix = f"{base_filename_prefix}_all_sessions"

    generate_true_vs_predicted_plot(
        true_target_for_plot, predicted_target_for_plot,
        model_output_full_path, filename_prefix=final_filename_prefix,
        target_name_for_plot=plot_target_name, target_unit=plot_target_unit
    )
    
    # --- Detailed Error Analysis (on Radius) ---
    if VERBOSE: print("\n--- Stage 12: Detailed Error Analysis (on Radius Predictions) ---")
    residuals_radius = y_test_evaluation_radius_true - y_pred_evaluation_radius
    min_true_rad = y_test_evaluation_radius_true.min()
    max_true_rad = y_test_evaluation_radius_true.max()
    if max_true_rad > min_true_rad :
        num_bins = 5
        if max_true_rad < RADIUS_CAP_VALUE * 0.9 :
             bins = np.linspace(min_true_rad, max_true_rad + 1e-9, num_bins + 1)
        else:
            pre_cap_max = y_test_evaluation_radius_true[y_test_evaluation_radius_true < RADIUS_CAP_VALUE * 0.9].max()
            if pd.isna(pre_cap_max) or pre_cap_max <= min_true_rad : pre_cap_max = min_true_rad + ( (max_true_rad if max_true_rad < RADIUS_CAP_VALUE * 0.9 else 200) - min_true_rad) * (num_bins-1)/num_bins # Fallback
            bins_up_to_pre_cap = np.linspace(min_true_rad, pre_cap_max, num_bins)
            bins = np.append(bins_up_to_pre_cap, [max_true_rad + 1e-9]) # Use actual max_true_rad for last bin edge
            bins = np.unique(bins)
    else: bins = [min_true_rad, max_true_rad + 1e-9]
    segmented_error_df = calculate_segmented_errors(
        y_test_evaluation_radius_true, y_pred_evaluation_radius, segment_bins=bins,
        target_name="Radius", target_unit=radius_unit_eval
    )
    print("\nSegmented Error Analysis (Radius):\n", segmented_error_df)

    if GENERATE_DETAILED_ERROR_ANALYSIS_PLOTS:
        if VERBOSE: print("\n  Generating Residual Plots (on Radius)...")
        plot_residuals(residuals_radius, y_test_evaluation_radius_true, 
                       f"True Radius ({radius_unit_eval})", "Radius",
                       "Residuals vs. True Radius", model_output_full_path, 
                       f"{final_filename_prefix}_residuals_vs_true_radius.png")
        plot_residuals(residuals_radius, y_pred_evaluation_radius, 
                       f"Predicted Radius ({radius_unit_eval})", "Radius",
                       "Residuals vs. Predicted Radius", model_output_full_path, 
                       f"{final_filename_prefix}_residuals_vs_predicted_radius.png")
        for feature_col in INPUT_FEATURE_COLUMNS:
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

    # --- JSON Output ---
    output_summary = {
        "model_type": "GaussianProcessRegressor", "kernel_type": "RationalQuadratic",
        "predict_target_variable_for_gpr": PREDICT_TARGET,
        "evaluation_always_on_radius": EVALUATION_TARGET_RADIUS_COLUMN,
        "input_features_used": INPUT_FEATURE_COLUMNS,
        "script_variant": f"80_20_pos_target_{PREDICT_TARGET.lower()}",
        "sessions_processed": SESSIONS_TO_PROCESS if SESSIONS_TO_PROCESS else "all",
        "input_data_directory": input_data_dir_path, "output_directory": model_output_full_path,
        "radius_zero_handling_for_source_data": RADIUS_ZERO_HANDLING,
        "gpr_kernel_initial_config": str(GPR_KERNEL), "gpr_kernel_optimized_config": str(gpr.kernel_),
        "gpr_alpha_value_added_to_diagonal": GPR_ALPHA, "gpr_n_restarts_optimizer": GPR_N_RESTARTS_OPTIMIZER,
        "test_size_split_ratio": TEST_SIZE, "random_state_seed": RANDOM_STATE,
        "performance_metrics_on_radius": {
            "mae": mae, "rmse": rmse, "r2_score": r2, "units": radius_unit_eval
        },
        "data_summary": {
            "total_files_processed": len(files_to_process),
            "combined_data_shape_before_nan_drop": combined_df.shape if 'combined_df' in locals() else "N/A",
            "combined_data_shape_after_nan_drop": df_final_processed.shape if 'df_final_processed' in locals() else "N/A",
            "training_set_shape": X_train.shape if 'X_train' in locals() else "N/A",
            "test_set_shape": X_test.shape if 'X_test' in locals() else "N/A"
        },
        "segmented_radius_errors": segmented_error_df.to_dict(orient='records') if 'segmented_error_df' in locals() else "N/A"
    }
    json_filename = f"{final_filename_prefix}_metrics_config.json"
    json_output_path = os.path.join(model_output_full_path, json_filename)
    try:
        with open(json_output_path, 'w') as f: json.dump(output_summary, f, indent=4, sort_keys=True)
        if VERBOSE: print(f"\nMetrics and configuration saved to: {json_output_path}")
    except Exception as e: print(f"Error saving metrics to JSON: {e}")

    # --- Stage 13: Example Output Table ---
    if VERBOSE: print(f"\n--- Stage 13: Example Output Table ---")
    example_table_df = pd.DataFrame(index=y_test_evaluation_radius_true.index)
    example_table_df[f'True_{EVALUATION_TARGET_RADIUS_COLUMN}'] = y_test_evaluation_radius_true
    example_table_df[f'Predicted_{EVALUATION_TARGET_RADIUS_COLUMN}'] = y_pred_evaluation_radius
    
    if PREDICT_TARGET == "CURVATURE":
        example_table_df['Predicted_Curvature_from_Model'] = y_pred_descaled_model_target
        example_table_df['True_Curvature'] = y_test_model_target_true_unscaled
    elif PREDICT_TARGET == "LOG_RADIUS":
        example_table_df['Predicted_LogRadius_from_Model'] = y_pred_descaled_model_target
        example_table_df['True_LogRadius'] = y_test_model_target_true_unscaled
        # Also show calculated curvature from predicted radius for completeness
        pred_curv_from_rad = np.full_like(y_pred_evaluation_radius.values, np.nan)
        straight_mask = np.isclose(y_pred_evaluation_radius.values, RADIUS_CAP_VALUE)
        pred_curv_from_rad[straight_mask] = 0.0
        bend_mask = (~straight_mask) & (~np.isclose(y_pred_evaluation_radius.values, 0))
        if np.any(bend_mask): pred_curv_from_rad[bend_mask] = 1.0 / y_pred_evaluation_radius.values[bend_mask]
        example_table_df['Calculated_Curvature_from_PredRadius'] = pred_curv_from_rad
        example_table_df['True_Curvature'] = combined_df.loc[y_test_evaluation_radius_true.index, CURVATURE_COLUMN]
    else: # PREDICT_TARGET == "RADIUS"
        pred_curv_from_rad = np.full_like(y_pred_evaluation_radius.values, np.nan)
        straight_mask = np.isclose(y_pred_evaluation_radius.values, RADIUS_CAP_VALUE)
        pred_curv_from_rad[straight_mask] = 0.0
        bend_mask = (~straight_mask) & (~np.isclose(y_pred_evaluation_radius.values, 0))
        if np.any(bend_mask): pred_curv_from_rad[bend_mask] = 1.0 / y_pred_evaluation_radius.values[bend_mask]
        example_table_df['Predicted_Curvature_from_Radius'] = pred_curv_from_rad
        example_table_df['True_Curvature'] = combined_df.loc[y_test_evaluation_radius_true.index, CURVATURE_COLUMN]
        
    print(f"  Example Results (first 5 test samples, GPR predicted {PREDICT_TARGET}):")
    print(example_table_df.head())

    print("\nScript finished successfully.")
