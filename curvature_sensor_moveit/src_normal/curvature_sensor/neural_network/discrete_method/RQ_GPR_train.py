"""
This script trains a Gaussian Process Regressor (GPR) model for a specific test object
to predict 'Curvature' based on FFT and 'Position_cm' features.

The script performs the following key operations:
1.  Data Loading and Preprocessing:
    -   Identifies and loads pre-cleaned CSV files (e.g., from a 'cleaned' directory,
        processed by a separate script like 'noise_remover.py') corresponding to a
        specified `TARGET_TEST_OBJECT_NUM`.
    -   For each file:
        -   Calculates an 'idle FFT baseline' using data points where 'Position_cm' is 0.0
            and 'Curvature_Active' is 0, specifically from rows *before* the first
            instance of 'Position_cm' being 0.0. This baseline represents the sensor's
            FFT readings in a neutral, inactive state from the pre-cleaned file.
        -   Filters data to retain only 'active' segments ('Curvature_Active' == 1)
            from the pre-cleaned file.
        -   Normalizes FFT features by subtracting the calculated idle baseline.
        -   Parses a 'group ID' (typically the curvature value) from the filename,
            which is used for Leave-One-Group-Out cross-validation.
    -   Combines all processed and normalized data segments from relevant files.

2.  Cross-Validation (Leave-One-Curvature-Out):
    -   If sufficient unique curvature groups exist (at least 2):
        -   Splits the data such that each fold uses one curvature group as the test set
            and the remaining groups as the training set.
        -   For each fold:
            -   Applies StandardScaler to the feature data.
            -   Defines a GPR model with a RationalQuadratic kernel combined with
                a WhiteKernel (for noise) and a ConstantKernel (for signal variance).
            -   Trains the GPR model.
            -   Evaluates performance using Mean Squared Error (MSE), Mean Absolute
                Error (MAE), and R-squared (R²) score.
            -   Collects detailed predictions (true vs. predicted, error, std dev).
        -   Saves aggregated CV metrics and detailed per-fold predictions to CSV files.

3.  Final Model Training:
    -   Trains a final GPR model using all available processed data for the
        `TARGET_TEST_OBJECT_NUM`.
    -   The features are scaled using StandardScaler.
    -   The trained GPR model and the scaler object are saved using `joblib`.

4.  Summary Report Generation:
    -   Generates a comprehensive Markdown (.md) report detailing:
        -   Data processing steps (e.g., number of files processed, baseline
            calculation success, files skipped with reasons).
        -   Cross-validation performance (average metrics and per-fold results table).
        -   Final model details (learned kernel parameters, paths to saved model/scaler).
        -   A list of all generated output files (CSV results, model files, report itself).

The script is driven by an `if __name__ == "__main__":` block, which allows setting
the `TARGET_TEST_OBJECT_NUM`, input data directory, and output model directory.
It includes error handling for missing input directories or files and will generate
a minimal report in such cases.
"""
import pandas as pd
import numpy as np
import os
import glob
import re
import time
from datetime import datetime # Added for report timestamp
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Added mean_absolute_error
import joblib

# --- Global Configuration ---
# FFT_COLUMNS: Specifies the Fast Fourier Transform frequency bins used as features.
FFT_COLUMNS = ['FFT_200Hz', 'FFT_400Hz', 'FFT_600Hz', 'FFT_800Hz',
               'FFT_1000Hz', 'FFT_1200Hz', 'FFT_1400Hz', 'FFT_1600Hz',
               'FFT_1800Hz', 'FFT_2000Hz']
POSITION_COLUMN = 'Position_cm' # Column name for the position feature.
TARGET_COLUMN = 'Curvature'     # Column name for the target variable to predict.
ACTIVITY_COLUMN = 'Curvature_Active' # Column indicating if the sensor is in an active state.
TIMESTAMP_COLUMN = 'Timestamp'  # Column for timestamps, used for temporal processing.
# FEATURE_COLUMNS: Defines the complete list of input features for the model.
FEATURE_COLUMNS = FFT_COLUMNS + [POSITION_COLUMN]

# --- Function: Parse Filename for Group (Curvature Value for a specific test object) ---
def parse_filename_for_group_curvature(filename, target_test_object_num_str):
    base_name = os.path.basename(filename)
    # Dynamically create the regex pattern for the target test object
    # Example: if target_test_object_num_str is "1", pattern looks for "[test 1]"
    pattern_str = rf".*?_([0-9._]+?)\s*\[test\s*{re.escape(target_test_object_num_str)}\]"
    match = re.match(pattern_str, base_name, re.IGNORECASE)
    if match:
        return match.group(1).replace('_', '.') 
    
    # Fallback for filenames starting with "cleaned_"
    pattern_cleaned_str = rf"cleaned_([0-9._]+?)\s*\[test\s*{re.escape(target_test_object_num_str)}\]"
    match_cleaned = re.match(pattern_cleaned_str, base_name, re.IGNORECASE)
    if match_cleaned:
        return match_cleaned.group(1).replace('_','.')

    print(f"    Warning: Could not parse curvature group from filename: {base_name} for [test {target_test_object_num_str}] pattern. Using full filename stem as group.")
    return os.path.splitext(base_name)[0] 

# --- Main GPR Training and Evaluation Function ---
def train_gpr_for_selected_object(file_paths, target_test_object_num_str, output_model_dir_for_test_obj): # Renamed and added param
    all_gpr_ready_segments = []
    successful_baseline_count = 0
    files_without_baseline = []
    generated_output_files = [] # To store paths of successfully created files for the report

    print(f"Starting GPR data preparation for {len(file_paths)} files for test object '[test {target_test_object_num_str}]' (expecting pre-cleaned files from 'cleaned' directory)...")

    for file_path in file_paths: 
        base_filename = os.path.basename(file_path)
        print(f"\n  Processing file: {base_filename}")
        try:
            # This df_cleaned_input is read from your 'csv_data/cleaned/' directory
            # It's assumed to have been ALREADY processed by your noise_remover.py
            df_cleaned_input = pd.read_csv(file_path) 
            if df_cleaned_input.empty:
                print(f"    Skipping empty pre-cleaned file: {base_filename}")
                files_without_baseline.append(f"{base_filename} (File was empty)")
                continue
        except Exception as e:
            print(f"    Error reading pre-cleaned file {base_filename}: {e}. Skipping.")
            files_without_baseline.append(f"{base_filename} (Read error: {e})")
            continue

        # 1. Calculate Idle Baseline (from the already "cleaned" df_cleaned_input)
        #    The success of this step now depends on noise_remover.py NOT removing
        #    the critical initial idle rows needed for this baseline logic.
        idle_fft_baseline = None
        if not all(col in df_cleaned_input.columns for col in [POSITION_COLUMN, ACTIVITY_COLUMN] + FFT_COLUMNS):
            print(f"    Skipping {base_filename}: missing one or more required columns for baseline calculation.")
            files_without_baseline.append(f"{base_filename} (Missing required columns for baseline)")
            continue
            
        first_pos_zero_indices = df_cleaned_input[df_cleaned_input[POSITION_COLUMN] == 0.0].index
        
        if not first_pos_zero_indices.empty:
            first_pos_zero_idx = first_pos_zero_indices[0]
            idle_candidate_rows = df_cleaned_input[
                (df_cleaned_input.index < first_pos_zero_idx) & 
                (df_cleaned_input[ACTIVITY_COLUMN] == 0)
            ]
            if not idle_candidate_rows.empty:
                idle_fft_baseline = idle_candidate_rows[FFT_COLUMNS].mean().values 
                print(f"    Calculated idle baseline from {len(idle_candidate_rows)} rows (from pre-cleaned file) for {base_filename}.")
                successful_baseline_count += 1
            else:
                print(f"    WARNING: In pre-cleaned file '{base_filename}', no 'Curvature_Active == 0' rows found before first 'Position_cm == 0.0'.")
                files_without_baseline.append(f"{base_filename} (No CA=0 before first PosCM=0 in cleaned file)")
        else:
            print(f"    WARNING: In pre-cleaned file '{base_filename}', no 'Position_cm == 0.0' found.")
            files_without_baseline.append(f"{base_filename} (No PosCM=0 found in cleaned file)")

        # <<<< THE CALL TO process_data_for_discrete_extraction IS REMOVED >>>>
        # We now directly use df_cleaned_input as the starting point for GPR data prep.
        # print(f"    Using pre-cleaned df shape: {df_cleaned_input.shape}") # Already implies this

        # 3. Filter for Active Data from df_cleaned_input
        active_df = df_cleaned_input[df_cleaned_input[ACTIVITY_COLUMN] == 1].copy()
        if active_df.empty:
            print(f"    No active data in pre-cleaned file {base_filename} after filtering. Skipping.")
            continue
        print(f"    Found {len(active_df)} active rows in pre-cleaned file.")


        # 4. Normalize FFT Features of Active Data (Subtraction)
        if idle_fft_baseline is not None and not active_df.empty:
            active_df.loc[:, FFT_COLUMNS] = active_df[FFT_COLUMNS].values - idle_fft_baseline
            print(f"    Normalized {len(active_df)} active rows using baseline subtraction.")
        elif not active_df.empty: # Only print this warning if there was active data to begin with
            print(f"    Active data for {base_filename} not baseline-subtracted (no suitable idle baseline was found from pre-cleaned file).")
        
        if active_df.empty: # Check again in case all rows were somehow filtered out
            continue 
            
        group_id = parse_filename_for_group_curvature(base_filename, target_test_object_num_str) # Pass param
        active_df['group'] = group_id
        
        all_gpr_ready_segments.append(active_df)

    print(f"\nNormalization Baseline Calculation Summary for {len(file_paths)} files:")
    print(f"  Successfully calculated idle baseline for {successful_baseline_count} files.")
    if files_without_baseline:
        print(f"  Files where specific idle baseline conditions were not met or other issues occurred:")
        for f_issue in files_without_baseline:
            print(f"    - {f_issue}")

    results_df = None # Initialize results_df for CV summary
    final_gpr_kernel_str = "N/A (Final model not trained or training skipped)"
    model_save_path_str = "N/A"
    scaler_save_path_str = "N/A"

    if not all_gpr_ready_segments:
        print("No data available for GPR training after processing all selected files.")
        # Call report generation even if no data, it will show processing summary
        _generate_summary_report(target_test_object_num_str, output_model_dir_for_test_obj, file_paths,
                                 successful_baseline_count, files_without_baseline, results_df, 
                                 final_gpr_kernel_str, model_save_path_str, scaler_save_path_str,
                                 generated_output_files)
        return

    combined_df = pd.concat(all_gpr_ready_segments, ignore_index=True)
    if combined_df.empty:
        print("Combined DataFrame is empty. Cannot proceed with GPR.")
        _generate_summary_report(target_test_object_num_str, output_model_dir_for_test_obj, file_paths,
                                 successful_baseline_count, files_without_baseline, results_df, 
                                 final_gpr_kernel_str, model_save_path_str, scaler_save_path_str,
                                 generated_output_files)
        return
        
    print(f"\nCombined preprocessed data shape for GPR: {combined_df.shape}")

    X = combined_df[FEATURE_COLUMNS]
    y = combined_df[TARGET_COLUMN]
    groups = combined_df['group']
    unique_group_ids = groups.unique()

    if len(unique_group_ids) < 2:
        print(f"Only {len(unique_group_ids)} unique group(s) found: {unique_group_ids}. LeaveOneGroupOut CV requires at least 2 groups. CV will be skipped.")
    else:
        logo = LeaveOneGroupOut()
        fold_metrics = []
        all_fold_predictions_data = [] 
        fold_num = 0

        print(f"\nStarting Leave-One-Curvature-Out Cross-Validation for test object '[test {target_test_object_num_str}]' using {len(unique_group_ids)} groups: {sorted(list(unique_group_ids))}...")

        for train_idx, test_idx in logo.split(X, y, groups):
            fold_num += 1
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            current_test_group = groups.iloc[test_idx].unique()[0]
            print(f"\n--- Fold {fold_num}: Testing on Curvature Group '{current_test_group}' (from [test {target_test_object_num_str}]) ---")
            print(f"  Train set size: {len(X_train)}, Test set size: {len(X_test)}")

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-2, 1e2), alpha_bounds=(1e-2, 1e2)) \
                     + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
            
            gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, random_state=42, normalize_y=True)

            print(f"  Fitting GPR for fold {fold_num}...")
            start_time_fold = time.time()
            gpr.fit(X_train_scaled, y_train)
            fit_time_fold = time.time() - start_time_fold
            print(f"  GPR fitting completed in {fit_time_fold:.2f}s. Learned kernel: {gpr.kernel_}")

            y_pred, y_std = gpr.predict(X_test_scaled, return_std=True)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred) # Calculate MAE
            r2 = r2_score(y_test, y_pred)
            
            fold_metrics.append({
                'fold': fold_num, 
                'tested_group': current_test_group, 
                'mse': mse, 
                'mae': mae, # Add MAE to metrics
                'r2': r2, 
                'fitting_time_s': fit_time_fold,
                'learned_kernel': str(gpr.kernel_)
            })
            print(f"  Fold {fold_num} Results: MSE = {mse:.6f}, MAE = {mae:.6f}, R2 = {r2:.4f}") # Print MAE

            fold_test_df = pd.DataFrame({
                'true_curvature': y_test.values, 'predicted_curvature': y_pred,
                'prediction_std_dev': y_std, 'error': y_test.values - y_pred,
                'abs_error': np.abs(y_test.values - y_pred), 'fold_number': fold_num,
                'tested_group_for_row': current_test_group 
            })
            fold_test_df[POSITION_COLUMN] = X_test[POSITION_COLUMN].values 
            all_fold_predictions_data.append(fold_test_df)

        print("\n--- Cross-Validation Finished ---")
        if fold_metrics:
            results_df = pd.DataFrame(fold_metrics) # Assign to the broader scope variable
            print(f"\nCross-Validation Results Summary (Leave-One-Curvature-Out for [test {target_test_object_num_str}] data):")
            # Ensure MAE is in the printed columns list
            columns_to_print = ['fold', 'tested_group', 'mse', 'mae', 'r2', 'fitting_time_s', 'learned_kernel']
            with pd.option_context('display.max_colwidth', None, 'display.width', 1000):
                 print(results_df[columns_to_print])
            print(f"\nAverage CV MSE: {results_df['mse'].mean():.6f}")
            print(f"Average CV MAE: {results_df['mae'].mean():.6f}") # Print Average MAE
            print(f"Average CV R2 Score: {results_df['r2'].mean():.4f}")
            
            output_csv_path = os.path.join(output_model_dir_for_test_obj, f"gpr_test{target_test_object_num_str}_LOCO_curvature_cv_results.csv")
            try:
                results_df.to_csv(output_csv_path, index=False)
                print(f"CV results saved to {output_csv_path}")
                generated_output_files.append(output_csv_path) # Add on successful save
            except Exception as e:
                print(f"Error saving CV results to {output_csv_path}: {e}")

        if all_fold_predictions_data:
            detailed_predictions_df = pd.concat(all_fold_predictions_data, ignore_index=True)
            detailed_output_csv_path = os.path.join(output_model_dir_for_test_obj, f"gpr_test{target_test_object_num_str}_detailed_predictions.csv")
            try:
                detailed_predictions_df.to_csv(detailed_output_csv_path, index=False)
                print(f"Detailed predictions and errors saved to {detailed_output_csv_path}")
                generated_output_files.append(detailed_output_csv_path) # Track successfully saved file
            except Exception as e:
                print(f"Error saving detailed predictions CSV: {e}")

    # --- Final Model Training ---
    # This section trains a GPR model on the entirety of the processed data for the current test object.
    # This model is intended for deployment or final evaluation, distinct from the CV models.
    print(f"\n--- Training Final Model on All Processed [test {target_test_object_num_str}] Data ---")
    if not X.empty and not y.empty: # Ensure there's data to train on (X and y are from combined_df).
        final_scaler = StandardScaler()
        X_scaled_final = final_scaler.fit_transform(X) # Scale all features.

        # Define the kernel for the final GPR model, same structure as CV.
        final_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=1.0, alpha=1.0, length_scale_bounds=(1e-2, 1e2), alpha_bounds=(1e-2, 1e2)) \
                       + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
        final_gpr = GaussianProcessRegressor(kernel=final_kernel, n_restarts_optimizer=9, random_state=42, normalize_y=True)

        print(f"Fitting final GPR model on all selected [test {target_test_object_num_str}] data (Size: {X_scaled_final.shape[0]})...")
        start_time_final = time.time()
        final_gpr.fit(X_scaled_final, y)
        fit_time_final = time.time() - start_time_final
        final_gpr_kernel_str = str(final_gpr.kernel_) # Store the learned kernel parameters as a string.
        print(f"Final GPR model fitted in {fit_time_final:.2f}s. Learned kernel: {final_gpr_kernel_str}")

        # Define paths for saving the trained model and the scaler.
        model_save_path = os.path.join(output_model_dir_for_test_obj, f"final_gpr_model_test{target_test_object_num_str}_data.joblib")
        scaler_save_path = os.path.join(output_model_dir_for_test_obj, f"final_scaler_test{target_test_object_num_str}_data.joblib")

        try:
            joblib.dump(final_gpr, model_save_path)
            joblib.dump(final_scaler, scaler_save_path)
            print(f"Final GPR model saved to: {model_save_path}")
            print(f"Final Scaler saved to: {scaler_save_path}")
            model_save_path_str = model_save_path # Update with actual path
            scaler_save_path_str = scaler_save_path # Update with actual path
            generated_output_files.append(model_save_path) # Add on successful save
            generated_output_files.append(scaler_save_path) # Add on successful save
        except Exception as e:
            print(f"Error saving final model/scaler: {e}")
    else:
        print("Skipping final model training as combined data is empty or X/y are not available.")

    # --- Generate Summary Report ---
    _generate_summary_report(
        target_test_object_num_str,
        output_model_dir_for_test_obj,
        file_paths, # Original list of files intended for processing
        successful_baseline_count,
        files_without_baseline,
        results_df, # CV results DataFrame, could be None
        final_gpr_kernel_str,
        model_save_path_str,
        scaler_save_path_str,
        generated_output_files # List of files actually created
    )

def _generate_summary_report(target_test_object_num_str, output_dir, input_file_paths,
                             successful_baseline_count, files_without_baseline,
                             cv_results_df, final_model_kernel_str, 
                             final_model_saved_path, final_scaler_saved_path,
                             generated_files_list_so_far):
    """Generates a Markdown summary report."""
    # ----------------------------------------------------------------------------------
    # Section: Initialization and Report Header
    # ----------------------------------------------------------------------------------
    report_filename = f"gpr_test{target_test_object_num_str}_summary_report.md"
    report_full_path = os.path.join(output_dir, report_filename)

    # Prepare the list of all files to be mentioned in the report (including the report itself)
    all_output_files_for_report = list(generated_files_list_so_far) # My note: Copy the list of already saved files
    all_output_files_for_report.append(report_full_path) # My note: Add the report's own path

    report_content = []
    report_content.append(f"# GPR Model Training Summary: Test Object `[test {target_test_object_num_str}]`")
    report_content.append(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ----------------------------------------------------------------------------------
    # Section: Data Processing Overview
    # ----------------------------------------------------------------------------------
    report_content.append("\n## 1. Data Processing Overview")
    report_content.append(f"- **Target Test Object Identifier:** `[test {target_test_object_num_str}]`")
    report_content.append(f"- **Number of Input CSV Files Scanned:** {len(input_file_paths)}")
    
    # ----------------------------------------------------------------------------------
    # Sub-Section: Normalization Baseline Calculation Details
    # ----------------------------------------------------------------------------------
    report_content.append("\n### Normalization Baseline Calculation:")
    report_content.append(f"- Successfully calculated idle FFT baseline for **{successful_baseline_count}** out of {len(input_file_paths)} scanned files.")
    if files_without_baseline:
        report_content.append("- Files with issues or where baseline conditions were not met:")
        for f_issue in files_without_baseline:
            report_content.append(f"  - `{f_issue}`")
    elif input_file_paths: # My note: Only print this if there were files to process
        report_content.append("- All scanned files had suitable baseline data or were processed accordingly (or skipped if empty/unreadable).")
    else:
        report_content.append("- No input files were processed for baseline calculation.")

    # ----------------------------------------------------------------------------------
    # Section: Cross-Validation Performance
    # ----------------------------------------------------------------------------------
    report_content.append("\n## 2. Cross-Validation Performance (Leave-One-Curvature-Out)")
    if cv_results_df is not None and not cv_results_df.empty:
        report_content.append(f"- **Number of CV Folds (Unique Curvature Groups):** {cv_results_df.shape[0]}")
        report_content.append(f"- **Average CV MSE:** {cv_results_df['mse'].mean():.6f}")
        report_content.append(f"- **Average CV MAE:** {cv_results_df['mae'].mean():.6f}") # My note: Added MAE
        report_content.append(f"- **Average CV R² Score:** {cv_results_df['r2'].mean():.4f}")
        report_content.append("\n### Per-Fold Metrics:")
        # My note: Ensure 'mae' is in the columns for the table
        header = "| Fold | Tested Group | MSE      | MAE      | R²     | Fitting Time (s) | Learned Kernel |"
        separator = "|------|--------------|----------|----------|--------|------------------|----------------|"
        report_content.append(header)
        report_content.append(separator)
        for _, row in cv_results_df.iterrows():
            report_content.append(f"| {row['fold']} | {row['tested_group']} | {row['mse']:.6f} | {row['mae']:.6f} | {row['r2']:.4f} | {row['fitting_time_s']:.2f} | `{row['learned_kernel']}` |")
    else:
        report_content.append("- Cross-validation was not performed or did not yield results.")
        report_content.append("  (This could be due to: insufficient unique groups for LOCO CV, no data remaining after preprocessing, or other data issues).")

    # ----------------------------------------------------------------------------------
    # Section: Final Model Information
    # ----------------------------------------------------------------------------------
    report_content.append("\n## 3. Final Model Information")
    if final_model_saved_path != "N/A":
        report_content.append(f"- **Learned Kernel Parameters (Final Model):** `{final_model_kernel_str}`")
        report_content.append(f"- **Final GPR Model Saved To:** `{final_model_saved_path}`")
        report_content.append(f"- **Final Scaler Saved To:** `{final_scaler_saved_path}`")
    else:
        report_content.append("- Final model was not trained or saved (e.g., combined data was empty, or training was skipped).")

    # ----------------------------------------------------------------------------------
    # Section: Generated Output Files
    # ----------------------------------------------------------------------------------
    report_content.append("\n## 4. Generated Output Files")
    if all_output_files_for_report:
        report_content.append(f"The following files were generated in or for the directory: `{output_dir}`")
        for f_path in sorted(list(set(all_output_files_for_report))): # My note: Use set for unique paths
            # My note: Check if f_path is absolute, if not, make it relative to output_dir for clarity if needed
            # However, os.path.basename and os.path.dirname work on full paths correctly.
            report_content.append(f"- `{os.path.basename(f_path)}`")
    else:
        report_content.append("- No output files were explicitly tracked or generated.")
        
    # ----------------------------------------------------------------------------------
    # Section: Saving the Report
    # ----------------------------------------------------------------------------------
    try:
        with open(report_full_path, 'w') as f:
            f.write("\n".join(report_content))
        print(f"\nSummary report saved to: {report_full_path}")
        # My note: Optionally, add report_full_path to a master list in the caller if it needs to be tracked globally
        # For the purpose of this function, generated_files_list_so_far was for files *before* this report.
    except Exception as e:
        print(f"Error saving summary report to {report_full_path}: {e}")


if __name__ == "__main__":
    # === OPERATOR: SET THE TARGET TEST OBJECT NUMBER HERE ===
    TARGET_TEST_OBJECT_NUM = "1" # Change to "2", "3", "4", etc. as needed
    # ========================================================

    script_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_DATA_DIR = os.path.join(script_dir, "csv_data/cleaned/")
    
    OUTPUT_MODEL_BASE_DIR = os.path.join(script_dir, "model_outputs")
    OUTPUT_MODEL_SUBDIR = f"RQ_GPR_test{TARGET_TEST_OBJECT_NUM}" 
    FINAL_OUTPUT_DIR_FOR_TEST_OBJ = os.path.join(OUTPUT_MODEL_BASE_DIR, OUTPUT_MODEL_SUBDIR)

    if not os.path.exists(FINAL_OUTPUT_DIR_FOR_TEST_OBJ):
        os.makedirs(FINAL_OUTPUT_DIR_FOR_TEST_OBJ)
        print(f"Created output directory: {FINAL_OUTPUT_DIR_FOR_TEST_OBJ}")

    if not os.path.isdir(INPUT_DATA_DIR):
        print(f"Error: Input directory '{INPUT_DATA_DIR}' not found.")
        # Generate a minimal report indicating the issue
        _generate_summary_report(
            TARGET_TEST_OBJECT_NUM,
            FINAL_OUTPUT_DIR_FOR_TEST_OBJ,
            [], # No input files processed
            0,  
            [f"Input directory '{INPUT_DATA_DIR}' not found."], 
            None, 
            "N/A (Setup error)", 
            "N/A", 
            "N/A", 
            [] 
        )
    else:
        all_csv_files_in_dir = glob.glob(os.path.join(INPUT_DATA_DIR, "*.csv"))
        
        target_tag = f"[test {TARGET_TEST_OBJECT_NUM}]".lower()
        
        selected_test_object_files = [
            f for f in all_csv_files_in_dir 
            if target_tag in os.path.basename(f).lower()
        ]
        
        if not selected_test_object_files:
            print(f"No files clearly identified as belonging to '{target_tag}' found in '{INPUT_DATA_DIR}'. Please check filenames and patterns.")
            # Generate a minimal report if no files are found for the target object
            _generate_summary_report(
                TARGET_TEST_OBJECT_NUM,
                FINAL_OUTPUT_DIR_FOR_TEST_OBJ,
                [], # No specific files found for this test object
                0, 
                [f"No files for '{target_tag}' found in '{INPUT_DATA_DIR}'."],
                None, 
                "N/A (No data processed)",
                "N/A",
                "N/A",
                [] 
            )
        else:
            print(f"Found {len(selected_test_object_files)} files specifically for '{target_tag}' (after filtering) to process for GPR training from '{INPUT_DATA_DIR}':")
            for f_path in sorted(selected_test_object_files): 
                print(f"  - {os.path.basename(f_path)}")
            
            train_gpr_for_selected_object(
                sorted(selected_test_object_files), 
                TARGET_TEST_OBJECT_NUM, 
                FINAL_OUTPUT_DIR_FOR_TEST_OBJ 
            )

    print("\nScript finished.")