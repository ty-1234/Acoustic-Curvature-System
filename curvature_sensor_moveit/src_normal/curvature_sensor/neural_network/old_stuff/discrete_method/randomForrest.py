import pandas as pd
import numpy as np
import os
import glob
import re
import time
from datetime import datetime
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor # Changed from GPR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# joblib might not be needed if not saving RF models, but keep if saving final RF
import joblib 

# --- Global Configuration (Same as your GPR script) ---
FFT_COLUMNS = ['FFT_200Hz', 'FFT_400Hz', 'FFT_600Hz', 'FFT_800Hz',
               'FFT_1000Hz', 'FFT_1200Hz', 'FFT_1400Hz', 'FFT_1600Hz',
               'FFT_1800Hz', 'FFT_2000Hz']
POSITION_COLUMN = 'Position_cm'
TARGET_COLUMN = 'Curvature'
ACTIVITY_COLUMN = 'Curvature_Active'
TIMESTAMP_COLUMN = 'Timestamp'
FEATURE_COLUMNS = FFT_COLUMNS + [POSITION_COLUMN]

# --- Function: process_data_for_discrete_extraction (Noise/Segment Removal) ---
# This function is REMOVED as noise_remover.py is responsible for this.

# --- Function: Parse Filename for Group (Curvature Value for a specific test object) ---
# (This is the same function from your RQ_GPR_train.py script)
def parse_filename_for_group_curvature(filename, target_test_object_num_str):
    base_name = os.path.basename(filename)
    pattern_str = rf".*?_([0-9._]+?)\s*\[test\s*{re.escape(target_test_object_num_str)}\]"
    match = re.match(pattern_str, base_name, re.IGNORECASE)
    if match:
        return match.group(1).replace('_', '.') 
    
    pattern_cleaned_str = rf"cleaned_([0-9._]+?)\s*\[test\s*{re.escape(target_test_object_num_str)}\]"
    match_cleaned = re.match(pattern_cleaned_str, base_name, re.IGNORECASE)
    if match_cleaned:
        return match_cleaned.group(1).replace('_','.')

    print(f"    Warning: Could not parse curvature group from filename: {base_name} for [test {target_test_object_num_str}] pattern. Using full filename stem as group.")
    return os.path.splitext(base_name)[0] 

# --- Main Random Forest Training and Evaluation Function (adapted from GPR version) ---
def train_random_forest_for_selected_object(file_paths, target_test_object_num_str, output_model_dir_for_test_obj):
    all_rf_ready_segments = [] # Changed variable name prefix
    successful_baseline_count = 0
    files_without_baseline = []
    generated_output_files = []

    print(f"Starting Random Forest data preparation for {len(file_paths)} files for test object '[test {target_test_object_num_str}]' (expecting pre-cleaned files from 'cleaned' directory)...")

    for file_path in file_paths: 
        base_filename = os.path.basename(file_path)
        print(f"\n  Processing file: {base_filename}")
        try:
            # df_cleaned_input IS the file already processed by noise_remover.py
            df_cleaned_input = pd.read_csv(file_path) 
            if df_cleaned_input.empty:
                print(f"    Skipping empty pre-cleaned file: {base_filename}")
                files_without_baseline.append(f"{base_filename} (File was empty)")
                continue
        except Exception as e:
            print(f"    Error reading pre-cleaned file {base_filename}: {e}. Skipping.")
            files_without_baseline.append(f"{base_filename} (Read error: {e})")
            continue

        # 1. Calculate Idle Baseline (from the pre-cleaned df_cleaned_input)
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

        # THE CALL TO process_data_for_discrete_extraction IS REMOVED.
        # We now directly use df_cleaned_input.
        print(f"    Using pre-cleaned df shape for RF processing: {df_cleaned_input.shape}")
        
        # Filter for Active Data from df_cleaned_input
        active_df = df_cleaned_input[df_cleaned_input[ACTIVITY_COLUMN] == 1].copy()
        if active_df.empty:
            print(f"    No active data in pre-cleaned file {base_filename} after filtering. Skipping.")
            continue
        print(f"    Found {len(active_df)} active rows in pre-cleaned file.")

        # Normalize FFT Features of Active Data (Subtraction)
        if idle_fft_baseline is not None and not active_df.empty:
            active_df.loc[:, FFT_COLUMNS] = active_df[FFT_COLUMNS].values - idle_fft_baseline
            print(f"    Normalized {len(active_df)} active rows using baseline subtraction.")
        elif not active_df.empty:
            print(f"    Active data for {base_filename} not baseline-subtracted (no suitable idle baseline was found from pre-cleaned file).")
        
        if active_df.empty:
            continue 
            
        group_id = parse_filename_for_group_curvature(base_filename, target_test_object_num_str)
        active_df['group'] = group_id
        
        all_rf_ready_segments.append(active_df)

    print(f"\nNormalization Baseline Calculation Summary for {len(file_paths)} files:")
    print(f"  Successfully calculated idle baseline for {successful_baseline_count} files.")
    if files_without_baseline:
        print(f"  Files where specific idle baseline conditions were not met or other issues occurred:")
        for f_issue in files_without_baseline:
            print(f"    - {f_issue}")
    
    results_df = None 
    final_rf_params_str = "N/A (Final model not trained or training skipped)"
    # For RF, we usually don't save the model in a sanity check, but paths can be defined if needed
    # model_save_path_str = "N/A" 
    # scaler_save_path_str = "N/A"

    if not all_rf_ready_segments:
        print("No data available for Random Forest training after processing all selected files.")
        _generate_rf_summary_report(target_test_object_num_str, output_model_dir_for_test_obj, file_paths,
                                 successful_baseline_count, files_without_baseline, results_df, 
                                 final_rf_params_str, #model_save_path_str, scaler_save_path_str,
                                 generated_output_files)
        return

    combined_df = pd.concat(all_rf_ready_segments, ignore_index=True)
    if combined_df.empty:
        print("Combined DataFrame is empty. Cannot proceed with Random Forest training.")
        _generate_rf_summary_report(target_test_object_num_str, output_model_dir_for_test_obj, file_paths,
                                 successful_baseline_count, files_without_baseline, results_df, 
                                 final_rf_params_str, #model_save_path_str, scaler_save_path_str,
                                 generated_output_files)
        return
        
    print(f"\nCombined preprocessed data shape for Random Forest: {combined_df.shape}")

    X = combined_df[FEATURE_COLUMNS]
    y = combined_df[TARGET_COLUMN]
    groups = combined_df['group']
    unique_group_ids = groups.unique()

    if len(unique_group_ids) < 2:
        print(f"Only {len(unique_group_ids)} unique group(s) found: {unique_group_ids}. LeaveOneGroupOut CV requires at least 2 groups. CV will be skipped for Random Forest.")
    else:
        logo = LeaveOneGroupOut()
        fold_metrics = []
        all_fold_predictions_data = [] 
        fold_num = 0

        print(f"\nStarting Leave-One-Curvature-Out Cross-Validation for Random Forest (test object '[test {target_test_object_num_str}]') using {len(unique_group_ids)} groups: {sorted(list(unique_group_ids))}...")

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

            # Initialize RandomForestRegressor
            # Common parameters: n_estimators, max_depth, random_state
            # For a quick sanity check, default or simple parameters are often fine.
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # Use all cores

            print(f"  Fitting Random Forest for fold {fold_num}...")
            start_time_fold = time.time()
            rf_model.fit(X_train_scaled, y_train)
            fit_time_fold = time.time() - start_time_fold
            # Feature importances could be interesting but not essential for quick sanity check
            # importances = rf_model.feature_importances_ 
            print(f"  Random Forest fitting completed in {fit_time_fold:.2f}s.")

            y_pred = rf_model.predict(X_test_scaled)
            # RF doesn't provide prediction_std_dev in the same way GPR does directly

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            fold_metrics.append({
                'fold': fold_num, 
                'tested_group': current_test_group, 
                'mse': mse, 
                'mae': mae,
                'r2': r2, 
                'fitting_time_s': fit_time_fold,
                # For RF, 'learned_kernel' is not applicable, store model params if desired
                'model_params': str(rf_model.get_params()) 
            })
            print(f"  Fold {fold_num} RF Results: MSE = {mse:.6f}, MAE = {mae:.6f}, R2 = {r2:.4f}")

            fold_test_df = pd.DataFrame({
                'true_curvature': y_test.values, 'predicted_curvature': y_pred,
                # 'prediction_std_dev': np.nan, # RF doesn't have this directly like GPR
                'error': y_test.values - y_pred,
                'abs_error': np.abs(y_test.values - y_pred), 'fold_number': fold_num,
                'tested_group_for_row': current_test_group 
            })
            fold_test_df[POSITION_COLUMN] = X_test[POSITION_COLUMN].values 
            all_fold_predictions_data.append(fold_test_df)

        print("\n--- Cross-Validation Finished ---")
        if fold_metrics:
            results_df = pd.DataFrame(fold_metrics)
            print(f"\nRandom Forest CV Results Summary (Leave-One-Curvature-Out for [test {target_test_object_num_str}] data):")
            columns_to_print = ['fold', 'tested_group', 'mse', 'mae', 'r2', 'fitting_time_s'] # Removed kernel, can add model_params
            with pd.option_context('display.max_colwidth', None, 'display.width', 1000):
                 print(results_df[columns_to_print])
            print(f"\nAverage CV MSE: {results_df['mse'].mean():.6f}")
            print(f"Average CV MAE: {results_df['mae'].mean():.6f}")
            print(f"Average CV R2 Score: {results_df['r2'].mean():.4f}")
            
            output_csv_path = os.path.join(output_model_dir_for_test_obj, f"rf_sanity_test{target_test_object_num_str}_LOCO_cv_results.csv")
            try:
                results_df.to_csv(output_csv_path, index=False)
                print(f"RF CV results saved to {output_csv_path}")
                generated_output_files.append(output_csv_path)
            except Exception as e:
                print(f"Error saving RF CV results to {output_csv_path}: {e}")

        if all_fold_predictions_data:
            detailed_predictions_df = pd.concat(all_fold_predictions_data, ignore_index=True)
            detailed_output_csv_path = os.path.join(output_model_dir_for_test_obj, f"rf_sanity_test{target_test_object_num_str}_detailed_predictions.csv")
            try:
                detailed_predictions_df.to_csv(detailed_output_csv_path, index=False)
                print(f"RF Detailed predictions saved to {detailed_output_csv_path}")
                generated_output_files.append(detailed_output_csv_path)
            except Exception as e:
                print(f"Error saving RF detailed predictions CSV: {e}")
    
    # No final model training for RF sanity check by default, can be added if needed
    # ...

    # --- Generate Summary Report for RF ---
    _generate_rf_summary_report(
        target_test_object_num_str,
        output_model_dir_for_test_obj,
        file_paths,
        successful_baseline_count,
        files_without_baseline,
        results_df, # This is the CV results DataFrame
        generated_output_files
    )

def _generate_rf_summary_report(target_test_object_num_str, output_dir, input_file_paths,
                             successful_baseline_count, files_without_baseline,
                             cv_results_df, # My note: GPR specific final model params removed here
                             generated_files_list_so_far):
    """Generates a Markdown summary report for Random Forest."""
    # ----------------------------------------------------------------------------------
    # Section: Initialization and Report Header
    # ----------------------------------------------------------------------------------
    report_filename = f"rf_sanity_test{target_test_object_num_str}_summary_report.md"
    report_full_path = os.path.join(output_dir, report_filename)
    all_output_files_for_report = list(generated_files_list_so_far)
    all_output_files_for_report.append(report_full_path)

    report_content = []
    report_content.append(f"# Random Forest (Sanity Check) Training Summary: Test Object `[test {target_test_object_num_str}]`")
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
    elif input_file_paths:
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
        report_content.append(f"- **Average CV MAE:** {cv_results_df['mae'].mean():.6f}")
        report_content.append(f"- **Average CV R² Score:** {cv_results_df['r2'].mean():.4f}")
        report_content.append("\n### Per-Fold Metrics:")
        header = "| Fold | Tested Group | MSE      | MAE      | R²     | Fitting Time (s) | Model Parameters (RF) |"
        separator = "|------|--------------|----------|----------|--------|------------------|-----------------------|"
        report_content.append(header)
        report_content.append(separator)
        for _, row in cv_results_df.iterrows():
            report_content.append(f"| {row['fold']} | {row['tested_group']} | {row['mse']:.6f} | {row['mae']:.6f} | {row['r2']:.4f} | {row['fitting_time_s']:.2f} | `{row.get('model_params', 'N/A')}` |")
    else:
        report_content.append("- Cross-validation was not performed or did not yield results.")

    # report_content.append("\n## 3. Final Model Information") # Usually not doing this for my sanity check RF
    # report_content.append("- Final Random Forest model was not trained/saved as part of this sanity check.")

    # ----------------------------------------------------------------------------------
    # Section: Generated Output Files
    # ----------------------------------------------------------------------------------
    report_content.append("\n## 3. Generated Output Files") # Renumbered section
    if all_output_files_for_report:
        report_content.append(f"The following files were generated in or for the directory: `{output_dir}`")
        for f_path in sorted(list(set(all_output_files_for_report))):
            report_content.append(f"- `{os.path.basename(f_path)}`")
    else:
        report_content.append("- No output files were explicitly tracked or generated.")
        
    # ----------------------------------------------------------------------------------
    # Section: Saving the Report
    # ----------------------------------------------------------------------------------
    try:
        with open(report_full_path, 'w') as f:
            f.write("\n".join(report_content))
        print(f"\nRandom Forest summary report saved to: {report_full_path}")
    except Exception as e:
        print(f"Error saving Random Forest summary report to {report_full_path}: {e}")


if __name__ == "__main__":
    # === OPERATOR: SET THE TARGET TEST OBJECT NUMBER HERE ===
    TARGET_TEST_OBJECT_NUM = "1" # Change to "2", "3", "4", etc. as needed
    # ========================================================

    script_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_DATA_DIR = os.path.join(script_dir, "csv_data/cleaned/") # Same input data
    
    # Modify output directory to be specific to the test object AND model type
    OUTPUT_MODEL_BASE_DIR = os.path.join(script_dir, "model_outputs")
    # Changed Subdir to reflect RF
    OUTPUT_MODEL_SUBDIR = f"RF_sanity_test{TARGET_TEST_OBJECT_NUM}" 
    FINAL_OUTPUT_DIR_FOR_TEST_OBJ = os.path.join(OUTPUT_MODEL_BASE_DIR, OUTPUT_MODEL_SUBDIR)

    if not os.path.exists(FINAL_OUTPUT_DIR_FOR_TEST_OBJ):
        os.makedirs(FINAL_OUTPUT_DIR_FOR_TEST_OBJ)
        print(f"Created output directory: {FINAL_OUTPUT_DIR_FOR_TEST_OBJ}")

    if not os.path.isdir(INPUT_DATA_DIR):
        print(f"Error: Input directory '{INPUT_DATA_DIR}' not found.")
        _generate_rf_summary_report(
            TARGET_TEST_OBJECT_NUM, FINAL_OUTPUT_DIR_FOR_TEST_OBJ,
            [], 0, [f"Input directory '{INPUT_DATA_DIR}' not found."],
            None, []
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
             _generate_rf_summary_report(
                 TARGET_TEST_OBJECT_NUM, FINAL_OUTPUT_DIR_FOR_TEST_OBJ,
                 [], 0, [f"No files for '{target_tag}' found in '{INPUT_DATA_DIR}'."],
                 None, []
             )
        else:
            print(f"Found {len(selected_test_object_files)} files specifically for '{target_tag}' for Random Forest sanity check from '{INPUT_DATA_DIR}':")
            for f_path in sorted(selected_test_object_files): 
                print(f"  - {os.path.basename(f_path)}")
            
            train_random_forest_for_selected_object(
                sorted(selected_test_object_files), 
                TARGET_TEST_OBJECT_NUM, 
                FINAL_OUTPUT_DIR_FOR_TEST_OBJ 
            )

    print("\nRandom Forest sanity check script finished.")