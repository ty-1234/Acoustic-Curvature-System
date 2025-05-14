"""
RBF_GPR_train.py - Gaussian Process Regression for Curvature Sensor Data

This script trains a Gaussian Process Regression (GPR) model with an RBF (Radial Basis Function) kernel
to predict surface curvature from sensor data collected by a curvature sensor.

Process Overview:
1. Loads pre-cleaned CSV files for a specific test session (e.g., "[test 1]")
2. Normalizes FFT readings by subtracting a baseline calculated from "idle" periods
3. Performs Leave-One-Curvature-Profile-Out cross-validation:
   - Trains on N-1 curvature profiles, tests on the held-out profile
   - Repeats until each curvature profile has been used as the test set once
4. Evaluates model performance with detailed metrics (MSE, MAE, R2)
5. Generates visualization plots to assess model accuracy
6. Trains a final model on all data for deployment

Cross-Validation Approach:
- Uses LeaveOneGroupOut where each "group" is a distinct curvature profile (e.g., 0.01, 0.05 m⁻¹)
- This tests how well the model generalizes to entirely new curvature values
- Represents a realistic evaluation of how the model will perform on unseen curvatures

Generated Visualizations:
1. Predicted vs. True Curvature: Shows prediction accuracy across all tested profiles
2. Residuals vs. True Curvature: Displays error patterns across different curvature values
3. Absolute Error Distribution (Box Plot by Group): Compares model accuracy across different curvature profiles
4. Absolute Error vs. Prediction Standard Deviation: Evaluates the reliability of the model's
   uncertainty estimates (important for Gaussian Processes)

Usage:
- Set TARGET_TEST_SESSION_NUM to select which test session to process
- Ensure input CSV files are in the expected location with naming convention:
  "cleaned_<CURVATURE>[test N].csv"
- Output files (models, metrics, plots) are saved to a session-specific directory
"""

import pandas as pd
import numpy as np
import os
import glob
import re
import time
from datetime import datetime
import json # Ensure this line is present
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel # Changed RationalQuadratic to RBF
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================================
# --- Global Configuration ---
# =================================================================================
FFT_COLUMNS = ['FFT_200Hz', 'FFT_400Hz', 'FFT_600Hz', 'FFT_800Hz',
               'FFT_1000Hz', 'FFT_1200Hz', 'FFT_1400Hz', 'FFT_1600Hz',
               'FFT_1800Hz', 'FFT_2000Hz']
POSITION_COLUMN = 'Position_cm'
TARGET_COLUMN = 'Curvature'
ACTIVITY_COLUMN = 'Curvature_Active'
TIMESTAMP_COLUMN = 'Timestamp' # Not used in GPR training but present in input files
FEATURE_COLUMNS = FFT_COLUMNS + [POSITION_COLUMN]

# =================================================================================
# --- Function: Parse Filename for Group (Curvature Value for a specific test object/session) ---
# =================================================================================
def parse_filename_for_curvature_group(filename, target_test_object_num_str):
    """
    Extracts a group identifier (curvature value string) from filenames like
    'cleaned_0_01[test 1].csv' -> '0.01' (if target_test_object_num_str is "1")
    It ensures it's parsing for the correct [test N] tag.
    """
    base_name = os.path.basename(filename)
    # Regex to capture curvature, specifically for the target_test_object_num_str
    # Example: if target_test_object_num_str is "1", pattern looks for (anyprefix)_<CURVATURE_STR>[test 1]
    pattern_str = rf".*?_([0-9._]+?)\s*\[test\s*{re.escape(target_test_object_num_str)}\]"
    match = re.match(pattern_str, base_name, re.IGNORECASE)
    if match:
        return match.group(1).replace('_', '.') 
    
    # Fallback if the above pattern (which requires the prefix like merged_ or cleaned_) fails,
    # but the file was already selected because it contains the target_test_object_num_str.
    # Try a more general extraction if the test tag is confirmed.
    if f"[test {target_test_object_num_str}]".lower() in base_name.lower():
        # Attempt to grab what's between the first underscore and "[test N]"
        simple_match = re.match(r".*?_([0-9._]+)", base_name) # Greedily takes up to first non-digit/._
        if simple_match:
            # Check if what follows is indeed the test tag (to avoid false positives)
            potential_curvature = simple_match.group(1)
            if f"{potential_curvature}[test {target_test_object_num_str}]".lower() in base_name.lower() or \
               f"{potential_curvature.replace('.', '_')}[test {target_test_object_num_str}]".lower() in base_name.lower():
                return potential_curvature.replace('_', '.')

    print(f"    Warning: Could not accurately parse curvature group from filename: {base_name} for test object/session {target_test_object_num_str}. Using full filename stem as group.")
    return os.path.splitext(base_name)[0]

# =================================================================================
# --- Function: Generate Performance Plots ---
# =================================================================================
def generate_performance_plots(detailed_predictions_df, output_dir, model_name_prefix="gpr"):
    if detailed_predictions_df.empty:
        print("  No detailed prediction data to generate plots.")
        return
    
    print("\n  Generating performance plots...")
    plot_filename_prefix = f"{model_name_prefix}_" # Corrected: use the passed prefix

    # Plot 1: Predicted vs. True Curvature
    plt.figure(figsize=(9, 8))
    try:
        sns.scatterplot(x='true_curvature', y='predicted_curvature', hue='tested_group_for_row', 
                        data=detailed_predictions_df, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
        min_val = min(detailed_predictions_df['true_curvature'].min(), detailed_predictions_df['predicted_curvature'].min())
        max_val = max(detailed_predictions_df['true_curvature'].max(), detailed_predictions_df['predicted_curvature'].max())
        padding = (max_val - min_val) * 0.05 
        plot_min = min_val - padding if (padding > 0 and max_val > min_val) else min_val - 0.1 
        plot_max = max_val + padding if (padding > 0 and max_val > min_val) else max_val + 0.1
        plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=2, label="Ideal (y=x)")
        plt.xlabel("True Curvature ($m^{-1}$)")
        plt.ylabel("Predicted Curvature ($m^{-1}$)")
        plt.title(f"Predicted vs. True Curvature ({model_name_prefix.upper()})")
        plt.legend(title="Tested Curvature Group", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.axis('equal')
        plt.xlim(plot_min, plot_max)
        plt.ylim(plot_min, plot_max)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) 
        plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}pred_vs_true.png"), bbox_inches='tight')
        plt.close()
        print(f"    Saved: Predicted vs. True plot ({os.path.join(output_dir, f'{plot_filename_prefix}pred_vs_true.png')})")
    except Exception as e:
        print(f"    Error generating Predicted vs. True plot: {e}")

    # Plot 2: Residuals vs. True Curvature
    plt.figure(figsize=(9, 6))
    try:
        sns.scatterplot(x='true_curvature', y='error', hue='tested_group_for_row', 
                        data=detailed_predictions_df, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
        plt.axhline(0, color='k', linestyle='--', lw=2)
        plt.xlabel("True Curvature ($m^{-1}$)")
        plt.ylabel("Residual (True - Predicted) ($m^{-1}$)")
        plt.title(f"Residuals vs. True Curvature ({model_name_prefix.upper()})")
        plt.legend(title="Tested Curvature Group", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}residuals_vs_true.png"), bbox_inches='tight')
        plt.close()
        print(f"    Saved: Residuals vs. True plot ({os.path.join(output_dir, f'{plot_filename_prefix}residuals_vs_true.png')})")
    except Exception as e:
        print(f"    Error generating Residuals vs. True plot: {e}")

    # Plot 3: Absolute Error Distribution (Box Plot by Group)
    if 'abs_error' in detailed_predictions_df.columns and 'tested_group_for_row' in detailed_predictions_df.columns:
        plt.figure(figsize=(12, 7))
        try:
            unique_groups_for_plot = detailed_predictions_df['tested_group_for_row'].unique()
            try:
                # Attempt numeric sort for group labels if possible
                sorted_groups = sorted(unique_groups_for_plot, key=lambda x: float(str(x).replace('_','.')))
            except ValueError: 
                sorted_groups = sorted(unique_groups_for_plot) # Fallback to string sort
            
            sns.boxplot(x='tested_group_for_row', y='abs_error', data=detailed_predictions_df, order=sorted_groups, palette="viridis")
            plt.xlabel("Tested Curvature Group")
            plt.ylabel("Absolute Error ($m^{-1}$)")
            plt.title(f"Absolute Error Distribution by Curvature Group ({model_name_prefix.upper()})")
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', linestyle=':', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}abs_error_boxplot_by_group.png"))
            plt.close()
            print(f"    Saved: Absolute Error Boxplot by Group ({os.path.join(output_dir, f'{plot_filename_prefix}abs_error_boxplot_by_group.png')})")
        except Exception as e:
            print(f"    Error generating Absolute Error Boxplot: {e}")
   
    # Plot 4: Prediction Standard Deviation vs. Absolute Error
    if 'prediction_std_dev' in detailed_predictions_df.columns and 'abs_error' in detailed_predictions_df.columns:
        plt.figure(figsize=(9, 6))
        try:
            sns.scatterplot(x='prediction_std_dev', y='abs_error', hue='tested_group_for_row',
                            data=detailed_predictions_df, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
            plt.xlabel("Prediction Standard Deviation ($m^{-1}$)")
            plt.ylabel("Absolute Error ($m^{-1}$)")
            plt.title(f"Absolute Error vs. Prediction Std Dev ({model_name_prefix.upper()})")
            plt.legend(title="Tested Curvature Group", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}abs_error_vs_pred_std.png"))
            plt.close()
            print(f"    Saved: Absolute Error vs. Prediction Std Dev plot ({os.path.join(output_dir, f'{plot_filename_prefix}abs_error_vs_pred_std.png')})")
        except Exception as e:
            print(f"    Error generating Error vs. Std Dev plot: {e}")

# =================================================================================
# --- Main GPR Training and Evaluation Function ---
# =================================================================================
def train_gpr_for_selected_test_run(
    input_files_for_session,
    test_session_num_str,
    output_dir_for_session
):
    all_gpr_ready_segments = []
    successful_baseline_count = 0
    files_without_baseline = []

    # --- Define GPR Configuration for Logging ---
    # Define initial kernel configuration string here to capture it before any fitting
    # This is the kernel used for CV folds and as the starting point for the final model
    initial_kernel_config = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)) # Changed RationalQuadratic to RBF
    initial_kernel_config_str = str(initial_kernel_config)
    gpr_n_restarts_optimizer_val = 9 # Store this as well
    gpr_normalize_y_val = True # As used in your GPR instances
    gpr_random_state_val = 42 # As used

    print(f"Starting GPR data preparation for {len(input_files_for_session)} files from session '[test {test_session_num_str}]'...")

    for file_path in input_files_for_session: 
        base_filename = os.path.basename(file_path)
        print(f"\n  Processing file: {base_filename}")
        try:
            # These files are from the 'cleaned' directory, already processed by noise_remover.py
            df_cleaned_input = pd.read_csv(file_path) 
            if df_cleaned_input.empty:
                print(f"    Skipping empty pre-cleaned file: {base_filename}")
                files_without_baseline.append(f"{base_filename} (File was empty)")
                continue
        except Exception as e:
            print(f"    Error reading pre-cleaned file {base_filename}: {e}. Skipping.")
            files_without_baseline.append(f"{base_filename} (Read error: {e})")
            continue

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
        
        active_df = df_cleaned_input[df_cleaned_input[ACTIVITY_COLUMN] == 1].copy()
        if active_df.empty:
            print(f"    No active data in {base_filename} after filtering. Skipping.")
            continue
        print(f"    Found {len(active_df)} active rows in pre-cleaned file.")

        if idle_fft_baseline is not None and not active_df.empty:
            active_df.loc[:, FFT_COLUMNS] = active_df[FFT_COLUMNS].values - idle_fft_baseline
            print(f"    Normalized {len(active_df)} active rows using baseline subtraction.")
        elif not active_df.empty:
            print(f"    Active data for {base_filename} not baseline-subtracted (no suitable idle baseline from pre-cleaned file).")
        
        if active_df.empty:
            continue 
            
        # Pass the test_session_num_str to correctly parse group for THIS session's files
        group_id = parse_filename_for_curvature_group(base_filename, test_session_num_str) 
        active_df['group'] = group_id
        
        all_gpr_ready_segments.append(active_df)

    print(f"\nNormalization Baseline Calculation Summary for {len(input_files_for_session)} files:")
    print(f"  Successfully calculated idle baseline for {successful_baseline_count} files.")
    if files_without_baseline:
        print(f"  Files where specific idle baseline conditions were not met or other issues occurred:")
        for f_issue in files_without_baseline:
            print(f"    - {f_issue}")

    if not all_gpr_ready_segments:
        print("No data available for GPR training after processing all selected files.")
        return

    combined_df = pd.concat(all_gpr_ready_segments, ignore_index=True)
    if combined_df.empty:
        print("Combined DataFrame is empty. Cannot proceed with GPR.")
        return
        
    print(f"\nCombined preprocessed data shape for GPR: {combined_df.shape}")

    X = combined_df[FEATURE_COLUMNS]
    y = combined_df[TARGET_COLUMN]
    groups = combined_df['group'] # This now contains the CURVATURE_STR for the selected test session
    unique_group_ids = groups.unique()

    model_name_prefix = f"rbf_gpr_test{test_session_num_str}" # Changed prefix to reflect RBF kernel

    if len(unique_group_ids) < 2:
        print(f"Only {len(unique_group_ids)} unique curvature group(s) found for test session {test_session_num_str}: {unique_group_ids}. LeaveOneGroupOut CV requires at least 2 groups. CV will be skipped.")
        # Optionally, train a single model if only one group or not enough for CV
        if not X.empty and not y.empty:
            print("Consider training a single model on all data for this session if CV cannot be performed.")
        return 
    
    logo = LeaveOneGroupOut()
    fold_metrics = []
    all_fold_predictions_data = [] 
    fold_num = 0

    print(f"\nStarting Leave-One-Curvature-Profile-Out CV for session '[test {test_session_num_str}]' using {len(unique_group_ids)} curvature groups: {sorted(list(unique_group_ids))}...")

    for train_idx, test_idx in logo.split(X, y, groups):
        fold_num += 1
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        current_test_group_curvature = groups.iloc[test_idx].unique()[0]
        print(f"\n--- Fold {fold_num}: Testing on Curvature Group '{current_test_group_curvature}' (from session [test {test_session_num_str}]) ---")
        print(f"  Train set size: {len(X_train)}, Test set size: {len(X_test)}")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) \
                 + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1)) # Changed RationalQuadratic to RBF
        
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=gpr_n_restarts_optimizer_val, # Use the stored value
                                     random_state=gpr_random_state_val, normalize_y=gpr_normalize_y_val) # Use stored values

        print(f"  Fitting GPR for fold {fold_num}...")
        start_time_fold = time.time()
        gpr.fit(X_train_scaled, y_train)
        fit_time_fold = time.time() - start_time_fold
        print(f"  GPR fitting completed in {fit_time_fold:.2f}s. Learned kernel: {gpr.kernel_}")

        y_pred, y_std = gpr.predict(X_test_scaled, return_std=True)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse) # Calculate RMSE
        mae = mean_absolute_error(y_test, y_pred) # MAE is already calculated
        r2 = r2_score(y_test, y_pred)
        
        fold_metrics.append({
            'fold': fold_num, 'tested_group': current_test_group_curvature, 
            'mse': mse, 'rmse': rmse, 'mae': mae, # Both rmse and mae are stored per fold
            'r2': r2, 'fitting_time_s': fit_time_fold, 'learned_kernel': str(gpr.kernel_)
        })
        print(f"  Fold {fold_num} Results: MSE = {mse:.6f}, RMSE = {rmse:.6f}, MAE = {mae:.6f}, R2 = {r2:.4f}")

        fold_test_df = pd.DataFrame({
            'true_curvature': y_test.values, 'predicted_curvature': y_pred,
            'prediction_std_dev': y_std, 'error': y_test.values - y_pred,
            'abs_error': np.abs(y_test.values - y_pred), 'fold_number': fold_num,
            'tested_group_for_row': current_test_group_curvature 
        })
        for col in X_test.columns: 
            if col not in fold_test_df.columns:
                fold_test_df[col] = X_test[col].values
        all_fold_predictions_data.append(fold_test_df)

    print("\n--- Cross-Validation Finished ---")

    # Initialize CV summary metrics to default values
    avg_mse, avg_rmse, avg_mae, avg_r2, avg_fit_time = None, None, None, None, None
    num_folds_run = 0 # Default to 0 folds

    if fold_metrics:
        results_df = pd.DataFrame(fold_metrics)
        print(f"\nCross-Validation Results Summary (Leave-One-Curvature-Profile-Out for session [test {test_session_num_str}]):")
        columns_to_print = ['fold', 'tested_group', 'mse', 'rmse', 'mae', 'r2', 'fitting_time_s', 'learned_kernel']
        with pd.option_context('display.max_colwidth', None, 'display.width', 1000):
             print(results_df[columns_to_print])
        
        avg_mse = results_df['mse'].mean()
        avg_rmse = results_df['rmse'].mean()
        avg_mae = results_df['mae'].mean()
        avg_r2 = results_df['r2'].mean()
        avg_fit_time = results_df['fitting_time_s'].mean()
        num_folds_run = len(results_df) # Assign num_folds_run here

        print(f"\nAverage CV MSE: {avg_mse:.6f}")
        print(f"Average CV RMSE: {avg_rmse:.6f}")
        print(f"Average CV MAE: {avg_mae:.6f}")
        print(f"Average CV R2 Score: {avg_r2:.4f}")
        
        output_csv_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_LOCO_curvature_cv_results.csv")
        try:
            results_df.to_csv(output_csv_path, index=False)
            print(f"CV results saved to {output_csv_path}")
        except Exception as e:
            print(f"Error saving CV results to {output_csv_path}: {e}")

    if all_fold_predictions_data:
        detailed_predictions_df = pd.concat(all_fold_predictions_data, ignore_index=True)
        detailed_output_csv_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_detailed_predictions.csv")
        try:
            detailed_predictions_df.to_csv(detailed_output_csv_path, index=False)
            print(f"Detailed predictions and errors saved to {detailed_output_csv_path}")
            generate_performance_plots(detailed_predictions_df, output_dir_for_session, model_name_prefix)
        except Exception as e:
            print(f"Error saving detailed predictions CSV or generating plots: {e}")

    print(f"\n--- Training Final Model on All Processed Data for session [test {test_session_num_str}] ---")
    
    # Initialize variables for final model summary to handle cases where training might be skipped
    learned_kernel_final_str = None
    fit_time_final_val = None
    # num_samples_final_train is implicitly handled by len(X) if not X.empty else 0 later

    if not X.empty and not y.empty:
        final_scaler = StandardScaler()
        X_scaled_final = final_scaler.fit_transform(X) 

        # Use the same initial kernel structure and GPR params for consistency
        # The 'initial_kernel_config' is already RBF due to changes above
        final_gpr = GaussianProcessRegressor(kernel=initial_kernel_config, 
                                             n_restarts_optimizer=gpr_n_restarts_optimizer_val, 
                                             random_state=gpr_random_state_val, 
                                             normalize_y=gpr_normalize_y_val)

        print(f"Fitting final GPR model on all data for session [test {test_session_num_str}] (Size: {X_scaled_final.shape[0]})...")
        start_time_final = time.time()
        final_gpr.fit(X_scaled_final, y)
        fit_time_final_val = time.time() - start_time_final # Assigned if model is trained
        learned_kernel_final_str = str(final_gpr.kernel_) # Assigned if model is trained
        print(f"Final GPR model fitted in {fit_time_final_val:.2f}s. Learned kernel: {learned_kernel_final_str}")

        model_save_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_final_model.joblib")
        scaler_save_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_final_scaler.joblib")

        try:
            joblib.dump(final_gpr, model_save_path)
            joblib.dump(final_scaler, scaler_save_path)
            print(f"Final GPR model saved to: {model_save_path}")
            print(f"Final Scaler saved to: {scaler_save_path}")
        except Exception as e:
            print(f"Error saving final model/scaler: {e}")
    else:
        print("Skipping final model training as combined data is empty or X/y were not available.")
        # learned_kernel_final_str and fit_time_final_val will remain None as initialized above

    # --- Create and Save JSON Summary ---
    summary_data = {
        "experiment_info": {
            "test_session_id": test_session_num_str,
            "timestamp_utc": datetime.now(datetime.timezone.utc).isoformat(), # Corrected for deprecation
            "model_name_prefix": model_name_prefix,
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN
        },
        "gpr_configuration": {
            "initial_kernel": initial_kernel_config_str,
            "n_restarts_optimizer": gpr_n_restarts_optimizer_val,
            "normalize_y_gpr": gpr_normalize_y_val,
            "random_state_gpr": gpr_random_state_val
        },
        "data_preprocessing": {
            "feature_scaler": "StandardScaler" 
            # Add more details here if scaling becomes more complex or configurable
        },
        "cross_validation_summary": {
            "cv_strategy": "LeaveOneGroupOut (on curvature profiles)",
            "num_folds_run": num_folds_run,
            "avg_mse": avg_mse,
            "avg_rmse": avg_rmse, # <--- Average RMSE is included here
            "avg_mae": avg_mae,   # <--- Average MAE is included here
            "avg_r2": avg_r2,
            "avg_fitting_time_s_per_fold": avg_fit_time
        },
        "final_model_summary": {
            "trained_on_all_session_data": (learned_kernel_final_str is not None),
            "learned_kernel_final_model": learned_kernel_final_str,
            "training_time_s_final_model": fit_time_final_val,
            "num_samples_final_train": len(X) if not X.empty else 0
        }
    }

    json_summary_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_experiment_summary.json")
    try:
        with open(json_summary_path, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"Experiment summary JSON saved to: {json_summary_path}")
    except Exception as e:
        print(f"Error saving experiment summary JSON: {e}")

# =================================================================================
# --- Main Execution ---
# =================================================================================
if __name__ == "__main__":
    # === OPERATOR: SET THE TARGET TEST SESSION NUMBER HERE (e.g., "1", "2", "3", "4") ===
    TARGET_TEST_SESSION_NUM = "1" 
    # =================================================================================

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # INPUT_DATA_DIR should point to where your 'cleaned_*.csv' files are,
    # which are the output of noise_remover.py
    INPUT_DATA_DIR = os.path.join(script_dir, "csv_data/cleaned/") 
    
    # Output directory for models, results, and plots from THIS script, specific to the session
    OUTPUT_RESULTS_BASE_DIR = os.path.join(script_dir, "gpr_model_outputs") 
    # Modified OUTPUT_SESSION_SUBDIR to include kernel type for better organization
    OUTPUT_SESSION_SUBDIR = f"RBF_GPR_test_session_{TARGET_TEST_SESSION_NUM}" 
    FINAL_OUTPUT_DIR_FOR_SESSION = os.path.join(OUTPUT_RESULTS_BASE_DIR, OUTPUT_SESSION_SUBDIR)

    if not os.path.exists(FINAL_OUTPUT_DIR_FOR_SESSION):
        os.makedirs(FINAL_OUTPUT_DIR_FOR_SESSION)
        print(f"Created output directory: {FINAL_OUTPUT_DIR_FOR_SESSION}")

    if not os.path.isdir(INPUT_DATA_DIR):
        print(f"Error: Input directory '{INPUT_DATA_DIR}' not found.")
    else:
        # Glob for files matching "cleaned_*" and containing the target test session tag
        all_cleaned_files_in_dir = glob.glob(os.path.join(INPUT_DATA_DIR, "cleaned_*.csv"))
        
        target_session_tag = f"[test {TARGET_TEST_SESSION_NUM}]".lower() # Ensure case-insensitivity
        
        selected_session_files = [
            f for f in all_cleaned_files_in_dir 
            if target_session_tag in os.path.basename(f).lower()
        ]
        
        if not selected_session_files:
             print(f"No 'cleaned_*.csv' files found containing '{target_session_tag}' in '{INPUT_DATA_DIR}'. Please ensure 'noise_remover.py' has run and filenames are correct.")
        else:
            print(f"Found {len(selected_session_files)} 'cleaned_*.csv' files for session '{target_session_tag}' to process for GPR training from '{INPUT_DATA_DIR}':")
            for f_path in sorted(selected_session_files): 
                print(f"  - {os.path.basename(f_path)}")
            
            train_gpr_for_selected_test_run(
                sorted(selected_session_files), 
                TARGET_TEST_SESSION_NUM, 
                FINAL_OUTPUT_DIR_FOR_SESSION 
            )

    print("\nScript finished.")