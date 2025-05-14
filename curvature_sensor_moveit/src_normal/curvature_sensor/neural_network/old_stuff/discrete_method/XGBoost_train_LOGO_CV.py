"""
XGBoost_train_LOGO_CV.py - XGBoost for Curvature Sensor Data with LOGO CV

This script trains an XGBoost model to predict surface curvature
from sensor data, using Leave-One-Curvature-Profile-Out cross-validation.

Process Overview:
1. Loads pre-cleaned CSV files from specified test sessions.
2. Combines data from all specified sessions.
3. Normalizes FFT readings by subtracting a baseline calculated from "idle" periods.
4. Converts target curvature units from mm^-1 to m^-1.
5. Performs Leave-One-Curvature-Profile-Out cross-validation (LOGO CV):
   - Groups data by unique curvature profiles.
   - Standardizes input features (FFT + Position) for each fold.
   - Trains an XGBoost model on N-1 curvature profiles, using an internal
     evaluation set for early stopping.
   - Tests on the held-out curvature profile.
   - Repeats until each curvature profile has been used as the test set.
6. Evaluates model performance with detailed metrics (MSE, RMSE, MAE, R2).
7. Generates visualization plots to assess model accuracy.
8. Trains a final XGBoost model on all combined data for potential deployment.

Cross-Validation Approach:
- Uses LeaveOneGroupOut where each "group" is a distinct curvature profile.
- This tests how well the model generalizes to entirely new curvature values.
"""

import pandas as pd
import numpy as np
import os
import glob
import re
import time
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
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
ORIGINAL_TARGET_COLUMN = 'Curvature' # Assuming this is in mm^-1
ACTIVITY_COLUMN = 'Curvature_Active'
TIMESTAMP_COLUMN = 'Timestamp' # Present in files, not used as a direct feature

FEATURE_COLUMNS = FFT_COLUMNS + [POSITION_COLUMN] # Inputs to XGBoost
TARGET_COLUMN_MM_INV = ORIGINAL_TARGET_COLUMN # Original target in mm^-1
TARGET_COLUMN_M_INV = 'Curvature_m_inv' # New target in m^-1, for model training

# =================================================================================
# --- Function: Parse Filename for Curvature Group (Independent of Session Tag) ---
# =================================================================================
def parse_filename_for_curvature_group(filename):
    """
    Extracts a group identifier (curvature value string) from filenames like
    'cleaned_0_01[test 1].csv' -> '0.01'
    'merged_0.007142[test 3].csv' -> '0.007142'
    It focuses on the numerical part assumed to be the curvature.
    """
    base_name = os.path.basename(filename)
    match = re.match(r".*?_([0-9][0-9._]*)(?:\[test.*|\.csv)", base_name, re.IGNORECASE)
    if match:
        curvature_str = match.group(1)
        return curvature_str.replace('_', '.')
    else:
        simple_match_start = re.match(r"([0-9][0-9._]*)(?:\[test.*|\.csv)", base_name, re.IGNORECASE)
        if simple_match_start:
             return simple_match_start.group(1).replace('_', '.')
        print(f"    Warning: Could not accurately parse curvature group from filename: {base_name}. Using full filename stem as group. Review parsing logic if this happens frequently.")
        return os.path.splitext(base_name)[0]

# =================================================================================
# --- Function: Generate Performance Plots ---
# =================================================================================
def generate_performance_plots(detailed_predictions_df, output_dir, model_name_prefix="xgb"):
    if detailed_predictions_df.empty:
        print("  No detailed prediction data to generate plots.")
        return

    print("\n  Generating performance plots...")
    plot_filename_prefix = f"{model_name_prefix}_"

    # Plot 1: Predicted vs. True Curvature
    plt.figure(figsize=(9, 8))
    try:
        sns.scatterplot(x='true_curvature', y='predicted_curvature', hue='tested_group_for_row',
                        data=detailed_predictions_df, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
        min_val_data = min(detailed_predictions_df['true_curvature'].min(), detailed_predictions_df['predicted_curvature'].min())
        max_val_data = max(detailed_predictions_df['true_curvature'].max(), detailed_predictions_df['predicted_curvature'].max())
        padding = (max_val_data - min_val_data) * 0.1 if (max_val_data > min_val_data) else 0.1
        plot_min = min_val_data - padding
        plot_max = max_val_data + padding

        plt.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', lw=2, label="Ideal (y=x)")
        plt.xlabel("True Curvature ($m^{-1}$)")
        plt.ylabel("Predicted Curvature ($m^{-1}$)")
        plt.title(f"Predicted vs. True Curvature ({model_name_prefix.upper()})")
        plt.legend(title="Tested Curvature Group", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle=':', alpha=0.7)
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
            # Attempt to sort groups numerically if possible, otherwise alphabetically
            unique_groups_for_plot = detailed_predictions_df['tested_group_for_row'].unique()
            try:
                sorted_groups = sorted(unique_groups_for_plot, key=lambda x: float(str(x).replace('_','.')))
            except ValueError:
                sorted_groups = sorted(unique_groups_for_plot, key=str) # Fallback to string sort
            
            sns.boxplot(x='tested_group_for_row', y='abs_error', data=detailed_predictions_df, order=sorted_groups, palette="viridis")
            plt.xlabel("Tested Curvature Group (Original $mm^{-1}$ labels used for grouping)")
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

# =================================================================================
# --- Main XGBoost Training and Evaluation Function ---
# =================================================================================
def train_xgboost_for_combined_sessions(
    input_files_for_run, # List of all files from combined sessions
    run_label,           # Descriptive label for the run (e.g., "LOGO_CV_Combined_Sessions_1_2_3_4")
    output_dir_for_run
):
    all_data_segments = []
    successful_baseline_count = 0
    files_without_baseline = []

    print(f"Starting XGBoost data preparation for {len(input_files_for_run)} files for run: {run_label}...")

    for file_path in input_files_for_run:
        base_filename = os.path.basename(file_path)
        print(f"\n  Processing file: {base_filename}")
        try:
            df_cleaned_input = pd.read_csv(file_path)
            if df_cleaned_input.empty:
                print(f"    Skipping empty pre-cleaned file: {base_filename}")
                files_without_baseline.append(f"{base_filename} (File was empty)")
                continue
        except Exception as e:
            print(f"    Error reading pre-cleaned file {base_filename}: {e}. Skipping.")
            files_without_baseline.append(f"{base_filename} (Read error: {e})")
            continue

        # --- Baseline FFT Subtraction ---
        idle_fft_baseline = None
        if not all(col in df_cleaned_input.columns for col in [POSITION_COLUMN, ACTIVITY_COLUMN] + FFT_COLUMNS + [ORIGINAL_TARGET_COLUMN]):
            print(f"    Skipping {base_filename}: missing one or more required columns.")
            files_without_baseline.append(f"{base_filename} (Missing required columns)")
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
                print(f"    Calculated idle baseline from {len(idle_candidate_rows)} rows for {base_filename}.")
                successful_baseline_count += 1
            else:
                print(f"    WARNING: In '{base_filename}', no 'Curvature_Active == 0' rows found before first 'Position_cm == 0.0'.")
                files_without_baseline.append(f"{base_filename} (No CA=0 before first PosCM=0)")
        else:
            print(f"    WARNING: In '{base_filename}', no 'Position_cm == 0.0' found.")
            files_without_baseline.append(f"{base_filename} (No PosCM=0 found)")

        active_df = df_cleaned_input[df_cleaned_input[ACTIVITY_COLUMN] == 1].copy()
        if active_df.empty:
            print(f"    No active data (Curvature_Active == 1) in {base_filename}. Skipping.")
            continue
        print(f"    Found {len(active_df)} active rows.")

        if idle_fft_baseline is not None:
            active_df.loc[:, FFT_COLUMNS] = active_df[FFT_COLUMNS].values - idle_fft_baseline
            print(f"    Normalized FFT for {len(active_df)} active rows using baseline subtraction.")
        else:
            print(f"    Active data for {base_filename} NOT baseline-subtracted (no suitable idle baseline).")
        
        # --- Unit Conversion for Target ---
        active_df[TARGET_COLUMN_M_INV] = active_df[TARGET_COLUMN_MM_INV] * 1000 # mm^-1 to m^-1
        print(f"    Converted target '{ORIGINAL_TARGET_COLUMN}' (mm^-1) to '{TARGET_COLUMN_M_INV}' (m^-1).")

        group_id = parse_filename_for_curvature_group(base_filename) # Uses revised parser
        active_df['group'] = group_id
        all_data_segments.append(active_df)

    print(f"\nBaseline Calculation & Preprocessing Summary for {len(input_files_for_run)} files:")
    print(f"  Successfully calculated idle baseline for {successful_baseline_count} files.")
    if files_without_baseline:
        print(f"  Files with warnings/issues during baseline/preprocessing:")
        for f_issue in files_without_baseline:
            print(f"    - {f_issue}")

    if not all_data_segments:
        print("No data available for XGBoost training after processing all files.")
        return

    combined_df = pd.concat(all_data_segments, ignore_index=True)
    if combined_df.empty:
        print("Combined DataFrame is empty. Cannot proceed with XGBoost.")
        return

    print(f"\nCombined preprocessed data shape for XGBoost: {combined_df.shape}")
    print(f"Unique curvature groups for LOGO CV: {sorted(combined_df['group'].unique())}")

    X = combined_df[FEATURE_COLUMNS]
    y = combined_df[TARGET_COLUMN_M_INV] # Target is now in m^-1
    groups = combined_df['group']
    unique_group_ids = groups.unique()

    model_name_prefix = f"xgb_{run_label}" # For naming output files

    if len(unique_group_ids) < 2:
        print(f"Only {len(unique_group_ids)} unique curvature group(s) found: {unique_group_ids}. LeaveOneGroupOut CV requires at least 2 groups. CV will be skipped.")
        return

    logo = LeaveOneGroupOut()
    fold_metrics = []
    all_fold_predictions_data = []
    fold_num = 0

    print(f"\nStarting Leave-One-Curvature-Profile-Out CV for run '{run_label}' using {len(unique_group_ids)} groups...")

    for train_idx, test_idx in logo.split(X, y, groups):
        fold_num += 1
        X_train_fold, X_test_fold = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train_fold, y_test_fold = y.iloc[train_idx].copy(), y.iloc[test_idx].copy() # y is in m^-1

        current_test_group_curvature = groups.iloc[test_idx].unique()[0]
        print(f"\n--- Fold {fold_num}: Testing on Curvature Group '{current_test_group_curvature}' ---")
        print(f"  Train set size: {len(X_train_fold)}, Test set size: {len(X_test_fold)}")

        # 1. Scale Input Features (X)
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_fold)
        X_test_scaled = scaler_X.transform(X_test_fold)

        # 2. Create Evaluation Set for XGBoost Early Stopping (from this fold's training data)
        X_sub_train, X_eval, y_sub_train, y_eval = train_test_split(
            X_train_scaled, y_train_fold, test_size=0.2, random_state=42
        )

        # 3. Create and Train XGBoost Model
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror', # For regression
            n_estimators=500,             # Number of trees (can be tuned)
            learning_rate=0.05,           # Step size shrinkage (can be tuned)
            max_depth=5,                  # Max depth of a tree (can be tuned)
            subsample=0.8,                # Subsample ratio of the training instance
            colsample_bytree=0.8,         # Subsample ratio of columns when constructing each tree
            random_state=42,
            n_jobs=-1,                    # Use all available cores
            early_stopping_rounds=50      # MOVED HERE
        )

        print(f"  Training XGBoost on {X_sub_train.shape[0]} samples, evaluating on {X_eval.shape[0]} samples for early stopping...")
        start_time_fold_fit = time.time()
        xgb_model.fit(X_sub_train, y_sub_train,
                      eval_set=[(X_eval, y_eval)], # Ensure eval_set is provided for early stopping
                      verbose=False) # Set to True or a number to see training progress
        
        fit_time_fold = time.time() - start_time_fold_fit
        # Access best_iteration if early stopping was triggered, otherwise it might not be directly available
        # For XGBoost >= 1.6, best_iteration_ is available after fitting if early stopping is used.
        # For older versions or if not triggered, it might be len(xgb_model.get_booster().get_dump())
        best_iter_val = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') and xgb_model.best_iteration is not None else xgb_model.n_estimators
        print(f"  XGBoost fitting completed in {fit_time_fold:.2f}s. Best iteration: {best_iter_val}")

        # 4. Predict
        y_pred_m_inv_fold = xgb_model.predict(X_test_scaled)

        # 5. Evaluate
        mse = mean_squared_error(y_test_fold, y_pred_m_inv_fold)
        mae = mean_absolute_error(y_test_fold, y_pred_m_inv_fold)
        r2 = r2_score(y_test_fold, y_pred_m_inv_fold)
        rmse = np.sqrt(mse)

        fold_metrics.append({
            'fold': fold_num, 'tested_group': current_test_group_curvature,
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'fitting_time_s': fit_time_fold,
            'best_iteration': best_iter_val
        })
        print(f"  Fold {fold_num} Results (units: m^-1): MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.4f}")

        fold_pred_df = pd.DataFrame({
            'true_curvature': y_test_fold.values.flatten(),
            'predicted_curvature': y_pred_m_inv_fold.flatten(),
            'error': y_test_fold.values.flatten() - y_pred_m_inv_fold.flatten(),
            'abs_error': np.abs(y_test_fold.values.flatten() - y_pred_m_inv_fold.flatten()),
            'fold_number': fold_num,
            'tested_group_for_row': current_test_group_curvature
        })
        for col in X_test_fold.columns: # Add original features if needed
             if col not in fold_pred_df.columns:
                 fold_pred_df[col] = X_test_fold[col].values
        all_fold_predictions_data.append(fold_pred_df)

    print("\n--- Cross-Validation Finished ---")
    if fold_metrics:
        results_df = pd.DataFrame(fold_metrics)
        print(f"\nCross-Validation Results Summary (LOGO CV for run '{run_label}'):")
        columns_to_print = ['fold', 'tested_group', 'mse', 'rmse', 'mae', 'r2', 'fitting_time_s', 'best_iteration']
        with pd.option_context('display.max_colwidth', None, 'display.width', 1000):
             print(results_df[columns_to_print])
        print(f"\nAverage CV MSE: {results_df['mse'].mean():.6f} (m^-1)^2")
        print(f"Average CV RMSE: {results_df['rmse'].mean():.6f} m^-1")
        print(f"Average CV MAE: {results_df['mae'].mean():.6f} m^-1")
        print(f"Average CV R2 Score: {results_df['r2'].mean():.4f}")

        output_csv_path = os.path.join(output_dir_for_run, f"{model_name_prefix}_LOGO_cv_results.csv")
        results_df.to_csv(output_csv_path, index=False)
        print(f"CV results saved to {output_csv_path}")

    if all_fold_predictions_data:
        detailed_predictions_df = pd.concat(all_fold_predictions_data, ignore_index=True)
        detailed_output_csv_path = os.path.join(output_dir_for_run, f"{model_name_prefix}_detailed_predictions.csv")
        detailed_predictions_df.to_csv(detailed_output_csv_path, index=False)
        print(f"Detailed predictions and errors saved to {detailed_output_csv_path}")
        generate_performance_plots(detailed_predictions_df, output_dir_for_run, model_name_prefix)

    # --- Training Final Model on All Data for the Run ---
    print(f"\n--- Training Final XGBoost Model on All Processed Data for run '{run_label}' ---")
    if not X.empty and not y.empty:
        final_scaler_X = StandardScaler()
        X_scaled_final = final_scaler_X.fit_transform(X)
        # y (target) is used directly in its m^-1 scale for XGBoost

        X_train_all, X_val_final, y_train_all, y_val_final = train_test_split(
            X_scaled_final, y, test_size=0.15, random_state=42
        )

        final_xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000, 
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50  # MOVED HERE for the final model as well
        )
        
        print(f"Fitting final XGBoost model on {X_train_all.shape[0]} samples, evaluating on {X_val_final.shape[0]}...")
        start_time_final = time.time()
        final_xgb_model.fit(X_train_all, y_train_all,
                            eval_set=[(X_val_final, y_val_final)], # Ensure eval_set for early stopping
                            verbose=False)
        fit_time_final = time.time() - start_time_final
        final_best_iter_val = final_xgb_model.best_iteration if hasattr(final_xgb_model, 'best_iteration') and final_xgb_model.best_iteration is not None else final_xgb_model.n_estimators
        print(f"Final XGBoost model fitted in {fit_time_final:.2f}s. Best iteration: {final_best_iter_val}")

        model_save_path = os.path.join(output_dir_for_run, f"{model_name_prefix}_final_model.joblib")
        scaler_X_save_path = os.path.join(output_dir_for_run, f"{model_name_prefix}_final_scaler_X.joblib")

        joblib.dump(final_xgb_model, model_save_path)
        joblib.dump(final_scaler_X, scaler_X_save_path)
        print(f"Final XGBoost model saved to: {model_save_path}")
        print(f"Final X Scaler saved to: {scaler_X_save_path}")
    else:
        print("Skipping final model training as combined data is empty.")

# =================================================================================
# --- Main Execution (Combined Sessions for LOGO CV) ---
# =================================================================================
if __name__ == "__main__":
    # === OPERATOR: DEFINE ALL TARGET TEST SESSION NUMBERS TO COMBINE HERE ===
    ALL_SESSION_NUMS_TO_PROCESS = ["1", "2", "3", "4"]
    # Example: To process only sessions 1 and 2: ALL_SESSION_NUMS_TO_PROCESS = ["1", "2"]
    # Example: To process a single session: ALL_SESSION_NUMS_TO_PROCESS = ["1"]
    # =================================================================================

    script_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_DATA_DIR = os.path.join(script_dir, "csv_data", "cleaned")
    
    OUTPUT_RESULTS_BASE_DIR_XGB = os.path.join(script_dir, "xgboost_model_outputs")
    
    if len(ALL_SESSION_NUMS_TO_PROCESS) > 1:
        run_descriptor = f"LOGO_CV_Combined_Sessions_{'_'.join(ALL_SESSION_NUMS_TO_PROCESS)}"
    elif len(ALL_SESSION_NUMS_TO_PROCESS) == 1:
        run_descriptor = f"LOGO_CV_Session_{ALL_SESSION_NUMS_TO_PROCESS[0]}"
    else:
        print("Error: No sessions specified in ALL_SESSION_NUMS_TO_PROCESS. Exiting.")
        exit()
        
    FINAL_OUTPUT_DIR_FOR_RUN = os.path.join(OUTPUT_RESULTS_BASE_DIR_XGB, run_descriptor)
    
    if not os.path.exists(FINAL_OUTPUT_DIR_FOR_RUN):
        os.makedirs(FINAL_OUTPUT_DIR_FOR_RUN)
        print(f"Created output directory for this run: {FINAL_OUTPUT_DIR_FOR_RUN}")

    all_files_to_process_for_this_run = []
    if not os.path.isdir(INPUT_DATA_DIR):
        print(f"Error: Input directory '{INPUT_DATA_DIR}' not found. Cannot proceed.")
    else:
        print(f"Attempting to collect files for sessions: {', '.join(ALL_SESSION_NUMS_TO_PROCESS)} from {INPUT_DATA_DIR}")
        for session_num_str_to_collect in ALL_SESSION_NUMS_TO_PROCESS:
            target_session_tag = f"[test {session_num_str_to_collect}]".lower()
            
            session_specific_files = [
                f for f in glob.glob(os.path.join(INPUT_DATA_DIR, "cleaned_*.csv"))
                if target_session_tag in os.path.basename(f).lower()
            ]
            
            if session_specific_files:
                print(f"  Found {len(session_specific_files)} files for session '[test {session_num_str_to_collect}]'.")
                all_files_to_process_for_this_run.extend(session_specific_files)
            else:
                print(f"  Warning: No 'cleaned_*.csv' files found for session '[test {session_num_str_to_collect}]' in '{INPUT_DATA_DIR}'.")
    
    if not all_files_to_process_for_this_run:
        print(f"No 'cleaned_*.csv' files found for any of the specified sessions. XGBoost training cannot proceed.")
    else:
        unique_files_to_process = sorted(list(set(all_files_to_process_for_this_run)))
        print(f"\nFound a total of {len(unique_files_to_process)} unique 'cleaned_*.csv' file(s) from the specified session(s) to process for XGBoost training:")
        for f_path in unique_files_to_process:
            print(f"  - {os.path.basename(f_path)}")
        
        model_and_file_prefix_label = run_descriptor

        train_xgboost_for_combined_sessions(
            unique_files_to_process, 
            model_and_file_prefix_label, 
            FINAL_OUTPUT_DIR_FOR_RUN
        )
    
    print(f"\nXGBoost Script finished processing for run: {run_descriptor}.")

