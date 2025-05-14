"""
XGBoost_feature_importance_80_20.py - XGBoost for Curvature Sensor Data
to determine feature importances using an 80/20 random split.

Process Overview:
1. Loads pre-cleaned CSV files from specified test sessions.
2. Combines data from all specified sessions.
3. Normalizes FFT readings by subtracting a baseline calculated from "idle" periods.
4. Converts target curvature units from mm^-1 to m^-1.
5. Performs a single 80/20 random train-test split on the combined data.
6. Standardizes input features (FFT + Position) - fit on train, transform train & test.
7. Trains an XGBoost model on the 80% training data, using a portion of it
   for early stopping.
8. Evaluates the model on the 20% test set.
9. Extracts and plots feature importances from the trained XGBoost model.
"""

import pandas as pd
import numpy as np
import os
import glob
import re
import time
from sklearn.model_selection import train_test_split
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
ORIGINAL_TARGET_COLUMN = 'Curvature' # This is in mm^-1
ACTIVITY_COLUMN = 'Curvature_Active'
TIMESTAMP_COLUMN = 'Timestamp' # Present in files, not used as a direct feature

FEATURE_COLUMNS = FFT_COLUMNS + [POSITION_COLUMN]
TARGET_COLUMN_MM_INV = ORIGINAL_TARGET_COLUMN
TARGET_COLUMN_M_INV = 'Curvature_m_inv' # Target for model training, in m^-1

# =================================================================================
# --- Function: Parse Filename for Curvature Group (for potential detailed analysis) ---
# =================================================================================
def parse_filename_for_curvature_group(filename):
    """
    Extracts a group identifier (curvature value string) from filenames like
    'cleaned_0_01[test 1].csv' -> '0.01'
    It focuses on the numerical part assumed to be the curvature, ignoring session tags.
    """
    base_name = os.path.basename(filename)
    # Regex to capture the numerical part (dots or underscores) assumed to be curvature
    # It looks for a number that is typically between an underscore and the session tag or end of filename.
    match = re.match(r".*?_([0-9][0-9._]*)(?:\[test.*|\.csv)", base_name, re.IGNORECASE)
    if match:
        curvature_str = match.group(1)
        return curvature_str.replace('_', '.') # Normalize underscores to dots
    else:
        # Fallback if the primary pattern fails
        simple_match_start = re.match(r"([0-9][0-9._]*)(?:\[test.*|\.csv)", base_name, re.IGNORECASE)
        if simple_match_start:
             return simple_match_start.group(1).replace('_', '.')
        print(f"    Warning: Could not accurately parse curvature group from filename: {base_name}. Using full filename stem as group. Review parsing logic if this happens frequently.")
        return os.path.splitext(base_name)[0]

# =================================================================================
# --- Function: Plot Feature Importances ---
# =================================================================================
def plot_feature_importances(importances, feature_names, output_dir, model_name_prefix="xgb"):
    """Plots feature importances from an XGBoost model."""
    if len(importances) == 0 or len(feature_names) == 0:
        print("  No feature importances to plot.")
        return

    print("\n  Generating feature importance plot...")
    plot_filename_prefix = f"{model_name_prefix}_"

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)

    plt.figure(figsize=(10, max(6, len(feature_names) * 0.35))) # Dynamic height
    sns.barplot(x='importance', y='feature', data=importance_df, palette="viridis")
    plt.title(f'Feature Importances ({model_name_prefix.upper()})')
    plt.xlabel('Importance Score (Gain)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{plot_filename_prefix}feature_importances.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"    Saved: Feature Importance plot ({plot_path})")

# =================================================================================
# --- Function: Generate Performance Plots (Adapted for general use) ---
# =================================================================================
def generate_performance_plots(detailed_predictions_df, output_dir, model_name_prefix="xgb", hue_column='original_group'):
    if detailed_predictions_df.empty:
        print("  No detailed prediction data to generate plots.")
        return

    print("\n  Generating performance plots...")
    plot_filename_prefix = f"{model_name_prefix}_"
    
    # Ensure hue_column exists, otherwise plot without hue
    if hue_column not in detailed_predictions_df.columns:
        print(f"    Warning: Hue column '{hue_column}' not found in detailed_predictions_df. Plots will not be colored by group.")
        hue_column = None # Disable hue if column not found

    # Plot 1: Predicted vs. True Curvature
    plt.figure(figsize=(9, 8))
    try:
        sns.scatterplot(x='true_curvature', y='predicted_curvature', 
                        hue=detailed_predictions_df[hue_column] if hue_column else None,
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
        if hue_column:
            plt.legend(title="Original Curvature Group", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        else:
            plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xlim(plot_min, plot_max)
        plt.ylim(plot_min, plot_max)
        plt.tight_layout(rect=[0, 0, 0.85 if hue_column else 0.95, 1])
        plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}pred_vs_true.png"), bbox_inches='tight')
        plt.close()
        print(f"    Saved: Predicted vs. True plot")
    except Exception as e:
        print(f"    Error generating Predicted vs. True plot: {e}")

    # Plot 2: Residuals vs. True Curvature
    plt.figure(figsize=(9, 6))
    try:
        sns.scatterplot(x='true_curvature', y='error', 
                        hue=detailed_predictions_df[hue_column] if hue_column else None,
                        data=detailed_predictions_df, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
        plt.axhline(0, color='k', linestyle='--', lw=2)
        plt.xlabel("True Curvature ($m^{-1}$)")
        plt.ylabel("Residual (True - Predicted) ($m^{-1}$)")
        plt.title(f"Residuals vs. True Curvature ({model_name_prefix.upper()})")
        if hue_column:
            plt.legend(title="Original Curvature Group", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        else:
            plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.85 if hue_column else 0.95, 1])
        plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}residuals_vs_true.png"), bbox_inches='tight')
        plt.close()
        print(f"    Saved: Residuals vs. True plot")
    except Exception as e:
        print(f"    Error generating Residuals vs. True plot: {e}")

    # Plot 3: Absolute Error Distribution (Box Plot by Group)
    if 'abs_error' in detailed_predictions_df.columns and hue_column and hue_column in detailed_predictions_df.columns:
        plt.figure(figsize=(12, 7))
        try:
            unique_groups_for_plot = detailed_predictions_df[hue_column].unique()
            try:
                sorted_groups = sorted(unique_groups_for_plot, key=lambda x: float(str(x).replace('_','.')))
            except ValueError:
                sorted_groups = sorted(unique_groups_for_plot, key=str) 
            
            sns.boxplot(x=hue_column, y='abs_error', data=detailed_predictions_df, order=sorted_groups, palette="viridis")
            plt.xlabel("Original Curvature Group (mm$^{-1}$ labels)")
            plt.ylabel("Absolute Error ($m^{-1}$)")
            plt.title(f"Absolute Error Distribution by Original Group ({model_name_prefix.upper()})")
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', linestyle=':', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}abs_error_boxplot_by_group.png"))
            plt.close()
            print(f"    Saved: Absolute Error Boxplot by Group")
        except Exception as e:
            print(f"    Error generating Absolute Error Boxplot: {e}")
    elif 'abs_error' in detailed_predictions_df.columns: # Plot without grouping if hue_column is not available
        plt.figure(figsize=(8, 6))
        sns.boxplot(y='abs_error', data=detailed_predictions_df, palette="viridis")
        plt.ylabel("Absolute Error ($m^{-1}$)")
        plt.title(f"Absolute Error Distribution ({model_name_prefix.upper()})")
        plt.grid(True, axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}abs_error_boxplot.png"))
        plt.close()
        print(f"    Saved: Overall Absolute Error Boxplot")


# =================================================================================
# --- Main XGBoost Training, Evaluation, and Feature Importance Function ---
# =================================================================================
def train_evaluate_xgboost_80_20(
    input_files_for_run,
    run_label,
    output_dir_for_run
):
    all_data_segments = []
    # ... (rest of the data loading and preprocessing loop as in the previous XGBoost script) ...
    # Ensure 'group' column is added to active_df using parse_filename_for_curvature_group
    # This is for potential detailed analysis of the test set later.

    print(f"Starting XGBoost data preparation for {len(input_files_for_run)} files for run: {run_label}...")
    successful_baseline_count = 0
    files_without_baseline = []

    for file_path in input_files_for_run:
        base_filename = os.path.basename(file_path)
        # print(f"\n  Processing file: {base_filename}") # Kept it less verbose
        try:
            df_cleaned_input = pd.read_csv(file_path)
            if df_cleaned_input.empty:
                files_without_baseline.append(f"{base_filename} (File was empty)")
                continue
        except Exception as e:
            files_without_baseline.append(f"{base_filename} (Read error: {e})")
            continue

        idle_fft_baseline = None
        required_cols = [POSITION_COLUMN, ACTIVITY_COLUMN] + FFT_COLUMNS + [ORIGINAL_TARGET_COLUMN]
        if not all(col in df_cleaned_input.columns for col in required_cols):
            files_without_baseline.append(f"{base_filename} (Missing required columns: {set(required_cols) - set(df_cleaned_input.columns)})")
            continue
            
        first_pos_zero_indices = df_cleaned_input[df_cleaned_input[POSITION_COLUMN] == 0.0].index
        if not first_pos_zero_indices.empty:
            first_pos_zero_idx = first_pos_zero_indices[0]
            idle_candidate_rows = df_cleaned_input[
                (df_cleaned_input.index < first_pos_zero_idx) & (df_cleaned_input[ACTIVITY_COLUMN] == 0)
            ]
            if not idle_candidate_rows.empty:
                idle_fft_baseline = idle_candidate_rows[FFT_COLUMNS].mean().values
                successful_baseline_count += 1
            else:
                files_without_baseline.append(f"{base_filename} (No CA=0 before first PosCM=0)")
        else:
            files_without_baseline.append(f"{base_filename} (No PosCM=0 found)")

        active_df = df_cleaned_input[df_cleaned_input[ACTIVITY_COLUMN] == 1].copy()
        if active_df.empty:
            continue

        if idle_fft_baseline is not None:
            active_df.loc[:, FFT_COLUMNS] = active_df[FFT_COLUMNS].values - idle_fft_baseline
        else:
            print(f"    WARNING: Active data for {base_filename} NOT baseline-subtracted (no suitable idle baseline). Using raw FFT values.")
        
        active_df[TARGET_COLUMN_M_INV] = active_df[TARGET_COLUMN_MM_INV] * 1000
        
        # Add original curvature group for potential detailed analysis later
        group_id = parse_filename_for_curvature_group(base_filename)
        active_df['original_group'] = group_id 
        
        all_data_segments.append(active_df)

    print(f"\nBaseline Calculation & Preprocessing Summary for {len(input_files_for_run)} files:")
    print(f"  Successfully applied idle baseline for {successful_baseline_count} files.")
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

    X = combined_df[FEATURE_COLUMNS]
    y = combined_df[TARGET_COLUMN_M_INV]
    # Store original group information if present for later use in plotting test set results
    original_groups_for_plotting = combined_df['original_group'] if 'original_group' in combined_df else None


    model_name_prefix = f"xgb_{run_label}_80_20_split"

    print("\nPerforming 80/20 random train-test split...")
    # Ensure indices are split if original_groups_for_plotting is to be used
    indices = np.arange(X.shape[0])
    X_train_full_idx, X_test_idx, y_train_full_idx, y_test_idx = train_test_split(indices, indices, test_size=0.2, random_state=42)

    X_train_full = X.iloc[X_train_full_idx]
    X_test = X.iloc[X_test_idx]
    y_train_full = y.iloc[y_train_full_idx]
    y_test = y.iloc[y_test_idx]
    
    if original_groups_for_plotting is not None:
        test_set_original_groups = original_groups_for_plotting.iloc[X_test_idx]
    else:
        test_set_original_groups = None

    print(f"  Full training set size: {X_train_full.shape[0]}, Test set size: {X_test.shape[0]}")

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_full)
    X_test_scaled = scaler_X.transform(X_test)
    print("  Input Feature (X) Scaling APPLIED.")

    X_train_sub, X_eval, y_train_sub, y_eval = train_test_split(
        X_train_scaled, y_train_full, test_size=0.25, random_state=42 
    )
    print(f"  Sub-training set for XGBoost: {X_train_sub.shape[0]}, Evaluation set for XGBoost: {X_eval.shape[0]}")

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05, 
        max_depth=6,      
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50 # MOVED HERE
    )

    print(f"  Training XGBoost on {X_train_sub.shape[0]} samples, evaluating on {X_eval.shape[0]} samples...")
    start_time_fit = time.time()
    xgb_model.fit(X_train_sub, y_train_sub,
                  eval_set=[(X_eval, y_eval)],
                  # early_stopping_rounds=50, # REMOVED FROM HERE
                  verbose=False) 
    
    fit_time = time.time() - start_time_fit
    print(f"  XGBoost fitting completed in {fit_time:.2f}s. Best iteration: {xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 'N/A (check XGBoost version or if early stopping triggered)'}")

    y_pred_test = xgb_model.predict(X_test_scaled)

    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)

    print("\n--- Evaluation Results on 20% Test Set ---")
    print(f"  MSE: {mse_test:.6f} (m^-1)^2")
    print(f"  RMSE: {rmse_test:.6f} m^-1")
    print(f"  MAE: {mae_test:.6f} m^-1")
    print(f"  R2 Score: {r2_test:.4f}")
    print(f"  Best Iteration: {xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 'N/A'}")
    print(f"  Training time: {fit_time:.2f}s")

    results_summary = {
        'run_label': run_label, 'split_type': '80_20_random',
        'mse': mse_test, 'rmse': rmse_test, 'mae': mae_test, 'r2': r2_test,
        'fitting_time_s': fit_time, 
        'best_iteration': xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else -1,
        'total_samples': len(combined_df),
        'train_samples_80_percent': len(X_train_full),
        'test_samples_20_percent': len(X_test)
    }
    results_df = pd.DataFrame([results_summary])
    summary_csv_path = os.path.join(output_dir_for_run, f"{model_name_prefix}_summary_results.csv")
    results_df.to_csv(summary_csv_path, index=False)
    print(f"Run summary results saved to {summary_csv_path}")
    
    detailed_test_predictions_df = pd.DataFrame({
        'true_curvature': y_test.values.flatten(),
        'predicted_curvature': y_pred_test.flatten(),
        'error': y_test.values.flatten() - y_pred_test.flatten(),
        'abs_error': np.abs(y_test.values.flatten() - y_pred_test.flatten())
    })
    if test_set_original_groups is not None:
        detailed_test_predictions_df['original_group'] = test_set_original_groups.values
    else:
        detailed_test_predictions_df['original_group'] = 'Test_Set_Data'


    detailed_output_csv_path = os.path.join(output_dir_for_run, f"{model_name_prefix}_detailed_test_predictions.csv")
    detailed_test_predictions_df.to_csv(detailed_output_csv_path, index=False)
    print(f"Detailed predictions and errors for test set saved to {detailed_output_csv_path}")
    
    generate_performance_plots(detailed_test_predictions_df, output_dir_for_run, model_name_prefix, hue_column='original_group')

    importances = xgb_model.feature_importances_
    print("\nFeature Importances:")
    for feature, importance in zip(FEATURE_COLUMNS, importances):
        print(f"  {feature}: {importance:.4f}")
    plot_feature_importances(importances, FEATURE_COLUMNS, output_dir_for_run, model_name_prefix)

    final_model_save_path = os.path.join(output_dir_for_run, f"{model_name_prefix}_trained_on_80pct.joblib")
    final_scaler_X_save_path = os.path.join(output_dir_for_run, f"{model_name_prefix}_scaler_X_from_80pct.joblib")
    
    joblib.dump(xgb_model, final_model_save_path)
    joblib.dump(scaler_X, final_scaler_X_save_path)
    print(f"XGBoost model (trained on 80%) saved to: {final_model_save_path}")
    print(f"X Scaler (from 80%) saved to: {final_scaler_X_save_path}")

# =================================================================================
# --- Main Execution (Combined Sessions for 80/20 Split) ---
# =================================================================================
if __name__ == "__main__":
    ALL_SESSION_NUMS_TO_PROCESS = ["1", "2", "3", "4"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_DATA_DIR = os.path.join(script_dir, "csv_data", "cleaned")
    
    OUTPUT_RESULTS_BASE_DIR_XGB = os.path.join(script_dir, "xgboost_model_outputs")
    
    run_descriptor_main = f"Combined_Sessions_{'_'.join(ALL_SESSION_NUMS_TO_PROCESS)}"
    FINAL_OUTPUT_DIR_FEATURE_IMPORTANCE = os.path.join(OUTPUT_RESULTS_BASE_DIR_XGB, f"{run_descriptor_main}_80_20_Split_FeatureImportance")
    
    if not os.path.exists(FINAL_OUTPUT_DIR_FEATURE_IMPORTANCE):
        os.makedirs(FINAL_OUTPUT_DIR_FEATURE_IMPORTANCE)
        print(f"Created output directory for this run: {FINAL_OUTPUT_DIR_FEATURE_IMPORTANCE}")

    all_files_to_combine = []
    if not os.path.isdir(INPUT_DATA_DIR):
        print(f"Error: Input directory '{INPUT_DATA_DIR}' not found. Cannot proceed.")
        exit()
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
                all_files_to_combine.extend(session_specific_files)
            else:
                print(f"  Warning: No 'cleaned_*.csv' files found for session '[test {session_num_str_to_collect}]' in '{INPUT_DATA_DIR}'.")
    
    if not all_files_to_combine:
        print(f"No 'cleaned_*.csv' files found for any of the specified sessions. XGBoost script cannot proceed.")
        exit()
    else:
        unique_files_to_process = sorted(list(set(all_files_to_combine)))
        print(f"\nFound a total of {len(unique_files_to_process)} unique 'cleaned_*.csv' file(s) from the specified session(s) to process.")
        
        train_evaluate_xgboost_80_20(
            unique_files_to_process, 
            run_descriptor_main, 
            FINAL_OUTPUT_DIR_FEATURE_IMPORTANCE
        )
    
    print(f"\nXGBoost Script (80/20 split for feature importance) finished processing for run: {run_descriptor_main}.")

