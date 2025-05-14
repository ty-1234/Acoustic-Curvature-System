import pandas as pd
import numpy as np
import os
import glob
import re
import time # To time GPR fitting
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score

# --- Global Configuration (Adjust these paths as needed) ---
ORIGINAL_DATA_DIR = "csv_data/merged/"      # <<< Path to your ORIGINAL "merged_*.csv" files
PROCESSED_DATA_DIR = "csv_data/interpolated/" # <<< Path to your "interpolated_*.csv" files

fft_columns = ['FFT_200Hz', 'FFT_400Hz', 'FFT_600Hz', 'FFT_800Hz', 
               'FFT_1000Hz', 'FFT_1200Hz', 'FFT_1400Hz', 'FFT_1600Hz', 
               'FFT_1800Hz', 'FFT_2000Hz']
position_column = 'Position_cm'
input_features_all = fft_columns + [position_column]
target_column = 'Curvature'
activity_column = 'Curvature_Active' # As recalculated by interpolator_feature.py

def parse_filename_details(filename):
    base_name = os.path.basename(filename)
    match = re.match(r"(interpolated|merged)_([0-9]+(?:_[0-9]+)?)\s*\[test\s*(\d+)\]\.csv", base_name, re.IGNORECASE)
    if match:
        object_id_str = match.group(2)
        run_id = int(match.group(3))
        return object_id_str, run_id
    else:
        print(f"Warning: Could not parse object_id/run_id from filename: {base_name}")
        return None, None

def load_and_combine_data(processed_data_dir, original_data_dir, 
                          processed_file_pattern="interpolated_*.csv", 
                          runs_to_include=None, 
                          fft_col_names=None):
    if runs_to_include is None:
        runs_to_include = [1, 2] 
    if fft_col_names is None:
        fft_col_names = fft_columns

    all_processed_files = glob.glob(os.path.join(processed_data_dir, processed_file_pattern))
    df_list = []
    original_first_row_ffts_dict = {} 

    print(f"Looking for processed files matching '{processed_file_pattern}' in '{processed_data_dir}' for runs: {runs_to_include}")

    for processed_f_path in all_processed_files:
        object_id, run_id = parse_filename_details(processed_f_path)
        
        if object_id is not None and run_id is not None and run_id in runs_to_include:
            try:
                temp_df = pd.read_csv(processed_f_path)
                if temp_df.empty:
                    print(f"Warning: Processed file {processed_f_path} is empty. Skipping.")
                    continue
                
                temp_df['object_id'] = object_id
                temp_df['run_id'] = run_id
                df_list.append(temp_df)
                
                original_filename = f"merged_{object_id}[test {run_id}].csv"
                original_f_path = os.path.join(original_data_dir, original_filename)

                if (object_id, run_id) not in original_first_row_ffts_dict: # Load baseline only once per original run
                    if os.path.exists(original_f_path):
                        original_temp_df_first_row = pd.read_csv(original_f_path, nrows=1) 
                        if not original_temp_df_first_row.empty:
                            if all(col in original_temp_df_first_row.columns for col in fft_col_names):
                                first_row_ffts = original_temp_df_first_row.iloc[0][fft_col_names].copy()
                                original_first_row_ffts_dict[(object_id, run_id)] = first_row_ffts
                            else:
                                print(f"Warning: Not all FFT columns found in first row of {original_f_path}. Baseline incomplete for this run.")
                        else:
                            print(f"Warning: Original file {original_f_path} for baseline was empty.")
                    else:
                        print(f"CRITICAL Warning: Original file {original_f_path} for FFT baseline not found! Ensure 'ORIGINAL_DATA_DIR' is correct and originals exist.")
                # print(f"  Loaded and tagged processed: {os.path.basename(processed_f_path)} (Object: {object_id}, Run: {run_id})") # Verbose
            except Exception as e:
                print(f"Error loading or processing file {processed_f_path} or its original for baseline: {e}")
        
    if not df_list:
        print(f"No processed data loaded for runs {runs_to_include}. Please check path and file names.")
        return pd.DataFrame(), {} # Return empty dict for baselines
        
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Total combined processed data: {combined_df.shape[0]} rows from {len(df_list)} files.")
    return combined_df, original_first_row_ffts_dict

def get_fold_specific_fft_baseline(train_fold_metadata_df, all_original_first_row_ffts_dict, fft_column_names_list):
    baselines_to_average = []
    unique_runs_in_train_fold = train_fold_metadata_df[['object_id', 'run_id']].drop_duplicates()

    for _, row in unique_runs_in_train_fold.iterrows():
        run_key = (row['object_id'], row['run_id']) # Key is (object_id_str, run_id_int)
        if run_key in all_original_first_row_ffts_dict:
            baselines_to_average.append(all_original_first_row_ffts_dict[run_key])
        else:
            print(f"Warning: Original first row FFT data not found for run key {run_key} in training fold baselines dict.")

    if not baselines_to_average:
        print("CRITICAL Warning: No original baseline FFTs found for this training fold. Using fallback (ones). FFT Normalization will be ineffective.")
        return pd.Series(1.0, index=fft_column_names_list) # Results in no change by division

    avg_baseline = pd.DataFrame(baselines_to_average).mean(axis=0) # Average across the collected first-row Series
    avg_baseline[avg_baseline <= 1e-9] = 1e-9 # Handle potential zeros before division
    print(f"  Calculated average FFT baseline for this fold from {len(baselines_to_average)} original first rows.")
    return avg_baseline

def normalize_fft_features_fold(X_df, fft_cols_list, fold_fft_baseline_series):
    X_norm_df = X_df.copy()
    for col in fft_cols_list:
        if col in fold_fft_baseline_series and fold_fft_baseline_series[col] != 0:
            X_norm_df[col] = X_df[col] / fold_fft_baseline_series[col]
        else: # Should be rare if baseline handling is correct
            print(f"Warning: FFT baseline for {col} is zero or missing for this fold. Skipping normalization for this column.")
    return X_norm_df

# --- Main GPR Training and Evaluation Logic ---
if __name__ == "__main__":
    development_df, all_original_first_row_ffts = load_and_combine_data(
        PROCESSED_DATA_DIR,
        ORIGINAL_DATA_DIR, 
        processed_file_pattern="interpolated_*.csv", 
        runs_to_include=[1, 2],
        fft_col_names=fft_columns
    )

    if development_df.empty:
        exit("No development data loaded. Exiting.")

    print(f"\nOriginal development_df size: {len(development_df)}")
    development_df_filtered = development_df[development_df[activity_column] == 1].copy()
    print(f"Filtered development_df size (where final '{activity_column}' == 1): {len(development_df_filtered)}")

    if development_df_filtered.empty:
        exit(f"No data after filtering by '{activity_column}' == 1. Exiting.")
    
    groups = development_df_filtered['object_id']
    logo = LeaveOneGroupOut()
    
    n_splits = logo.get_n_splits(groups=groups)
    if n_splits < 2 and len(development_df_filtered['object_id'].unique()) > 1 :
         print(f"Warning: LeaveOneGroupOut may not be effective with only {n_splits} split(s) for >1 unique objects. Consider if 'object_id' grouping is as intended for LOOOCV.")
    elif len(development_df_filtered['object_id'].unique()) == 0: # Should be caught by development_df.empty
         exit("Error: No unique object_ids found. Cannot perform LOOOCV.")
    elif n_splits == 0 and len(development_df_filtered['object_id'].unique()) > 0 : # n_splits could be 0 if only one group
         exit("Error: LeaveOneGroupOut resulted in 0 splits, likely only one object group present in filtered data. LOOOCV requires at least 2 groups.")


    print(f"\nStarting Leave-One-Object-Out Cross-Validation on {n_splits} object group(s)...")
    
    fold_metrics = []

    for fold_num, (train_idx, test_idx) in enumerate(logo.split(X=development_df_filtered, y=development_df_filtered[target_column], groups=groups)):
        train_df_fold = development_df_filtered.iloc[train_idx]
        test_df_fold = development_df_filtered.iloc[test_idx]

        if train_df_fold.empty or test_df_fold.empty:
            print(f"Warning: Fold {fold_num + 1} resulted in empty train or test set. Skipping fold.")
            print(f"  Tested on Object IDs: {development_df_filtered.iloc[test_idx]['object_id'].unique() if not test_df_fold.empty else 'N/A'}")
            continue

        X_train_fold = train_df_fold[input_features_all].copy()
        y_train_fold = train_df_fold[target_column].copy()
        X_test_fold = test_df_fold[input_features_all].copy()
        y_test_fold_original = test_df_fold[target_column].copy() 

        X_train_fold_metadata = train_df_fold[['object_id', 'run_id']].copy()

        print(f"\n--- Fold {fold_num + 1} (Testing on Object IDs: {test_df_fold['object_id'].unique()}) ---")
        print(f"  Training data shape for fold: {X_train_fold.shape}")
        print(f"  Test data shape for fold: {X_test_fold.shape}")

        fold_fft_baseline = get_fold_specific_fft_baseline(X_train_fold_metadata, all_original_first_row_ffts, fft_columns)
        
        X_train_fold_scaled_fft = normalize_fft_features_fold(X_train_fold, fft_columns, fold_fft_baseline)
        X_test_fold_scaled_fft = normalize_fft_features_fold(X_test_fold, fft_columns, fold_fft_baseline)
        
        pos_scaler = StandardScaler()
        # Fit and transform on training data
        X_train_fold_scaled_fft[position_column] = pos_scaler.fit_transform(X_train_fold_scaled_fft[[position_column]])
        # Only transform test data
        X_test_fold_scaled_fft[position_column] = pos_scaler.transform(X_test_fold_scaled_fft[[position_column]])
        print(f"  '{position_column}' standardized for fold.")
        
        X_train_fold_final = X_train_fold_scaled_fft
        X_test_fold_final = X_test_fold_scaled_fft

        target_scaler = StandardScaler()
        y_train_fold_scaled = target_scaler.fit_transform(y_train_fold.values.reshape(-1, 1)).ravel()
        print(f"  Target '{target_column}' standardized for fold training.")
        
        # === GPR Model Training, Prediction, and Evaluation ===
        print(f"  Initializing GPR model for fold {fold_num + 1}...")
        
        length_scale_bounds_val = (1e-2, 1e3) 
        alpha_bounds_val = (1e-2, 1e3) 
        noise_level_init = 0.01 # Initial guess for noise, can be optimized
        noise_level_bounds_val = (1e-5, 1e1) # Allow noise to be very small or moderately large

        # Kernel suggested by your paper: Rational Quadratic, often combined with others.
        # C() is a ConstantKernel for signal variance.
        # RationalQuadratic is good for functions that vary on multiple scales.
        # WhiteKernel models observation noise.
        kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(
                        length_scale=1.0, # Initial length_scale
                        alpha=0.1,        # Initial alpha for RQ (shape parameter)
                        length_scale_bounds=length_scale_bounds_val, 
                        alpha_bounds=alpha_bounds_val
                    ) + WhiteKernel(
                        noise_level=noise_level_init, 
                        noise_level_bounds=noise_level_bounds_val
                    )
        
        gpr_model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=9, # Number of times optimizer is restarted for hyperparameter tuning
            alpha=1e-10,            # Small value added to diagonal for numerical stability during fitting
            random_state=fold_num   # For reproducibility of hyperparameter optimization
        )
        
        print(f"  Fitting GPR for fold {fold_num + 1} (Data size: {X_train_fold_final.shape[0]})...")
        start_time = time.time()
        try:
            gpr_model.fit(X_train_fold_final, y_train_fold_scaled)
            end_time = time.time()
            fit_time = end_time - start_time
            print(f"  GPR fitting complete. Time taken: {fit_time:.2f} seconds.")
            print(f"  Learned kernel for fold {fold_num + 1}: {gpr_model.kernel_}")

            print(f"  Predicting with GPR for fold {fold_num + 1}...")
            y_pred_scaled, y_pred_std_scaled = gpr_model.predict(X_test_fold_final, return_std=True)
            
            # Inverse transform predictions to original curvature scale
            y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
            # Note: y_pred_std_scaled is the std dev in the *scaled* space. 
            # To convert it to original scale: std_original = y_pred_std_scaled * target_scaler.scale_[0]
            
            # Evaluate the model
            mse = mean_squared_error(y_test_fold_original, y_pred_original)
            r2 = r2_score(y_test_fold_original, y_pred_original)
            
            print(f"  Fold {fold_num + 1} Test MSE: {mse:.6f}")
            print(f"  Fold {fold_num + 1} Test R2 Score: {r2:.6f}")
            
            fold_metrics.append({
                'fold': fold_num + 1, 
                'object_tested': test_df_fold['object_id'].unique()[0] if not test_df_fold.empty else 'N/A', 
                'mse': mse, 
                'r2': r2,
                'fitting_time_s': fit_time,
                'learned_kernel': str(gpr_model.kernel_)
            })
        except Exception as e:
            print(f"ERROR during GPR fit/predict for fold {fold_num + 1}: {e}")
            fold_metrics.append({
                'fold': fold_num + 1, 
                'object_tested': test_df_fold['object_id'].unique()[0] if not test_df_fold.empty else 'N/A', 
                'mse': np.nan, 'r2': np.nan,
                'fitting_time_s': 0,
                'learned_kernel': 'Error'
            })
        # === End of GPR Code ===
        
        if fold_num == 0: 
            print("\n  --- Example: First Fold - Predictions (Head) ---")
            print("  y_test_fold_original (head):", y_test_fold_original.head().values)
            print("  y_pred_original (head):", y_pred_original[:5])
            print("  ---")
        
        print(f"--- End of Fold {fold_num + 1} ---")

    print("\n--- Cross-Validation Finished ---")
    if fold_metrics:
        results_df = pd.DataFrame(fold_metrics)
        print("\nCross-Validation Results Summary:")
        # Ensure 'object_tested' is string for display if it's an array from unique()
        results_df['object_tested_str'] = results_df['object_tested'].apply(lambda x: str(x) if not isinstance(x, str) else x)
        print(results_df[['fold', 'object_tested_str', 'mse', 'r2', 'fitting_time_s']])
        
        # Calculate and print mean for metrics that are not NaN
        valid_mse = results_df['mse'].dropna()
        valid_r2 = results_df['r2'].dropna()
        if not valid_mse.empty:
            print(f"\nAverage CV MSE: {valid_mse.mean():.6f} (Std: {valid_mse.std():.6f})")
        if not valid_r2.empty:
            print(f"Average CV R2 Score: {valid_r2.mean():.6f} (Std: {valid_r2.std():.6f})")
        print(f"Average Fitting Time per Fold: {results_df['fitting_time_s'].mean():.2f}s")
        
        try:
            results_df.to_csv("gpr_cv_results.csv", index=False)
            print("CV results saved to gpr_cv_results.csv")
        except Exception as e:
            print(f"Error saving CV results: {e}")

    print("\n--- GPR Training and Cross-Validation Script Finished ---")
    print("Next steps: ")
    print("1. Analyze CV results: Check consistency, inspect learned kernels, errors per object.")
    print("2. Hyperparameter Tuning: Based on results, you might adjust kernel choices, initial parameters, or bounds.")
    print("3. Final Model Training: Once satisfied, train a final GPR model on ALL development data (all filtered runs 1 & 2).")
    print("4. Final Evaluation: Test this final model on your hold-out test set (processed runs 3 & 4).")