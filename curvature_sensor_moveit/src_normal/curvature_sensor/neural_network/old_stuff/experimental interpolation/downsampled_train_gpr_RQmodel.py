import pandas as pd
import numpy as np
import os
import glob
import re
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
# GPR specific imports from Scikit-learn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, WhiteKernel, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score
import time # To time GPR fitting

# --- Global Configuration (Adjust these paths as needed) ---
ORIGINAL_DATA_DIR = "csv_data/merged/"      
PROCESSED_DATA_DIR = "csv_data/interpolated/" 

fft_columns = ['FFT_200Hz', 'FFT_400Hz', 'FFT_600Hz', 'FFT_800Hz', 
               'FFT_1000Hz', 'FFT_1200Hz', 'FFT_1400Hz', 'FFT_1600Hz', 
               'FFT_1800Hz', 'FFT_2000Hz']
position_column = 'Position_cm'
input_features_all = fft_columns + [position_column]
target_column = 'Curvature'
activity_column = 'Curvature_Active'

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

                if (object_id, run_id) not in original_first_row_ffts_dict:
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
                        print(f"CRITICAL Warning: Original file {original_f_path} for FFT baseline not found!")
                # print(f"  Loaded and tagged processed: {os.path.basename(processed_f_path)} (Object: {object_id}, Run: {run_id})") # Reduced verbosity
            except Exception as e:
                print(f"Error loading or processing file {processed_f_path} or its original for baseline: {e}")
        
    if not df_list:
        print(f"No processed data loaded for runs {runs_to_include}. Please check path and file names.")
        return pd.DataFrame(), {}
        
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Total combined processed data: {combined_df.shape[0]} rows from {len(df_list)} files.")
    return combined_df, original_first_row_ffts_dict

def get_fold_specific_fft_baseline(train_fold_metadata_df, all_original_first_row_ffts_dict, fft_column_names_list):
    baselines_to_average = []
    unique_runs_in_train_fold = train_fold_metadata_df[['object_id', 'run_id']].drop_duplicates()

    for _, row in unique_runs_in_train_fold.iterrows():
        run_key = (row['object_id'], row['run_id'])
        if run_key in all_original_first_row_ffts_dict:
            baselines_to_average.append(all_original_first_row_ffts_dict[run_key])
        else:
            print(f"Warning: Original first row FFT data not found for run key {run_key} in training fold baselines dict.")

    if not baselines_to_average:
        print("CRITICAL Warning: No original baseline FFTs found for this training fold. Using fallback (ones).")
        return pd.Series(1.0, index=fft_column_names_list)

    avg_baseline = pd.DataFrame(baselines_to_average).mean(axis=0) 
    avg_baseline[avg_baseline <= 1e-9] = 1e-9 
    print(f"  Calculated average FFT baseline for this fold from {len(baselines_to_average)} original first rows.")
    return avg_baseline

def normalize_fft_features_fold(X_df, fft_cols_list, fold_fft_baseline_series):
    X_norm_df = X_df.copy()
    for col in fft_cols_list:
        if col in fold_fft_baseline_series and fold_fft_baseline_series[col] != 0:
            X_norm_df[col] = X_df[col] / fold_fft_baseline_series[col]
        else:
            print(f"Warning: FFT baseline for {col} is zero or missing for this fold. Skipping normalization.")
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
    if n_splits < 2 and len(development_df_filtered['object_id'].unique()) > 1:
         print(f"Warning: LeaveOneGroupOut will result in only {n_splits} split(s). Ensure 'object_id' correctly identifies multiple distinct objects for proper LOOOCV.")
    elif len(development_df_filtered['object_id'].unique()) == 0:
         exit("Error: No unique object_ids found. Cannot perform LOOOCV.")
    elif n_splits == 0: # Should not happen if unique object_ids > 0
         exit("Error: LeaveOneGroupOut resulted in 0 splits. Check data and grouping.")


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
        X_train_fold_scaled_fft[position_column] = pos_scaler.fit_transform(X_train_fold_scaled_fft[[position_column]])
        X_test_fold_scaled_fft[position_column] = pos_scaler.transform(X_test_fold_scaled_fft[[position_column]])
        print(f"  '{position_column}' standardized for fold.")
        
        X_train_fold_final = X_train_fold_scaled_fft
        X_test_fold_final = X_test_fold_scaled_fft

        target_scaler = StandardScaler()
        y_train_fold_scaled = target_scaler.fit_transform(y_train_fold.values.reshape(-1, 1)).ravel()
        print(f"  Target '{target_column}' standardized for fold training.")
        
        # --- GPR Model Training, Prediction, and Evaluation ---
        print(f"  Initializing GPR model for fold {fold_num + 1}...")
        
        # Define the kernel (example, tune this based on paper and experimentation)
        # Length scale bounds are important. Using ARD by default for RBF if length_scale is array.
        # For RationalQuadratic, length_scale and alpha are key.
        # Add a WhiteKernel to account for noise in the observations.
        length_scale_bounds_val = (1e-2, 1e3) # Broad bounds, adjust based on feature scales
        alpha_bounds_val = (1e-2, 1e3) # For RationalQuadratic alpha
        noise_level_bounds_val = (1e-5, 1e1)

        # Example: Rational Quadratic Kernel as mentioned in the paper
        kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(
                        length_scale=1.0, # Initial guess, will be optimized if length_scale_bounds is not "fixed"
                        alpha=0.1,        # Initial guess for RQ alpha
                        length_scale_bounds=length_scale_bounds_val, 
                        alpha_bounds=alpha_bounds_val
                    ) + WhiteKernel(
                        noise_level=0.1, # Initial guess for noise
                        noise_level_bounds=noise_level_bounds_val
                    )
        
        # For ARD (Automatic Relevance Determination), provide an array for length_scale for RBF or Matern
        # If using RQ, it has a single length_scale. For ARD on all features, use RBF with length_scale array.
        # Example with RBF ARD:
        # kernel_rbf_ard = C(1.0) * RBF(length_scale=[1.0]*X_train_fold_final.shape[1], length_scale_bounds=length_scale_bounds_val) + \
        #                  WhiteKernel(noise_level=0.1, noise_level_bounds=noise_level_bounds_val)
        # Choose one kernel to start with. Let's use RQ based on the paper.

        gpr_model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=10, # Important for finding good hyperparameters
            alpha=1e-10,             # Small value added to diagonal of kernel matrix for numerical stability
            random_state=fold_num    # For reproducibility
        )
        
        print(f"  Fitting GPR for fold {fold_num + 1} (this might take some time)...")
        start_time = time.time()
        gpr_model.fit(X_train_fold_final, y_train_fold_scaled)
        end_time = time.time()
        print(f"  GPR fitting complete. Time taken: {end_time - start_time:.2f} seconds.")
        print(f"  Learned kernel for fold {fold_num + 1}: {gpr_model.kernel_}")

        print(f"  Predicting with GPR for fold {fold_num + 1}...")
        y_pred_scaled, y_pred_std_scaled = gpr_model.predict(X_test_fold_final, return_std=True)
        
        # Inverse transform predictions to original curvature scale
        y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
        # Standard deviation is also on the scaled target. To get it to original scale:
        # std_dev_original_scale = y_pred_std_scaled * target_scaler.scale_
        
        # Evaluate the model
        mse = mean_squared_error(y_test_fold_original, y_pred_original)
        r2 = r2_score(y_test_fold_original, y_pred_original)
        
        print(f"  Fold {fold_num + 1} Test MSE: {mse:.6f}")
        print(f"  Fold {fold_num + 1} Test R2 Score: {r2:.6f}")
        
        fold_metrics.append({
            'fold': fold_num + 1, 
            'object_tested': test_df_fold['object_id'].unique()[0], 
            'mse': mse, 
            'r2': r2,
            'fitting_time_s': end_time - start_time,
            'learned_kernel': str(gpr_model.kernel_) # Store as string
        })
        
        if fold_num == 0: # Just print some info for the first fold for brevity
            print("\n  --- Example: First Fold - Scaled Data & Predictions (Head) ---")
            print("  X_train_fold_final (head):")
            print(X_train_fold_final.head())
            print("\n  y_train_fold_scaled (head):")
            print(y_train_fold_scaled[:5])
            print("\n  X_test_fold_final (head):")
            print(X_test_fold_final.head())
            print("\n  y_test_fold_original (head):")
            print(y_test_fold_original.head())
            print("\n  y_pred_original (head):")
            print(y_pred_original[:5])
            print("  ---")
        
        print(f"--- End of Fold {fold_num + 1} ---")

    print("\n--- Cross-Validation Finished ---")
    if fold_metrics:
        results_df = pd.DataFrame(fold_metrics)
        print("\nCross-Validation Results Summary:")
        print(results_df[['fold', 'object_tested', 'mse', 'r2', 'fitting_time_s']])
        print(f"\nAverage CV MSE: {results_df['mse'].mean():.6f}")
        print(f"Average CV R2 Score: {results_df['r2'].mean():.6f}")
        print(f"Average Fitting Time per Fold: {results_df['fitting_time_s'].mean():.2f}s")
        
        # You might want to save results_df to a CSV
        # results_df.to_csv("gpr_cv_results.csv", index=False)
        # print("CV results saved to gpr_cv_results.csv")

    print("\n--- GPR Training and Cross-Validation Script Outline Finished ---")
    print("Next steps: ")
    print("1. Analyze CV results: Check consistency across folds, inspect learned kernels.")
    print("2. Hyperparameter Tuning: If performance is not optimal, you might need to adjust kernel choices, bounds, or use more advanced tuning.")
    print("3. Final Model Training: Once satisfied, train a final GPR model on ALL development data (all runs 1 & 2).")
    print("4. Final Evaluation: Test the final model on your hold-out test set (runs 3 & 4).")