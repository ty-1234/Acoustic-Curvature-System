"""
MLP_train_LOGO_CV.py - Multi-Layer Perceptron for Curvature Sensor Data with LOGO CV

This script trains a Multi-Layer Perceptron (MLP) model to predict
surface curvature from sensor data, using Leave-One-Curvature-Profile-Out
cross-validation.

Process Overview:
1. Loads pre-cleaned CSV files for a specific test session (e.g., "[test 1]").
   (Assumes these files are outputs from a previous cleaning/merging script).
2. Normalizes FFT readings by subtracting a baseline calculated from "idle" periods.
3. Converts target curvature units from mm^-1 to m^-1.
4. Performs Leave-One-Curvature-Profile-Out cross-validation (LOGO CV):
   - Standardizes input features (FFT + Position) for each fold.
   - Standardizes target variable (Curvature in m^-1) for each fold for MLP training.
   - Trains an MLP on N-1 curvature profiles.
   - Tests on the held-out curvature profile.
   - Repeats until each curvature profile has been used as the test set.
5. Evaluates model performance with detailed metrics (MSE, RMSE, MAE, R2).
6. Generates visualization plots to assess model accuracy.
7. Trains a final MLP model on all data for the session for potential deployment.

Cross-Validation Approach:
- Uses LeaveOneGroupOut where each "group" is a distinct curvature profile.
- This tests how well the model generalizes to entirely new curvature values.

Generated Visualizations (similar to GPR script):
1. Predicted vs. True Curvature
2. Residuals vs. True Curvature
3. Absolute Error Distribution (Box Plot by Group)
 (Note: Std Dev plot specific to GPR uncertainty is omitted for standard MLP)

Usage:
- Set TARGET_TEST_SESSION_NUM to select which test session to process.
- Ensure input CSV files are in the expected location (e.g., "csv_data/cleaned/").
- Output files (models, metrics, plots) are saved to a session-specific directory
  within "mlp_model_outputs/".
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
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow / Keras Imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # BatchNormalization is optional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # ReduceLROnPlateau is optional

# Ensure consistent results for Keras/TF for reproducibility during experimentation
# (Can be commented out for final production runs if stochasticity is desired)
# tf.keras.utils.set_random_seed(42)
# tf.config.experimental.enable_op_determinism()


# =================================================================================
# --- Global Configuration (Adapted from GPR script) ---
# =================================================================================
FFT_COLUMNS = ['FFT_200Hz', 'FFT_400Hz', 'FFT_600Hz', 'FFT_800Hz',
               'FFT_1000Hz', 'FFT_1200Hz', 'FFT_1400Hz', 'FFT_1600Hz',
               'FFT_1800Hz', 'FFT_2000Hz']
POSITION_COLUMN = 'Position_cm'
ORIGINAL_TARGET_COLUMN = 'Curvature' # Assuming this is in mm^-1
ACTIVITY_COLUMN = 'Curvature_Active'
TIMESTAMP_COLUMN = 'Timestamp'

FEATURE_COLUMNS = FFT_COLUMNS + [POSITION_COLUMN] # Inputs to MLP
TARGET_COLUMN_MM_INV = ORIGINAL_TARGET_COLUMN # Original target in mm^-1
TARGET_COLUMN_M_INV = 'Curvature_m_inv' # New target in m^-1

# =================================================================================
# --- Function: Parse Filename for Group (Curvature Value - Independent of Session Tag for Combined Runs) ---
# =================================================================================
def parse_filename_for_curvature_group(filename): # Removed target_test_object_num_str as direct dependency for parsing
    """
    Extracts a group identifier (curvature value string) from filenames like
    'cleaned_0_01[test 1].csv' -> '0.01'
    'merged_0.007142[test 3].csv' -> '0.007142'
    It focuses on the numerical part assumed to be the curvature.
    """
    base_name = os.path.basename(filename)
    
    # Try to find a pattern like "_[NUMBER_WITH_DOTS_OR_UNDERSCORES]" followed by optional space and "[test...]" or just end of string/extension
    # This regex aims to capture the curvature value, which can contain dots or underscores (later replaced by dots)
    # It looks for a number that is typically between an underscore and the session tag or end of filename.
    match = re.match(r".*?_([0-9][0-9._]*)(?:\[test.*|\.csv)", base_name, re.IGNORECASE)
    
    if match:
        curvature_str = match.group(1)
        # Replace underscores used in some filenames (e.g., 0_05) with dots
        return curvature_str.replace('_', '.')
    else:
        # Fallback if the primary pattern fails - this might need adjustment based on exact naming variations
        # Try a simpler match for numbers that might be at the start or after a prefix
        simple_match_start = re.match(r"([0-9][0-9._]*)(?:\[test.*|\.csv)", base_name, re.IGNORECASE) # If curvature is at the start
        if simple_match_start:
             return simple_match_start.group(1).replace('_', '.')

        print(f"    Warning: Could not accurately parse curvature group from filename: {base_name}. Using full filename stem as group. Review parsing logic if this happens frequently.")
        return os.path.splitext(base_name)[0]

# =================================================================================
# --- Function: Create MLP Model (NEW) ---
# =================================================================================
def create_mlp_model(input_dim, output_dim=1, learning_rate=0.001):
    """Defines and compiles the MLP model."""
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
    # model.add(BatchNormalization()) # Optional
    model.add(Dropout(0.3)) # Regularization
    model.add(Dense(64, activation='relu'))
    # model.add(BatchNormalization()) # Optional
    model.add(Dropout(0.2)) # Regularization
    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.1)) # Optional
    model.add(Dense(output_dim, activation='linear')) # Linear activation for regression

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    print("MLP Model Summary:")
    model.summary()
    return model

# =================================================================================
# --- Function: Generate Performance Plots (Adapted from GPR script) ---
# =================================================================================
def generate_performance_plots(detailed_predictions_df, output_dir, model_name_prefix="mlp"):
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
        plt.xlabel("True Curvature ($m^{-1}$)") # Assuming units are now m^-1
        plt.ylabel("Predicted Curvature ($m^{-1}$)")
        plt.title(f"Predicted vs. True Curvature ({model_name_prefix.upper()})")
        plt.legend(title="Tested Curvature Group", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle=':', alpha=0.7)
        # plt.axis('equal') # Can sometimes make plots too small if ranges differ significantly
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
                sorted_groups = sorted(unique_groups_for_plot, key=lambda x: float(str(x).replace('_','.')))
            except ValueError:
                sorted_groups = sorted(unique_groups_for_plot, key=str)
            sns.boxplot(x='tested_group_for_row', y='abs_error', data=detailed_predictions_df, order=sorted_groups, palette="viridis")
            plt.xlabel("Tested Curvature Group ($mm^{-1}$ values used for grouping)") # Clarify group label units
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

    # Plot 4 (Std Dev plot) is omitted for standard MLP as it doesn't directly output prediction std dev.
    # Could be added if using methods like MC Dropout for uncertainty.

# =================================================================================
# --- Main MLP Training and Evaluation Function (NEW Structure) ---
# =================================================================================
def train_mlp_for_selected_test_run(
    input_files_for_session,
    test_session_num_str, # This now serves as a general label/prefix for the run
    output_dir_for_session
):
    all_mlp_ready_segments = []
    successful_baseline_count = 0
    files_without_baseline = []

    print(f"Starting MLP data preparation for {len(input_files_for_session)} files from session '[test {test_session_num_str}]'...")

    for file_path in input_files_for_session:
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

        # --- Baseline FFT Subtraction (from GPR script) ---
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
            # Decide if you want to proceed without baseline subtraction or skip these files
            # For now, we proceed but this might affect results.
        
        # --- Unit Conversion for Target ---
        active_df[TARGET_COLUMN_M_INV] = active_df[TARGET_COLUMN_MM_INV] * 1000
        print(f"    Converted target '{ORIGINAL_TARGET_COLUMN}' (mm^-1) to '{TARGET_COLUMN_M_INV}' (m^-1).")

        # group_id = parse_filename_for_curvature_group(base_filename, test_session_num_str) # OLD CALL
        group_id = parse_filename_for_curvature_group(base_filename) # NEW CALL - uses the revised function
        active_df['group'] = group_id
        all_mlp_ready_segments.append(active_df)

    print(f"\nBaseline Calculation & Preprocessing Summary for {len(input_files_for_session)} files:")
    print(f"  Successfully calculated idle baseline for {successful_baseline_count} files.")
    if files_without_baseline:
        print(f"  Files with warnings/issues during baseline/preprocessing:")
        for f_issue in files_without_baseline:
            print(f"    - {f_issue}")

    if not all_mlp_ready_segments:
        print("No data available for MLP training after processing all selected files.")
        return

    combined_df = pd.concat(all_mlp_ready_segments, ignore_index=True)
    if combined_df.empty:
        print("Combined DataFrame is empty. Cannot proceed with MLP.")
        return

    print(f"\nCombined preprocessed data shape for MLP: {combined_df.shape}")
    print(f"Unique curvature groups for LOGO CV: {sorted(combined_df['group'].unique())}")

    X = combined_df[FEATURE_COLUMNS]
    y = combined_df[TARGET_COLUMN_M_INV] # Target is now in m^-1
    groups = combined_df['group']
    unique_group_ids = groups.unique()

    model_name_prefix = f"mlp_test{test_session_num_str}"

    if len(unique_group_ids) < 2:
        print(f"Only {len(unique_group_ids)} unique curvature group(s) found: {unique_group_ids}. LeaveOneGroupOut CV requires at least 2 groups. CV will be skipped.")
        # Consider training a single model on all data if CV cannot be performed.
        return

    logo = LeaveOneGroupOut()
    fold_metrics = []
    all_fold_predictions_data = []
    fold_num = 0

    print(f"\nStarting Leave-One-Curvature-Profile-Out CV for session '[test {test_session_num_str}]' using {len(unique_group_ids)} groups...")

    for train_idx, test_idx in logo.split(X, y, groups):
        fold_num += 1
        X_train_fold, X_test_fold = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train_fold, y_test_fold = y.iloc[train_idx].copy(), y.iloc[test_idx].copy() # These are in m^-1

        current_test_group_curvature = groups.iloc[test_idx].unique()[0]
        print(f"\n--- Fold {fold_num}: Testing on Curvature Group '{current_test_group_curvature}' ---")
        print(f"  Train set size: {len(X_train_fold)}, Test set size: {len(X_test_fold)}")

        # 1. Scale Input Features (X)
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_fold)
        X_test_scaled = scaler_X.transform(X_test_fold)

        # 2. Scale Target Variable (y) for MLP training
        scaler_y = StandardScaler() # Or MinMaxScaler()
        y_train_scaled = scaler_y.fit_transform(y_train_fold.values.reshape(-1, 1))
        # y_test_fold remains in its original m^-1 scale for evaluation

        # 3. Create Validation Split for Early Stopping (from this fold's training data)
        X_sub_train, X_val, y_sub_train_scaled, y_val_scaled = train_test_split(
            X_train_scaled, y_train_scaled, test_size=0.2, random_state=42 # For reproducibility of this split
        )

        # 4. Create and Train MLP Model
        mlp_model = create_mlp_model(X_train_scaled.shape[1], learning_rate=0.001) # Pass input dim

        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1) # Increased patience
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1) # Optional
        callbacks_list = [early_stopping, reduce_lr]

        print(f"  Training MLP on {X_sub_train.shape[0]} samples, validating on {X_val.shape[0]} samples...")
        start_time_fold_fit = time.time()
        history = mlp_model.fit(X_sub_train, y_sub_train_scaled,
                                epochs=250, # Max epochs
                                batch_size=32, # Experiment with 16, 64
                                validation_data=(X_val, y_val_scaled),
                                callbacks=callbacks_list,
                                verbose=1)
        fit_time_fold = time.time() - start_time_fold_fit
        print(f"  MLP fitting completed in {fit_time_fold:.2f}s.")

        # 5. Predict and Inverse Scale
        y_pred_scaled_fold = mlp_model.predict(X_test_scaled)
        y_pred_m_inv_fold = scaler_y.inverse_transform(y_pred_scaled_fold)

        # 6. Evaluate
        mse = mean_squared_error(y_test_fold, y_pred_m_inv_fold)
        mae = mean_absolute_error(y_test_fold, y_pred_m_inv_fold)
        r2 = r2_score(y_test_fold, y_pred_m_inv_fold)
        rmse = np.sqrt(mse)

        fold_metrics.append({
            'fold': fold_num, 'tested_group': current_test_group_curvature,
            'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
            'fitting_time_s': fit_time_fold,
            'best_epoch': np.argmin(history.history['val_loss']) + 1 if history and 'val_loss' in history.history else 'N/A'
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
        # Add original features back for detailed analysis if needed
        for col in X_test_fold.columns:
             if col not in fold_pred_df.columns:
                 fold_pred_df[col] = X_test_fold[col].values
        all_fold_predictions_data.append(fold_pred_df)

    print("\n--- Cross-Validation Finished ---")
    if fold_metrics:
        results_df = pd.DataFrame(fold_metrics)
        print(f"\nCross-Validation Results Summary (LOGO CV for session [test {test_session_num_str}]):")
        columns_to_print = ['fold', 'tested_group', 'mse', 'rmse', 'mae', 'r2', 'fitting_time_s', 'best_epoch']
        with pd.option_context('display.max_colwidth', None, 'display.width', 1000):
             print(results_df[columns_to_print])
        print(f"\nAverage CV MSE: {results_df['mse'].mean():.6f} (m^-1)^2")
        print(f"Average CV RMSE: {results_df['rmse'].mean():.6f} m^-1")
        print(f"Average CV MAE: {results_df['mae'].mean():.6f} m^-1")
        print(f"Average CV R2 Score: {results_df['r2'].mean():.4f}")

        output_csv_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_LOGO_cv_results.csv")
        results_df.to_csv(output_csv_path, index=False)
        print(f"CV results saved to {output_csv_path}")

    if all_fold_predictions_data:
        detailed_predictions_df = pd.concat(all_fold_predictions_data, ignore_index=True)
        detailed_output_csv_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_detailed_predictions.csv")
        detailed_predictions_df.to_csv(detailed_output_csv_path, index=False)
        print(f"Detailed predictions and errors saved to {detailed_output_csv_path}")
        generate_performance_plots(detailed_predictions_df, output_dir_for_session, model_name_prefix)

    # --- Training Final Model on All Data for the Session ---
    print(f"\n--- Training Final MLP Model on All Processed Data for session [test {test_session_num_str}] ---")
    if not X.empty and not y.empty:
        final_scaler_X = StandardScaler()
        X_scaled_final = final_scaler_X.fit_transform(X)

        final_scaler_y = StandardScaler() # Or MinMaxScaler()
        y_scaled_final = final_scaler_y.fit_transform(y.values.reshape(-1, 1))

        # For the final model, you might want to use insights from CV for epochs,
        # or use early stopping with a small final validation split from the whole dataset.
        # Example: Split all data for final training + validation for early stopping
        X_train_all, X_val_final, y_train_all_scaled, y_val_all_scaled = train_test_split(
            X_scaled_final, y_scaled_final, test_size=0.15, random_state=42
        )

        final_mlp_model = create_mlp_model(X_scaled_final.shape[1], learning_rate=0.001) # Use tuned LR if found

        print(f"Fitting final MLP model on {X_train_all.shape[0]} samples, validating on {X_val_final.shape[0]}...")
        start_time_final = time.time()
        final_mlp_model.fit(X_train_all, y_train_all_scaled,
                            epochs=250, # Or a value guided by CV average best_epoch
                            batch_size=32,
                            validation_data=(X_val_final, y_val_all_scaled),
                            callbacks=[EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
                                       ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1)],
                            verbose=1)
        fit_time_final = time.time() - start_time_final
        print(f"Final MLP model fitted in {fit_time_final:.2f}s.")

        model_save_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_final_model.keras") # Use .keras format
        scaler_X_save_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_final_scaler_X.joblib")
        scaler_y_save_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_final_scaler_y.joblib")

        final_mlp_model.save(model_save_path)
        joblib.dump(final_scaler_X, scaler_X_save_path)
        joblib.dump(final_scaler_y, scaler_y_save_path)
        print(f"Final MLP model saved to: {model_save_path}")
        print(f"Final X Scaler saved to: {scaler_X_save_path}")
        print(f"Final y Scaler saved to: {scaler_y_save_path}")
    else:
        print("Skipping final model training as combined data is empty.")

# =================================================================================
# --- Main Execution (Combined Sessions for LOGO CV) ---
# =================================================================================
if __name__ == "__main__":
    # === OPERATOR: DEFINE ALL TARGET TEST SESSION NUMBERS TO COMBINE HERE ===
    ALL_SESSION_NUMS_TO_PROCESS = ["1", "2", "3", "4"] 
    # Or, to process only sessions 1 and 2: ALL_SESSION_NUMS_TO_PROCESS = ["1", "2"]
    # To process a single session (original behavior if list has one item): ALL_SESSION_NUMS_TO_PROCESS = ["1"]
    # =================================================================================

    script_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_DATA_DIR = os.path.join(script_dir, "csv_data", "cleaned") 
    
    # Base output directory for this script's runs
    OUTPUT_RESULTS_BASE_DIR_MLP = os.path.join(script_dir, "mlp_model_outputs") 
    
    # Create a descriptive label for the combined run based on sessions used
    if len(ALL_SESSION_NUMS_TO_PROCESS) > 1:
        run_descriptor = f"LOGO_CV_Combined_Sessions_{'_'.join(ALL_SESSION_NUMS_TO_PROCESS)}"
    elif len(ALL_SESSION_NUMS_TO_PROCESS) == 1:
        run_descriptor = f"LOGO_CV_Session_{ALL_SESSION_NUMS_TO_PROCESS[0]}"
    else:
        print("Error: No sessions specified in ALL_SESSION_NUMS_TO_PROCESS. Exiting.")
        exit() # Or raise an error
        
    FINAL_OUTPUT_DIR_FOR_RUN = os.path.join(OUTPUT_RESULTS_BASE_DIR_MLP, run_descriptor)
    
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
        print(f"No 'cleaned_*.csv' files found for any of the specified sessions. MLP training cannot proceed.")
    else:
        # Use set to ensure unique files if any overlap by mistake, then sort for consistency
        unique_files_to_process = sorted(list(set(all_files_to_process_for_this_run)))
        print(f"\nFound a total of {len(unique_files_to_process)} unique 'cleaned_*.csv' file(s) from the specified session(s) to process for MLP training:")
        for f_path in unique_files_to_process:
            print(f"  - {os.path.basename(f_path)}")
        
        # The `test_session_num_str` argument to train_mlp_for_selected_test_run now acts
        # primarily as a prefix for output files within the FINAL_OUTPUT_DIR_FOR_RUN.
        # The actual data being processed is defined by unique_files_to_process.
        model_and_file_prefix_label = run_descriptor # Use the descriptive run label for file naming

        train_mlp_for_selected_test_run(
            unique_files_to_process, 
            model_and_file_prefix_label, 
            FINAL_OUTPUT_DIR_FOR_RUN
        )
    
    print(f"\nMLP Script finished processing for run: {run_descriptor}.")