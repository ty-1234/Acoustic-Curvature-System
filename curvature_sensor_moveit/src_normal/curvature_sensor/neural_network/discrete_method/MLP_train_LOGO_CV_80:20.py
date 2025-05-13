"""
MLP_train_LOGO_CV_80:20.py - Multi-Layer Perceptron for Curvature Sensor Data.
This version is MODIFIED FOR 80/20 RANDOM SPLIT SANITY CHECK.

This script trains a Multi-Layer Perceptron (MLP) model to predict
surface curvature from sensor data, using a random 80/20 train-test split
on the combined data from specified sessions.

Process Overview:
1. Loads pre-cleaned CSV files for specified test sessions.
2. Normalizes FFT readings by subtracting a baseline calculated from "idle" periods.
3. Converts target curvature units from mm^-1 to m^-1.
4. Performs an 80/20 random train-test split on the combined dataset:
   - Standardizes input features (FFT + Position).
   - Standardizes target variable (Curvature in m^-1) for MLP training.
   - Trains an MLP on 80% of the data (using a portion of this for validation).
   - Tests on the held-out 20% of the data.
5. Evaluates model performance with detailed metrics (MSE, RMSE, MAE, R2).
6. Generates visualization plots to assess model accuracy.
7. Trains a final MLP model on the 80% training data for potential deployment.

Cross-Validation Approach:
- This version uses a single random 80/20 train-test split, NOT LeaveOneGroupOut CV.
- Stratification by curvature group is attempted for the split.

Generated Visualizations (similar to GPR script):
1. Predicted vs. True Curvature
2. Residuals vs. True Curvature
3. Absolute Error Distribution (Box Plot by Group, if group info available for test set)

Usage:
- Set ALL_SESSION_NUMS_TO_PROCESS to select which test sessions to combine and process.
- Ensure input CSV files are in the expected location (e.g., "csv_data/cleaned/").
- Output files (models, metrics, plots) are saved to a run-specific directory
  within "mlp_model_outputs/".
"""

import pandas as pd
import numpy as np
import os
import glob
import re
import time
from sklearn.model_selection import train_test_split # Changed from LeaveOneGroupOut
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
def generate_performance_plots(detailed_predictions_df, output_dir, model_name_prefix="mlp_80_20_split"):
    if detailed_predictions_df.empty:
        print("  No detailed prediction data to generate plots.")
        return

    print("\n  Generating performance plots...")
    plot_filename_prefix = f"{model_name_prefix}_"

    # Plot 1: Predicted vs. True Curvature
    plt.figure(figsize=(9, 8))
    try:
        # For random split, 'tested_group_for_row' will be the original group of the sample
        hue_column = 'original_group' if 'original_group' in detailed_predictions_df.columns else None
        sns.scatterplot(x='true_curvature', y='predicted_curvature', hue=hue_column,
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
            plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xlim(plot_min, plot_max)
        plt.ylim(plot_min, plot_max)
        plt.tight_layout(rect=[0, 0, 0.85 if hue_column else 1, 1])
        plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}pred_vs_true.png"), bbox_inches='tight')
        plt.close()
        print(f"    Saved: Predicted vs. True plot ({os.path.join(output_dir, f'{plot_filename_prefix}pred_vs_true.png')})")
    except Exception as e:
        print(f"    Error generating Predicted vs. True plot: {e}")

    # Plot 2: Residuals vs. True Curvature
    plt.figure(figsize=(9, 6))
    try:
        sns.scatterplot(x='true_curvature', y='error', hue=hue_column,
                        data=detailed_predictions_df, alpha=0.6, s=30, edgecolor='k', linewidth=0.5)
        plt.axhline(0, color='k', linestyle='--', lw=2)
        plt.xlabel("True Curvature ($m^{-1}$)")
        plt.ylabel("Residual (True - Predicted) ($m^{-1}$)")
        plt.title(f"Residuals vs. True Curvature ({model_name_prefix.upper()})")
        if hue_column:
            plt.legend(title="Original Curvature Group", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout(rect=[0, 0, 0.85 if hue_column else 1, 1])
        plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}residuals_vs_true.png"), bbox_inches='tight')
        plt.close()
        print(f"    Saved: Residuals vs. True plot ({os.path.join(output_dir, f'{plot_filename_prefix}residuals_vs_true.png')})")
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
                sorted_groups = sorted(unique_groups_for_plot, key=str) # Fallback sort
            sns.boxplot(x=hue_column, y='abs_error', data=detailed_predictions_df, order=sorted_groups, palette="viridis")
            plt.xlabel("Original Curvature Group ($mm^{-1}$ values used for grouping)")
            plt.ylabel("Absolute Error ($m^{-1}$)")
            plt.title(f"Absolute Error Distribution by Original Group ({model_name_prefix.upper()})")
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, axis='y', linestyle=':', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{plot_filename_prefix}abs_error_boxplot_by_group.png"))
            plt.close()
            print(f"    Saved: Absolute Error Boxplot by Group ({os.path.join(output_dir, f'{plot_filename_prefix}abs_error_boxplot_by_group.png')})")
        except Exception as e:
            print(f"    Error generating Absolute Error Boxplot: {e}")

# =================================================================================
# --- Main MLP Training and Evaluation Function (80/20 Split Version) ---
# =================================================================================
def train_mlp_for_selected_test_run(
    input_files_for_session,
    run_label, # General label for the run (e.g., "Combined_Sessions_1_2_3_4")
    output_dir_for_session
):
    all_mlp_ready_segments = []
    successful_baseline_count = 0
    files_without_baseline = []

    # The run_label might contain session info, but data prep is for all input_files_for_session
    print(f"Starting MLP data preparation for {len(input_files_for_session)} files for run: '{run_label}'...")

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

        if not all(col in df_cleaned_input.columns for col in [POSITION_COLUMN, ACTIVITY_COLUMN] + FFT_COLUMNS + [ORIGINAL_TARGET_COLUMN]):
            print(f"    Skipping {base_filename}: missing one or more required columns.")
            files_without_baseline.append(f"{base_filename} (Missing required columns)")
            continue

        idle_fft_baseline = None
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
        
        active_df[TARGET_COLUMN_M_INV] = active_df[TARGET_COLUMN_MM_INV] * 1000
        print(f"    Converted target '{ORIGINAL_TARGET_COLUMN}' (mm^-1) to '{TARGET_COLUMN_M_INV}' (m^-1).")

        group_id = parse_filename_for_curvature_group(base_filename)
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
    
    X = combined_df[FEATURE_COLUMNS]
    y = combined_df[TARGET_COLUMN_M_INV]
    groups = combined_df['group'] # For stratification and plotting

    model_name_prefix = f"mlp_{run_label}_80_20_split" # Reflects the run and split type

    # --- 80/20 Train-Test Split ---
    print(f"\nPerforming 80/20 random train-test split...")
    stratify_on = groups if groups is not None and len(groups.unique()) > 1 else None
    if stratify_on is not None and len(stratify_on.unique()) < 2 : # Check for scikit-learn min groups for stratify
        print(f"  Warning: Stratification by group requested, but found only {len(stratify_on.unique())} unique groups. Stratification may not be effective or possible.")
        # if len(stratify_on.unique()) < 2: stratify_on = None # Disable stratification if not enough groups

    # Split data into training (80%) and test (20%)
    # X_train_full will be further split for validation
    X_train_full, X_test, y_train_full, y_test, groups_train_full, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42, stratify=stratify_on
    )
    print(f"  Full training set size: {len(X_train_full)}, Test set size: {len(X_test)}")

    # 1. Scale Input Features (X)
    scaler_X = StandardScaler()
    X_train_full_scaled = scaler_X.fit_transform(X_train_full)
    X_test_scaled = scaler_X.transform(X_test)

    # 2. Scale Target Variable (y) for MLP training
    scaler_y = StandardScaler()
    y_train_full_scaled = scaler_y.fit_transform(y_train_full.values.reshape(-1, 1))
    # y_test remains in its original m^-1 scale for evaluation

    # 3. Create Validation Split for Early Stopping (from the 80% training data)
    # This means 20% of the 80% training data is used for validation (i.e., 16% of total)
    X_sub_train_scaled, X_val_scaled, y_sub_train_scaled, y_val_scaled = train_test_split(
        X_train_full_scaled, y_train_full_scaled, test_size=0.2, random_state=42 # For reproducibility of this internal split
    )
    print(f"  Sub-training set for MLP: {len(X_sub_train_scaled)}, Validation set for MLP: {len(X_val_scaled)}")

    # 4. Create and Train MLP Model
    mlp_model = create_mlp_model(X_sub_train_scaled.shape[1], learning_rate=0.001)

    early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    callbacks_list = [early_stopping, reduce_lr]

    print(f"  Training MLP on {X_sub_train_scaled.shape[0]} samples, validating on {X_val_scaled.shape[0]} samples...")
    start_time_fit = time.time()
    history = mlp_model.fit(X_sub_train_scaled, y_sub_train_scaled,
                            epochs=250,
                            batch_size=32,
                            validation_data=(X_val_scaled, y_val_scaled),
                            callbacks=callbacks_list,
                            verbose=1)
    fit_time = time.time() - start_time_fit
    print(f"  MLP fitting completed in {fit_time:.2f}s.")

    # 5. Predict and Inverse Scale on the Test Set
    y_pred_scaled_test = mlp_model.predict(X_test_scaled)
    y_pred_m_inv_test = scaler_y.inverse_transform(y_pred_scaled_test)

    # 6. Evaluate on the Test Set
    mse = mean_squared_error(y_test, y_pred_m_inv_test)
    mae = mean_absolute_error(y_test, y_pred_m_inv_test)
    r2 = r2_score(y_test, y_pred_m_inv_test)
    rmse = np.sqrt(mse)
    best_epoch_val = np.argmin(history.history['val_loss']) + 1 if history and 'val_loss' in history.history else 'N/A'

    print(f"\n--- Evaluation Results on 20% Test Set ---")
    print(f"  MSE: {mse:.6f} (m^-1)^2")
    print(f"  RMSE: {rmse:.6f} m^-1")
    print(f"  MAE: {mae:.6f} m^-1")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  Best Epoch (based on val_loss): {best_epoch_val}")
    print(f"  Training time: {fit_time:.2f}s")

    # Save results
    results_summary = {
        'run_label': run_label,
        'split_type': '80_20_random',
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'fitting_time_s': fit_time,
        'best_epoch_val_loss': best_epoch_val,
        'total_samples': len(X),
        'train_samples_80_percent': len(X_train_full),
        'test_samples_20_percent': len(X_test)
    }
    results_df = pd.DataFrame([results_summary])
    output_csv_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_results.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"Run results saved to {output_csv_path}")

    # Prepare detailed predictions for plotting
    detailed_predictions_df = pd.DataFrame({
        'true_curvature': y_test.values.flatten(),
        'predicted_curvature': y_pred_m_inv_test.flatten(),
        'error': y_test.values.flatten() - y_pred_m_inv_test.flatten(),
        'abs_error': np.abs(y_test.values.flatten() - y_pred_m_inv_test.flatten()),
        'original_group': groups_test.values.flatten() # Keep original group for plotting
    })
    # Add original features from X_test back for detailed analysis if needed
    for col in X_test.columns:
         if col not in detailed_predictions_df.columns:
             detailed_predictions_df[col] = X_test[col].values

    detailed_output_csv_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_detailed_predictions.csv")
    detailed_predictions_df.to_csv(detailed_output_csv_path, index=False)
    print(f"Detailed predictions and errors for test set saved to {detailed_output_csv_path}")
    generate_performance_plots(detailed_predictions_df, output_dir_for_session, model_name_prefix)

    # --- Training "Final" Model on the 80% Training Data ---
    print(f"\n--- Training MLP Model on the full 80% Training Data ({run_label}) ---")
    # We already have X_train_full_scaled and y_train_full_scaled
    # For this "final" training on the 80% data, we can choose to use early stopping with a new validation split
    # or train for a fixed number of epochs (e.g., best_epoch_val from the previous run).
    # Let's use early stopping with a new split from the 80% data.
    
    X_final_train_sub_scaled, X_final_val_scaled, y_final_train_sub_scaled, y_final_val_scaled = train_test_split(
        X_train_full_scaled, y_train_full_scaled, test_size=0.15, random_state=123 # Different random state
    )

    final_model_on_80_percent = create_mlp_model(X_train_full_scaled.shape[1], learning_rate=0.001)
    print(f"Fitting model on {X_final_train_sub_scaled.shape[0]} samples (from 80% train), validating on {X_final_val_scaled.shape[0]} samples...")
    start_time_final_fit = time.time()
    final_model_on_80_percent.fit(X_final_train_sub_scaled, y_final_train_sub_scaled,
                        epochs=250,
                        batch_size=32,
                        validation_data=(X_final_val_scaled, y_final_val_scaled),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
                                   ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1)],
                        verbose=1)
    fit_time_final = time.time() - start_time_final_fit
    print(f"Model fitting on 80% training data completed in {fit_time_final:.2f}s.")

    model_save_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_trained_on_80pct.keras")
    scaler_X_save_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_scaler_X_from_80pct.joblib")
    scaler_y_save_path = os.path.join(output_dir_for_session, f"{model_name_prefix}_scaler_y_from_80pct.joblib")

    final_model_on_80_percent.save(model_save_path)
    joblib.dump(scaler_X, scaler_X_save_path) # scaler_X was fit on X_train_full
    joblib.dump(scaler_y, scaler_y_save_path) # scaler_y was fit on y_train_full
    print(f"Model (trained on 80%) saved to: {model_save_path}")
    print(f"X Scaler (from 80%) saved to: {scaler_X_save_path}")
    print(f"y Scaler (from 80%) saved to: {scaler_y_save_path}")

# =================================================================================
# --- Main Execution (Combined Sessions for 80/20 Split) ---
# =================================================================================
if __name__ == "__main__":
    # === OPERATOR: DEFINE ALL TARGET TEST SESSION NUMBERS TO COMBINE HERE ===
    ALL_SESSION_NUMS_TO_PROCESS = ["1", "2", "3", "4"] 
    # Or, to process only sessions 1 and 2: ALL_SESSION_NUMS_TO_PROCESS = ["1", "2"]
    # Or a single session: ALL_SESSION_NUMS_TO_PROCESS = ["1"]
    # =================================================================================

    script_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_DATA_DIR = os.path.join(script_dir, "csv_data", "cleaned") 
    
    OUTPUT_RESULTS_BASE_DIR_MLP = os.path.join(script_dir, "mlp_model_outputs") 
    
    if len(ALL_SESSION_NUMS_TO_PROCESS) > 1:
        run_descriptor = f"Combined_Sessions_{'_'.join(ALL_SESSION_NUMS_TO_PROCESS)}"
    elif len(ALL_SESSION_NUMS_TO_PROCESS) == 1:
        run_descriptor = f"Session_{ALL_SESSION_NUMS_TO_PROCESS[0]}"
    else:
        print("Error: No sessions specified in ALL_SESSION_NUMS_TO_PROCESS. Exiting.")
        exit()
        
    # Append split type to directory name for clarity
    FINAL_OUTPUT_DIR_FOR_RUN = os.path.join(OUTPUT_RESULTS_BASE_DIR_MLP, f"{run_descriptor}_80_20_Split")
    
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
        unique_files_to_process = sorted(list(set(all_files_to_process_for_this_run)))
        print(f"\nFound a total of {len(unique_files_to_process)} unique 'cleaned_*.csv' file(s) from the specified session(s) to process for MLP training:")
        for f_path in unique_files_to_process:
            print(f"  - {os.path.basename(f_path)}")
        
        # The `run_descriptor` is passed as the general label for the run.
        train_mlp_for_selected_test_run(
            unique_files_to_process, 
            run_descriptor, # This is the general label for the run
            FINAL_OUTPUT_DIR_FOR_RUN
        )
    
    print(f"\nMLP Script (80/20 split version) finished processing for run: {run_descriptor}.")