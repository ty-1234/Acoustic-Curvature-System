"""
sparse_gpr_gpytorch_curvature_v5_leave_one_csv_out.py

An end-to-end machine learning pipeline for predicting sensor radius using
Sparse Gaussian Process Regression (SGPR) with GPyTorch, incorporating
Leave-One-CSV-Out Cross-Validation.

Key features in v5:
- Leave-One-CSV-Out Cross-Validation: Trains and evaluates the model K times,
  where K is the number of input CSV files. Each file serves as the test set once.
- SGPR Configuration (based on v3 success, further refined):
    - Noise: GaussianLikelihood with GammaPrior(concentration=1.1, rate=10.0).
    - Training Iterations: 1000.
    - Inducing Points: 1000.
    - Kernel Init: From scikit-learn best hparams.
    - Inducing Point Init: K-Means.
- Aggregated CV metrics reported.

Pipeline Steps (within each CV fold):
1.  Configuration.
2.  Data Splitting (one CSV for test, rest for train for the current fold).
3.  Load and Combine Data for train/test sets.
4.  Filter for Active Rows and Prepare Target (Radius_mm).
5.  Prepare Features (X) and Target (y).
6.  Data Cleaning.
7.  Scaling (Fit on current fold's train data, transform train & test).
8.  SGPR Model Definition (GPyTorch).
9.  SGPR Model Training.
10. Prediction.
11. Descaling.
12. Evaluation for the current fold.
After all folds: Aggregate and report CV metrics.
"""

# Standard library imports
import pandas as pd
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
from collections import defaultdict

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans

# GPyTorch and PyTorch imports
import torch
import gpytorch

# --- Stage 0: Configuration ---
VERBOSE = True
SEED = 42

# Data parameters
INPUT_DATA_DIR = '../csv_data/merged'
TARGET_COLUMN = 'Curvature'
TARGET_RADIUS_COLUMN = 'Radius_mm'
CURVATURE_ACTIVE_COLUMN = 'Curvature_Active'
FFT_PREFIX = 'FFT_'
POSITION_COLUMN = 'Position_cm'
RADIUS_CAP_VALUE = 1e6
FFT_MIN_HZ = 100
FFT_MAX_HZ = 16000
FFT_STEP_HZ = 100

# SGPR Specific Configuration
NUM_INDUCING_POINTS_TARGET = 1000 # Target number of inducing points
TRAINING_ITERATIONS = 1000
LEARNING_RATE = 0.01
USE_GPU = torch.cuda.is_available()

# Kernel parameter initialization
INITIALIZE_FROM_SKLEARN_HPARAMS = True
SKLEARN_BEST_LENGTHSCALE = 1.80
SKLEARN_BEST_ALPHA = 28.90
SKLEARN_BEST_OUTPUTSCALE = 0.155
# SKLEARN_BEST_NOISE is not used for direct init here, relying on GammaPrior

# Output directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_BASE_DIR = 'model_outputs'
RUN_ID = f"run_{TIMESTAMP}_sgpr_v5_leave_one_csv_out_cv"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, RUN_ID)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Plotting (plots will be per-fold if enabled, or an aggregate plot could be complex)
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
SAVE_PLOTS_PER_FOLD = False # Set to True to save plots for each CV fold
SHOW_PLOTS = False

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if USE_GPU:
    torch.cuda.manual_seed_all(SEED)

# Helper to convert NumPy to PyTorch Tensors
def to_tensor(data, use_gpu=False):
    tensor = torch.from_numpy(data).float()
    if use_gpu:
        tensor = tensor.cuda()
    return tensor

# --- SGPR Model Definition (same as v3) ---
class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, initial_raw_lengthscale=None, initial_raw_alpha=None, initial_raw_outputscale=None):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(ApproximateGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel_kwargs = {
            'lengthscale_prior': gpytorch.priors.SmoothedBoxPrior(0.01, 100.0, sigma=0.1, transform=torch.exp),
            'alpha_prior': gpytorch.priors.SmoothedBoxPrior(0.01, 100.0, sigma=0.1,  transform=torch.exp)
        }
        base_kernel = gpytorch.kernels.RQKernel(**base_kernel_kwargs)
        if initial_raw_lengthscale is not None: base_kernel.initialize(raw_lengthscale=initial_raw_lengthscale)
        if initial_raw_alpha is not None: base_kernel.initialize(raw_alpha=initial_raw_alpha)
        scale_kernel_kwargs = {
            'outputscale_prior': gpytorch.priors.SmoothedBoxPrior(0.001, 1000.0, sigma=0.1, transform=torch.exp)
        }
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, **scale_kernel_kwargs)
        if initial_raw_outputscale is not None: self.covar_module.initialize(raw_outputscale=initial_raw_outputscale)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# JSON Encoder (same as v3/v4)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        if isinstance(obj, (gpytorch.priors.Prior, torch.Tensor)): return str(obj)
        if hasattr(obj, 'state_dict'):
            params = {}
            for name, param in obj.named_parameters():
                params[name] = param.detach().cpu().numpy().tolist() if param.numel() > 1 else param.item()
            for attr_name in ['lengthscale', 'alpha', 'outputscale', 'noise']:
                if hasattr(obj, attr_name):
                    attr_val = getattr(obj, attr_name)
                    if isinstance(attr_val, torch.Tensor):
                        val = attr_val.detach().cpu(); params[attr_name] = val.tolist() if val.numel() > 1 else val.item()
                    elif isinstance(attr_val, (float, int)): params[attr_name] = attr_val
            if hasattr(obj, 'base_kernel'):
                if hasattr(obj.base_kernel, 'lengthscale'):
                    val = obj.base_kernel.lengthscale.detach().cpu(); params['base_kernel_lengthscale'] = val.tolist() if val.numel() > 1 else val.item()
                if hasattr(obj.base_kernel, 'alpha'):
                    val = obj.base_kernel.alpha.detach().cpu(); params['base_kernel_alpha'] = val.tolist() if val.numel() > 1 else val.item()
            if not params: return str(obj)
            return params
        return super(NpEncoder, self).default(obj)

def save_final_summary_to_json(output_summary, json_output_path):
    try:
        with open(json_output_path, 'w') as f: json.dump(output_summary, f, indent=4, sort_keys=True, cls=NpEncoder)
        if VERBOSE: print(f"\nFinal CV summary saved to: {json_output_path}")
    except Exception as e:
        print(f"Error saving final summary to JSON: {e}")
        # Simplified fallback for CV summary
        try:
            def fallback_default(obj):
                if isinstance(obj, (np.ndarray, pd.DataFrame)): return str(obj) # Avoid complex objects in top-level summary
                return NpEncoder().default(obj)
            with open(json_output_path, 'w') as f:
                 json.dump(output_summary, f, indent=4, sort_keys=True, default=fallback_default)
            if VERBOSE: print(f"\nFinal CV summary saved to (with fallback): {json_output_path}")
        except Exception as e2: print(f"Fallback JSON saving for CV summary also failed: {e2}")


def load_and_preprocess_data(file_paths, is_training_set=True, feature_columns_list=None):
    """Loads data from specified file paths and preprocesses it."""
    if not file_paths:
        return pd.DataFrame(), pd.Series()

    all_data_frames = [pd.read_csv(f) for f in file_paths]
    combined_df = pd.concat(all_data_frames, ignore_index=True)

    # Filter for active rows
    active_df = combined_df[combined_df[CURVATURE_ACTIVE_COLUMN] == 1].copy()
    
    # Prepare target
    active_df[TARGET_COLUMN] = pd.to_numeric(active_df[TARGET_COLUMN], errors='coerce')
    active_df.dropna(subset=[TARGET_COLUMN], inplace=True)
    active_df[TARGET_RADIUS_COLUMN] = np.where(
        np.isclose(active_df[TARGET_COLUMN], 0) | (active_df[TARGET_COLUMN] < 0),
        RADIUS_CAP_VALUE, 1.0 / active_df[TARGET_COLUMN]
    )
    active_df[TARGET_RADIUS_COLUMN] = np.clip(active_df[TARGET_RADIUS_COLUMN], a_min=None, a_max=RADIUS_CAP_VALUE)

    # Prepare features
    if feature_columns_list is None: # Should be defined globally or passed
        fft_cols = [f"{FFT_PREFIX}{i}Hz" for i in range(FFT_MIN_HZ, FFT_MAX_HZ + FFT_STEP_HZ, FFT_STEP_HZ)]
        feature_columns_list = fft_cols + [POSITION_COLUMN]
    
    missing_cols = [col for col in feature_columns_list if col not in active_df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols} in files: {file_paths}")

    X = active_df[feature_columns_list].copy()
    y = active_df[TARGET_RADIUS_COLUMN].copy()

    # Data Cleaning
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y.replace([np.inf, -np.inf], np.nan, inplace=True)
    rows_to_drop = X.isnull().any(axis=1) | y.isnull()
    X_cleaned = X[~rows_to_drop]
    y_cleaned = y[~rows_to_drop]
    
    if VERBOSE and is_training_set:
        print(f"Loaded {len(file_paths)} files for training. Shape before NaN drop: {X.shape}, After: {X_cleaned.shape}")
    elif VERBOSE and not is_training_set:
        print(f"Loaded {file_paths[0]} for testing. Shape before NaN drop: {X.shape}, After: {X_cleaned.shape}")

    return X_cleaned, y_cleaned


def main_cv():
    if VERBOSE: print(f"--- SGPR Experiment Start (v5 - Leave-One-CSV-Out CV): {RUN_ID} ---")
    if VERBOSE: print(f"Using GPU: {USE_GPU}" if USE_GPU else "Using CPU")

    # --- Stage 1: File Discovery for CV ---
    if VERBOSE: print(f"\n--- Stage 1: Discovering merged CSV files for CV ---")
    search_pattern = os.path.join(INPUT_DATA_DIR, 'merged_*.csv')
    all_csv_files = sorted(glob.glob(search_pattern))
    if not all_csv_files:
        print(f"Error: No 'merged_*.csv' files found in '{INPUT_DATA_DIR}'. Exiting.")
        return
    num_folds = len(all_csv_files)
    if num_folds < 2:
        print(f"Error: Need at least 2 CSV files for Leave-One-Out CV. Found {num_folds}. Exiting.")
        return
    if VERBOSE: print(f"Found {num_folds} CSV files. Will perform {num_folds}-fold Leave-One-CSV-Out CV.")

    # Define feature columns once
    fft_feature_cols = [f"{FFT_PREFIX}{i}Hz" for i in range(FFT_MIN_HZ, FFT_MAX_HZ + FFT_STEP_HZ, FFT_STEP_HZ)]
    global_feature_columns = fft_feature_cols + [POSITION_COLUMN]

    cv_results = defaultdict(list)
    all_fold_predictions = [] # To store (true, pred) for an overall plot if desired

    for fold_idx in range(num_folds):
        if VERBOSE: print(f"\n--- Starting CV Fold {fold_idx + 1}/{num_folds} ---")
        
        # --- Stage 2: Data Splitting for current fold ---
        test_file_path = [all_csv_files[fold_idx]]
        train_file_paths = [f for i, f in enumerate(all_csv_files) if i != fold_idx]
        if VERBOSE: print(f"Test file: {os.path.basename(test_file_path[0])}")
        if VERBOSE: print(f"Training files: {[os.path.basename(f) for f in train_file_paths]}")

        # --- Stages 3-6: Load, Preprocess, Clean Data for current fold ---
        X_train, y_train = load_and_preprocess_data(train_file_paths, is_training_set=True, feature_columns_list=global_feature_columns)
        X_test, y_test = load_and_preprocess_data(test_file_path, is_training_set=False, feature_columns_list=global_feature_columns)

        if X_train.empty or y_train.empty:
            print(f"Warning: Fold {fold_idx+1} has no training data after preprocessing. Skipping fold.")
            cv_results['r2_score'].append(np.nan); cv_results['mae'].append(np.nan); cv_results['rmse'].append(np.nan)
            continue
        if X_test.empty or y_test.empty:
            print(f"Warning: Fold {fold_idx+1} has no test data after preprocessing. Skipping fold.")
            cv_results['r2_score'].append(np.nan); cv_results['mae'].append(np.nan); cv_results['rmse'].append(np.nan)
            continue

        # --- Stage 7: Scaling (Fit on current fold's train data) ---
        if VERBOSE: print(f"\nScaling data for Fold {fold_idx + 1}...")
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        X_train_scaled_np = feature_scaler.fit_transform(X_train)
        y_train_scaled_np = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
        X_test_scaled_np = feature_scaler.transform(X_test)
        # y_test_scaled_np = target_scaler.transform(y_test.values.reshape(-1, 1)) # For inverse transform later

        X_train_tensor = to_tensor(X_train_scaled_np, USE_GPU)
        y_train_tensor = to_tensor(y_train_scaled_np.ravel(), USE_GPU)
        X_test_tensor = to_tensor(X_test_scaled_np, USE_GPU)

        # --- Stage 8: SGPR Model Definition ---
        if VERBOSE: print(f"\nDefining SGPR Model for Fold {fold_idx + 1}...")
        num_train_samples_fold = X_train_tensor.size(0)
        current_num_inducing_fold = min(NUM_INDUCING_POINTS_TARGET, num_train_samples_fold -1 if num_train_samples_fold >1 else 1)

        if current_num_inducing_fold <=0:
            print(f"Warning: Not enough training samples ({num_train_samples_fold}) in fold {fold_idx+1} for inducing points. Skipping fold.")
            cv_results['r2_score'].append(np.nan); cv_results['mae'].append(np.nan); cv_results['rmse'].append(np.nan)
            continue
            
        if VERBOSE: print(f"Initializing {current_num_inducing_fold} inducing points using K-Means for Fold {fold_idx + 1}...")
        kmeans = KMeans(n_clusters=current_num_inducing_fold, random_state=SEED, n_init='auto', verbose=0)
        kmeans.fit(X_train_scaled_np) # Fit K-Means on the current fold's training data
        inducing_points = to_tensor(kmeans.cluster_centers_, USE_GPU)

        init_raw_lengthscale, init_raw_alpha, init_raw_outputscale = None, None, None
        if INITIALIZE_FROM_SKLEARN_HPARAMS:
            init_raw_lengthscale = torch.tensor(np.log(SKLEARN_BEST_LENGTHSCALE)).float()
            init_raw_alpha = torch.tensor(np.log(SKLEARN_BEST_ALPHA)).float()
            init_raw_outputscale = torch.tensor(np.log(SKLEARN_BEST_OUTPUTSCALE)).float()
            if USE_GPU:
                init_raw_lengthscale=init_raw_lengthscale.cuda(); init_raw_alpha=init_raw_alpha.cuda(); init_raw_outputscale=init_raw_outputscale.cuda()
        
        model = ApproximateGPModel(inducing_points, init_raw_lengthscale, init_raw_alpha, init_raw_outputscale)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=gpytorch.priors.GammaPrior(concentration=1.1, rate=10.0),
            noise_constraint=gpytorch.constraints.GreaterThan(1e-7)
        )
        if USE_GPU: model = model.cuda(); likelihood = likelihood.cuda()

        # --- Stage 9: SGPR Model Training for current fold ---
        if VERBOSE: print(f"\nTraining SGPR Model for Fold {fold_idx + 1} ({TRAINING_ITERATIONS} iterations)...")
        model.train(); likelihood.train()
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=LEARNING_RATE)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train_tensor.size(0))

        fold_training_start_time = time.time()
        for i in range(TRAINING_ITERATIONS):
            optimizer.zero_grad(); output = model(X_train_tensor); loss = -mll(output, y_train_tensor); loss.backward()
            if VERBOSE and (i + 1) % 100 == 0: # Log less frequently during CV
                print(f"Fold {fold_idx+1}, Iter {i+1}/{TRAINING_ITERATIONS} - Loss: {loss.item():.3f}, Noise: {likelihood.noise.item():.6f}")
            optimizer.step()
        fold_training_time = time.time() - fold_training_start_time
        if VERBOSE: print(f"Fold {fold_idx+1} training complete in {fold_training_time:.2f}s. Final ELBO: {-loss.item():.3f}")

        # --- Stage 10 & 11: Prediction and Descaling for current fold ---
        model.eval(); likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(X_test_tensor))
            y_pred_scaled_tensor = observed_pred.mean
        y_pred_scaled_np_fold = y_pred_scaled_tensor.cpu().numpy().reshape(-1, 1)
        
        y_pred_eval_fold = target_scaler.inverse_transform(y_pred_scaled_np_fold).ravel()
        y_true_eval_fold = y_test.values.ravel() # y_test is already the true, original scale values

        all_fold_predictions.extend(list(zip(y_true_eval_fold, y_pred_eval_fold)))


        # --- Stage 12: Evaluation for current fold ---
        r2_fold = r2_score(y_true_eval_fold, y_pred_eval_fold)
        mae_fold = mean_absolute_error(y_true_eval_fold, y_pred_eval_fold)
        rmse_fold = np.sqrt(mean_squared_error(y_true_eval_fold, y_pred_eval_fold))
        
        if VERBOSE:
            print(f"\n--- Fold {fold_idx + 1} Performance ---")
            print(f"  Test File: {os.path.basename(test_file_path[0])}")
            print(f"  R² Score: {r2_fold:.4f}")
            print(f"  MAE: {mae_fold:.2f} mm")
            print(f"  RMSE: {rmse_fold:.2f} mm")
            print(f"  Optimized Noise for fold: {likelihood.noise.item():.6f}")

        cv_results['fold'].append(fold_idx + 1)
        cv_results['test_file'].append(os.path.basename(test_file_path[0]))
        cv_results['r2_score'].append(r2_fold)
        cv_results['mae'].append(mae_fold)
        cv_results['rmse'].append(rmse_fold)
        cv_results['learned_noise'].append(likelihood.noise.item())
        cv_results['training_time_seconds'].append(fold_training_time)
        cv_results['final_elbo'].append(-loss.item())

        # Optional: Save per-fold plots
        if SAVE_PLOTS_PER_FOLD:
            plt.style.use(PLOT_STYLE)
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true_eval_fold, y_pred_eval_fold, alpha=0.6, edgecolors='k', s=50)
            min_val_f = min(y_true_eval_fold.min(), y_pred_eval_fold.min())*0.95
            max_val_f = max(y_true_eval_fold.max(), y_pred_eval_fold.max())*1.05
            plt.plot([min_val_f, max_val_f], [min_val_f, max_val_f], 'r--', lw=2)
            plt.xlabel(f"True {TARGET_RADIUS_COLUMN} (mm)")
            plt.ylabel(f"Predicted {TARGET_RADIUS_COLUMN} (mm)")
            plt.title(f"Fold {fold_idx+1} ({os.path.basename(test_file_path[0])}): R²={r2_fold:.3f}")
            plt.grid(True)
            plt.savefig(os.path.join(OUTPUT_DIR, f"{RUN_ID}_fold_{fold_idx+1}_true_vs_pred.png"))
            plt.close()
        
        # Clear GPU memory if used
        if USE_GPU:
            del model, likelihood, X_train_tensor, y_train_tensor, X_test_tensor, optimizer, mll, output, loss, observed_pred, y_pred_scaled_tensor
            torch.cuda.empty_cache()


    # --- After all folds: Aggregate and Report CV Results ---
    if VERBOSE: print("\n\n--- Overall Cross-Validation Results ---")
    cv_results_df = pd.DataFrame(cv_results)
    if VERBOSE: print(cv_results_df.to_string(index=False))

    mean_r2 = np.nanmean(cv_results_df['r2_score'])
    mean_mae = np.nanmean(cv_results_df['mae'])
    mean_rmse = np.nanmean(cv_results_df['rmse'])
    std_r2 = np.nanstd(cv_results_df['r2_score'])
    std_mae = np.nanstd(cv_results_df['mae'])
    std_rmse = np.nanstd(cv_results_df['rmse'])

    if VERBOSE:
        print(f"\nAverage R² Score: {mean_r2:.4f} ± {std_r2:.4f}")
        print(f"Average MAE: {mean_mae:.2f} mm ± {std_mae:.2f} mm")
        print(f"Average RMSE: {mean_rmse:.2f} mm ± {std_rmse:.2f} mm")
        print(f"Average Learned Noise: {np.nanmean(cv_results_df['learned_noise']):.6f}")

    # Save overall summary
    final_summary = {
        'run_id': RUN_ID,
        'model_type': "SparseGaussianProcessRegressor_GPyTorch_v5_CV",
        'script_variant': os.path.basename(__file__),
        'num_cv_folds': num_folds,
        'sgpr_config': {
            'num_inducing_points_target': NUM_INDUCING_POINTS_TARGET,
            'training_iterations_per_fold': TRAINING_ITERATIONS,
            'learning_rate': LEARNING_RATE,
            'kernel_initialization_from_sklearn': INITIALIZE_FROM_SKLEARN_HPARAMS,
            'sklearn_hparams_for_init': {
                'lengthscale': SKLEARN_BEST_LENGTHSCALE,
                'alpha': SKLEARN_BEST_ALPHA,
                'outputscale': SKLEARN_BEST_OUTPUTSCALE
            },
            'noise_handling': 'GaussianLikelihood with GammaPrior(1.1, 10.0) and GreaterThan(1e-7) constraint'
        },
        'cv_fold_results': cv_results_df.to_dict(orient='list'),
        'cv_aggregated_metrics': {
            'mean_r2_score': mean_r2, 'std_r2_score': std_r2,
            'mean_mae_mm': mean_mae, 'std_mae_mm': std_mae,
            'mean_rmse_mm': mean_rmse, 'std_rmse_mm': std_rmse,
            'mean_learned_noise': np.nanmean(cv_results_df['learned_noise']),
            'mean_training_time_seconds': np.nanmean(cv_results_df['training_time_seconds']),
            'mean_final_elbo': np.nanmean(cv_results_df['final_elbo'])
        }
    }
    
    # Overall True vs Predicted Plot from all folds
    if all_fold_predictions:
        y_true_all_folds, y_pred_all_folds = zip(*all_fold_predictions)
        y_true_all_folds = np.array(y_true_all_folds)
        y_pred_all_folds = np.array(y_pred_all_folds)
        
        plt.style.use(PLOT_STYLE)
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true_all_folds, y_pred_all_folds, alpha=0.3, edgecolors='k', s=30, label='Per-Fold Predictions')
        
        # Calculate overall metrics for this combined plot (for title only)
        overall_r2 = r2_score(y_true_all_folds, y_pred_all_folds)
        overall_mae = mean_absolute_error(y_true_all_folds, y_pred_all_folds)

        min_val_overall = min(y_true_all_folds.min(), y_pred_all_folds.min()) * 0.95
        max_val_overall = max(y_true_all_folds.max(), y_pred_all_folds.max()) * 1.05
        plt.plot([min_val_overall, max_val_overall], [min_val_overall, max_val_overall], 'r--', lw=2, label='Ideal Fit (y=x)')
        plt.xlabel(f"True {TARGET_RADIUS_COLUMN} (mm)")
        plt.ylabel(f"Predicted {TARGET_RADIUS_COLUMN} (mm)")
        plt.title(f"SGPR (v5) CV: All Folds True vs. Predicted (Overall R²: {overall_r2:.3f}, MAE: {overall_mae:.2f} mm)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{RUN_ID}_ALL_FOLDS_true_vs_predicted.png"))
        if SHOW_PLOTS: plt.show()
        plt.close()
        final_summary['overall_plot_metrics_for_title_only'] = {'r2': overall_r2, 'mae': overall_mae}


    json_output_path = os.path.join(OUTPUT_DIR, f"{RUN_ID}_CV_summary.json")
    save_final_summary_to_json(final_summary, json_output_path)

    if VERBOSE: print(f"\n--- SGPR CV Experiment End (v5): {RUN_ID} ---")

if __name__ == '__main__':
    main_cv()
