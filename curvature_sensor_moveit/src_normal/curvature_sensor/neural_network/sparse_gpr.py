"""
sparse_gpr_gpytorch_curvature_v3.py

An end-to-end machine learning pipeline for predicting sensor radius using
Sparse Gaussian Process Regression (SGPR) with GPyTorch.
This version includes fixes for parameter initialization when using priors
with transforms.

Key improvements in v3:
- Corrected initialization of kernel hyperparameters (lengthscale, alpha, outputscale)
  by setting their 'raw' versions when INITIALIZE_FROM_SKLEARN_HPARAMS is True.
- Added 'global NUM_INDUCING_POINTS' to main to address potential UnboundLocalError.

Pipeline Steps:
1.  Configuration.
2.  File Discovery.
3.  Load and Combine Data.
4.  Filter for Active Rows and Prepare Target (Radius_mm).
5.  Prepare Features (X) and Target (y).
6.  Data Cleaning.
7.  Train-Test Split.
8.  Scaling (Features and Target) and Tensor Conversion.
9.  SGPR Model Definition (GPyTorch).
10. SGPR Model Training (Variational Inference).
11. Prediction.
12. Descaling.
13. Evaluation.
14. Detailed Error Analysis.
15. Save outputs (metrics, configuration, model state, plots).
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

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans # For inducing point initialization

# GPyTorch and PyTorch imports
import torch
import gpytorch

# --- Stage 0: Configuration ---
VERBOSE = True
SEED = 42 # For reproducibility

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

# SGPR Specific Configuration (GPyTorch)
NUM_INDUCING_POINTS = 800
TRAINING_ITERATIONS = 500
LEARNING_RATE = 0.01
USE_GPU = torch.cuda.is_available()

# Option to initialize kernel parameters
INITIALIZE_FROM_SKLEARN_HPARAMS = True
SKLEARN_BEST_LENGTHSCALE = 1.80
SKLEARN_BEST_ALPHA = 28.90
SKLEARN_BEST_OUTPUTSCALE = 0.155
SKLEARN_BEST_NOISE = 1e-5

# Output directory
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_BASE_DIR = 'model_outputs'
RUN_ID = f"run_{TIMESTAMP}_sgpr_v3_rq_radius_pos_widefft_full"
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, RUN_ID)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Plotting
PLOT_STYLE = 'seaborn-v0_8-whitegrid'
SAVE_PLOTS = True
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

# --- Stage 1: Define GPyTorch SGPR Model ---
class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, initial_raw_lengthscale=None, initial_raw_alpha=None, initial_raw_outputscale=None):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )
        super(ApproximateGPModel, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()

        # Define base kernel (RQKernel)
        base_kernel_kwargs = {
            'lengthscale_prior': gpytorch.priors.SmoothedBoxPrior(0.01, 100.0, sigma=0.1, transform=torch.exp),
            'alpha_prior': gpytorch.priors.SmoothedBoxPrior(0.01, 100.0, sigma=0.1,  transform=torch.exp)
        }
        base_kernel = gpytorch.kernels.RQKernel(**base_kernel_kwargs)

        # Initialize raw parameters if provided
        if initial_raw_lengthscale is not None:
            base_kernel.initialize(raw_lengthscale=initial_raw_lengthscale)
        if initial_raw_alpha is not None:
            # RQKernel might not have a direct 'raw_alpha' setter in initialize,
            # but lengthscale is the primary one. Let's check GPyTorch docs or try setting .alpha
            # For now, let's assume direct assignment after prior setup is okay if initialize doesn't take raw_alpha
            # Or, if alpha also has a constraint/transform, it would need similar raw initialization.
            # The prior for alpha also uses torch.exp, so it should be a raw value.
             base_kernel.initialize(raw_alpha=initial_raw_alpha)


        # Define ScaleKernel
        scale_kernel_kwargs = {
            'outputscale_prior': gpytorch.priors.SmoothedBoxPrior(0.001, 1000.0, sigma=0.1, transform=torch.exp)
        }
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, **scale_kernel_kwargs)
        
        if initial_raw_outputscale is not None:
            self.covar_module.initialize(raw_outputscale=initial_raw_outputscale)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# JSON Encoder
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
            for attr_name in ['lengthscale', 'alpha', 'outputscale', 'noise']: # Check for common kernel params
                if hasattr(obj, attr_name):
                    attr_val = getattr(obj, attr_name)
                    if isinstance(attr_val, torch.Tensor):
                        val = attr_val.detach().cpu()
                        params[attr_name] = val.tolist() if val.numel() > 1 else val.item()
                    elif isinstance(attr_val, (float, int)): # if it's already a Python number
                        params[attr_name] = attr_val

            # Specifically for ScaleKernel's base_kernel attributes
            if hasattr(obj, 'base_kernel'):
                if hasattr(obj.base_kernel, 'lengthscale'):
                    val = obj.base_kernel.lengthscale.detach().cpu()
                    params['base_kernel_lengthscale'] = val.tolist() if val.numel() > 1 else val.item()
                if hasattr(obj.base_kernel, 'alpha'):
                    val = obj.base_kernel.alpha.detach().cpu()
                    params['base_kernel_alpha'] = val.tolist() if val.numel() > 1 else val.item()
            if not params: return str(obj)
            return params
        return super(NpEncoder, self).default(obj)

def save_metrics_and_config_to_json(output_summary, json_output_path):
    try:
        with open(json_output_path, 'w') as f:
            json.dump(output_summary, f, indent=4, sort_keys=True, cls=NpEncoder)
        if VERBOSE: print(f"\nMetrics and configuration saved to: {json_output_path}")
    except Exception as e:
        print(f"Error saving metrics to JSON: {e}")
        try:
            def fallback_default(obj):
                if isinstance(obj, (torch.Tensor, gpytorch.Module)): return str(obj)
                return NpEncoder().default(obj)
            with open(json_output_path, 'w') as f:
                 json.dump(output_summary, f, indent=4, sort_keys=True, default=fallback_default)
            if VERBOSE: print(f"\nMetrics and configuration saved to (with fallback): {json_output_path}")
        except Exception as e2:
            print(f"Fallback JSON saving also failed: {e2}")

def main():
    global NUM_INDUCING_POINTS # Ensure we can modify if num_inducing > num_train_samples

    if VERBOSE: print(f"--- SGPR Experiment Start (v3): {RUN_ID} ---")
    if VERBOSE: print(f"Using GPU: {USE_GPU}" if USE_GPU else "Using CPU")

    # --- Stage 2: File Discovery ---
    if VERBOSE: print(f"\n--- Stage 2: Discovering merged CSV files ---")
    search_pattern = os.path.join(INPUT_DATA_DIR, 'merged_*.csv')
    merged_files = sorted(glob.glob(search_pattern))
    if not merged_files: print(f"Error: No 'merged_*.csv' files found in '{INPUT_DATA_DIR}'."); return
    if VERBOSE: print(f"Found {len(merged_files)} merged CSV files.")

    # --- Stage 3: Load and Combine Data ---
    if VERBOSE: print(f"\n--- Stage 3: Loading and Combining Data ---")
    all_data_frames = [pd.read_csv(f) for f in merged_files]
    combined_df = pd.concat(all_data_frames, ignore_index=True)
    output_summary = {'data_summary': {'combined_data_shape_before_processing_active_filter': list(combined_df.shape),
                                       'total_files_processed': len(merged_files)}}

    # --- Stage 4: Filter for Active Rows and Prepare Target (Radius_mm) ---
    if VERBOSE: print(f"\n--- Stage 4: Filtering Active Rows & Preparing Target ---")
    active_df = combined_df[combined_df[CURVATURE_ACTIVE_COLUMN] == 1].copy()
    active_df[TARGET_COLUMN] = pd.to_numeric(active_df[TARGET_COLUMN], errors='coerce')
    active_df.dropna(subset=[TARGET_COLUMN], inplace=True)
    active_df[TARGET_RADIUS_COLUMN] = np.where(
        np.isclose(active_df[TARGET_COLUMN], 0) | (active_df[TARGET_COLUMN] < 0),
        RADIUS_CAP_VALUE, 1.0 / active_df[TARGET_COLUMN]
    )
    active_df[TARGET_RADIUS_COLUMN] = np.clip(active_df[TARGET_RADIUS_COLUMN], a_min=None, a_max=RADIUS_CAP_VALUE)
    if VERBOSE: print(f"Calculated '{TARGET_RADIUS_COLUMN}'. Min: {active_df[TARGET_RADIUS_COLUMN].min():.2f}, Max: {active_df[TARGET_RADIUS_COLUMN].max():.2f}")

    # --- Stage 5: Prepare Features (X) and Target (y) ---
    if VERBOSE: print(f"\n--- Stage 5: Preparing Features (X) and Target (y) ---")
    fft_columns = [f"{FFT_PREFIX}{i}Hz" for i in range(FFT_MIN_HZ, FFT_MAX_HZ + FFT_STEP_HZ, FFT_STEP_HZ)]
    feature_columns = fft_columns + [POSITION_COLUMN]
    missing_cols = [col for col in feature_columns if col not in active_df.columns];
    if missing_cols: print(f"Error: Missing feature columns: {missing_cols}. Exiting."); return
    X = active_df[feature_columns].copy(); y = active_df[TARGET_RADIUS_COLUMN].copy()
    output_summary['input_features_used'] = feature_columns

    # --- Stage 6: Data Cleaning ---
    if VERBOSE: print(f"\n--- Stage 6: Data Cleaning (NaN/Inf) ---")
    X.replace([np.inf, -np.inf], np.nan, inplace=True); y.replace([np.inf, -np.inf], np.nan, inplace=True)
    rows_to_drop = X.isnull().any(axis=1) | y.isnull()
    X_cleaned = X[~rows_to_drop]; y_cleaned = y[~rows_to_drop]
    if VERBOSE: print(f"Dropped {rows_to_drop.sum()} rows. Final data shape (X): {X_cleaned.shape}, (y): {y_cleaned.shape}")
    output_summary['data_summary']['final_processed_data_shape_for_modeling'] = [list(X_cleaned.shape), list(y_cleaned.shape)]
    if X_cleaned.empty: print("Error: No data left after cleaning. Exiting."); return

    # --- Stage 7: Train-Test Split ---
    if VERBOSE: print(f"\n--- Stage 7: Train-Test Split (80/20) ---")
    X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=SEED)
    output_summary['data_summary']['training_set_shape'] = [list(X_train.shape), list(y_train.shape)]
    output_summary['data_summary']['test_set_shape'] = [list(X_test.shape), list(y_test.shape)]

    # --- Stage 8: Scaling and Tensor Conversion ---
    if VERBOSE: print(f"\n--- Stage 8: Scaling Data and Converting to Tensors ---")
    feature_scaler = MinMaxScaler(feature_range=(0, 1)); target_scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled_np = feature_scaler.fit_transform(X_train)
    y_train_scaled_np = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled_np = feature_scaler.transform(X_test)
    y_test_scaled_np = target_scaler.transform(y_test.values.reshape(-1, 1))
    X_train_tensor = to_tensor(X_train_scaled_np, USE_GPU); y_train_tensor = to_tensor(y_train_scaled_np.ravel(), USE_GPU)
    X_test_tensor = to_tensor(X_test_scaled_np, USE_GPU)

    # --- Stage 9: SGPR Model Definition ---
    if VERBOSE: print(f"\n--- Stage 9: Defining GPyTorch SGPR Model ---")
    num_train_samples = X_train_tensor.size(0)
    current_num_inducing = NUM_INDUCING_POINTS # Use a temporary variable for K-means
    if current_num_inducing >= num_train_samples: # Check if actual num_inducing needs adjustment
        if VERBOSE: print(f"Warning: NUM_INDUCING_POINTS ({current_num_inducing}) >= num_train_samples ({num_train_samples}). Using num_train_samples -1 as inducing points for K-Means.")
        current_num_inducing = num_train_samples - 1 if num_train_samples > 1 else 1


    if current_num_inducing < num_train_samples and current_num_inducing > 0 : # Ensure k-means can run
        if VERBOSE: print(f"Initializing {current_num_inducing} inducing points using K-Means...")
        kmeans = KMeans(n_clusters=current_num_inducing, random_state=SEED, n_init='auto', verbose=0)
        kmeans.fit(X_train_scaled_np)
        inducing_points = to_tensor(kmeans.cluster_centers_, USE_GPU)
    else: # Fallback to random if k-means is not suitable or if using all points
        if VERBOSE: print(f"Using random initialization for inducing points or all training points.")
        perm = torch.randperm(num_train_samples)
        inducing_points = X_train_tensor[perm[:current_num_inducing], :].clone()

    if inducing_points.size(0) == 0 and num_train_samples > 0 : # Safety for k-means failing with small N
         inducing_points = X_train_tensor[0:1, :].clone() # take first point
         if VERBOSE: print(f"Warning: K-means resulted in 0 inducing points. Defaulting to 1 inducing point.")


    NUM_INDUCING_POINTS = inducing_points.size(0) # Update actual number of inducing points used
    if VERBOSE: print(f"Initialized {NUM_INDUCING_POINTS} inducing points.")
    output_summary['sgpr_num_inducing_points'] = NUM_INDUCING_POINTS


    init_raw_lengthscale, init_raw_alpha, init_raw_outputscale = None, None, None
    if INITIALIZE_FROM_SKLEARN_HPARAMS:
        init_raw_lengthscale = torch.tensor(np.log(SKLEARN_BEST_LENGTHSCALE)).float()
        init_raw_alpha = torch.tensor(np.log(SKLEARN_BEST_ALPHA)).float()
        init_raw_outputscale = torch.tensor(np.log(SKLEARN_BEST_OUTPUTSCALE)).float()
        if USE_GPU:
            init_raw_lengthscale = init_raw_lengthscale.cuda(); init_raw_alpha = init_raw_alpha.cuda(); init_raw_outputscale = init_raw_outputscale.cuda()
        if VERBOSE: print("Initializing kernel raw parameters from scikit-learn best hparams.")

    model = ApproximateGPModel(inducing_points, init_raw_lengthscale, init_raw_alpha, init_raw_outputscale)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=gpytorch.priors.GammaPrior(concentration=1.1, rate=10.0), # Mean around 0.11, encourages smaller noise
        noise_constraint=gpytorch.constraints.GreaterThan(1e-7) # Constraint for numerical stability
    )
    if INITIALIZE_FROM_SKLEARN_HPARAMS and SKLEARN_BEST_NOISE is not None:
        # Initialize raw_noise directly if possible, considering the constraint
        # The GammaPrior makes direct initialization of the 'transformed' noise value tricky.
        # We'll rely on the prior and constraint to guide it towards a small value.
        # If you were using a prior with an exp transform, you'd do:
        # likelihood.initialize(raw_noise=likelihood.noise_prior.inverse_transform(torch.tensor(SKLEARN_BEST_NOISE).float()))
        # For now, let the GammaPrior and constraint handle it.
        # Or, to be more direct without a prior on noise for initialization:
        # likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-7))
        # likelihood.noise = torch.tensor([SKLEARN_BEST_NOISE]) # if scalar
        pass


    if USE_GPU: model = model.cuda(); likelihood = likelihood.cuda()
    output_summary['gpr_kernel_initial_config'] = {
        'model_type': model.__class__.__name__, 'likelihood': likelihood.__class__.__name__,
        'kernel_type': model.covar_module.__class__.__name__, 'base_kernel_type': model.covar_module.base_kernel.__class__.__name__,
        'num_inducing_points': NUM_INDUCING_POINTS,
        'initial_lengthscale_val': model.covar_module.base_kernel.lengthscale.item(),
        'initial_alpha_val': model.covar_module.base_kernel.alpha.item(),
        'initial_outputscale_val': model.covar_module.outputscale.item(),
        'initial_noise_val': likelihood.noise.item(),
        'initialized_from_sklearn': INITIALIZE_FROM_SKLEARN_HPARAMS
    }

    # --- Stage 10: SGPR Model Training ---
    if VERBOSE: print(f"\n--- Stage 10: Training GPyTorch SGPR Model ---")
    if VERBOSE: print(f"Training for {TRAINING_ITERATIONS} iterations with LR={LEARNING_RATE}.")
    model.train(); likelihood.train()
    # Include likelihood parameters in the optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=LEARNING_RATE)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train_tensor.size(0))

    training_start_time = time.time()
    for i in range(TRAINING_ITERATIONS):
        optimizer.zero_grad(); output = model(X_train_tensor); loss = -mll(output, y_train_tensor); loss.backward()
        if VERBOSE and (i + 1) % 20 == 0:
            print(f'Iter {i+1}/{TRAINING_ITERATIONS} - Loss: {loss.item():.3f} ' +
                  f'lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f} ' +
                  f'alpha: {model.covar_module.base_kernel.alpha.item():.3f} ' +
                  f'outputscale: {model.covar_module.outputscale.item():.3f} ' +
                  f'noise: {likelihood.noise.item():.6f}') # Increased precision for noise
        optimizer.step()
    training_time = time.time() - training_start_time
    if VERBOSE: print(f"Training complete in {training_time:.2f} seconds.")
    output_summary['training_time_seconds'] = training_time
    output_summary['log_marginal_likelihood_equivalent_ELBO'] = -loss.item() # Final ELBO
    output_summary['gpr_kernel_optimized_config'] = {
        'optimized_lengthscale': model.covar_module.base_kernel.lengthscale.item(),
        'optimized_alpha': model.covar_module.base_kernel.alpha.item(),
        'optimized_outputscale': model.covar_module.outputscale.item(),
        'optimized_noise': likelihood.noise.item()
    }

    # --- Stage 11: Prediction ---
    if VERBOSE: print(f"\n--- Stage 11: Making Predictions on Test Set ---")
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_test_tensor))
        y_pred_scaled_tensor = observed_pred.mean
    y_pred_scaled_np = y_pred_scaled_tensor.cpu().numpy().reshape(-1, 1)

    # --- Stage 12: Descaling ---
    if VERBOSE: print(f"\n--- Stage 12: Descaling Predictions ---")
    y_pred_eval = target_scaler.inverse_transform(y_pred_scaled_np).ravel()
    y_true_eval = target_scaler.inverse_transform(y_test_scaled_np).ravel()

    # --- Stage 13: Evaluation ---
    if VERBOSE: print(f"\n--- Stage 13: Evaluating Model Performance ---")
    r2 = r2_score(y_true_eval, y_pred_eval); mae = mean_absolute_error(y_true_eval, y_pred_eval)
    rmse = np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))
    if VERBOSE: print(f"R²: {r2:.4f}, MAE: {mae:.2f} mm, RMSE: {rmse:.2f} mm")
    output_summary['performance_metrics_on_radius'] = {'r2_score': r2, 'mae': mae, 'rmse': rmse}

    # --- Stage 14: Detailed Error Analysis & Plotting ---
    # (Plotting and segmented error analysis code remains the same as v2)
    if VERBOSE: print(f"\n--- Stage 14: Detailed Error Analysis & Plotting ---")
    plt.style.use(PLOT_STYLE)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true_eval, y_pred_eval, alpha=0.5, edgecolors='k', s=50)
    min_val = min(y_true_eval.min(), y_pred_eval.min())*0.95; max_val = max(y_true_eval.max(), y_pred_eval.max())*1.05
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Fit (y=x)')
    plt.xlabel(f"True {TARGET_RADIUS_COLUMN} (mm)"); plt.ylabel(f"Predicted {TARGET_RADIUS_COLUMN} (mm)")
    plt.title(f"SGPR (v3): True vs. Predicted (R²: {r2:.3f}, MAE: {mae:.2f} mm)"); plt.legend(); plt.grid(True)
    if SAVE_PLOTS: plt.savefig(os.path.join(OUTPUT_DIR, f"{RUN_ID}_true_vs_predicted.png"))
    if SHOW_PLOTS: plt.show(); plt.close()

    residuals = y_true_eval - y_pred_eval
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_eval, residuals, alpha=0.5, edgecolors='k', s=50); plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.xlabel(f"Predicted {TARGET_RADIUS_COLUMN} (mm)"); plt.ylabel("Residuals (mm)")
    plt.title("SGPR (v3): Residual Plot"); plt.grid(True)
    if SAVE_PLOTS: plt.savefig(os.path.join(OUTPUT_DIR, f"{RUN_ID}_residual_plot.png"))
    if SHOW_PLOTS: plt.show(); plt.close()

    min_true_rad, max_true_rad = y_true_eval.min(), y_true_eval.max()
    radius_segments = [(min_true_rad, 56), (56, 92), (92, 128), (128, 164), (164, max_true_rad + 0.1)]
    radius_segments = [s for s in radius_segments if s[0] < s[1]] # Ensure start < end
    segmented_errors = []
    if VERBOSE: print("\nSegmented MAE Analysis:")
    for r_min, r_max in radius_segments:
        mask = (y_true_eval >= r_min) & (y_true_eval < r_max)
        if np.sum(mask) > 0:
            seg_true, seg_pred = y_true_eval[mask], y_pred_eval[mask]
            info = {"Segment": f"{r_min:.1f}-{r_max:.1f}", "Count": int(np.sum(mask)),
                    "MAE": mean_absolute_error(seg_true, seg_pred),
                    "RMSE": np.sqrt(mean_squared_error(seg_true, seg_pred)),
                    "Mean_Error (Bias)": np.mean(seg_pred - seg_true)}
            segmented_errors.append(info)
            if VERBOSE: print(f"  {info['Segment']} mm ({info['Count']} samples): MAE = {info['MAE']:.2f} mm, Bias = {info['Mean_Error (Bias)']:.2f} mm")
    output_summary['segmented_radius_errors'] = segmented_errors


    # --- Stage 15: Save Outputs ---
    if VERBOSE: print(f"\n--- Stage 15: Saving Outputs ---")
    model_path = os.path.join(OUTPUT_DIR, f"{RUN_ID}_sgpr_model.pth")
    torch.save({'model_state_dict': model.state_dict(), 'likelihood_state_dict': likelihood.state_dict(),
                'feature_scaler': feature_scaler, 'target_scaler': target_scaler,
                'feature_columns': feature_columns, 'num_inducing_points': NUM_INDUCING_POINTS}, model_path)
    if VERBOSE: print(f"Saved GPyTorch model and scalers to {model_path}")

    output_summary['model_type'] = "SparseGaussianProcessRegressor_GPyTorch_v3"
    output_summary['script_variant'] = os.path.basename(__file__)
    output_summary['output_directory'] = OUTPUT_DIR
    json_output_path = os.path.join(OUTPUT_DIR, f"{RUN_ID}_summary.json")
    save_metrics_and_config_to_json(output_summary, json_output_path)

    if VERBOSE: print(f"\n--- SGPR Experiment End (v3): {RUN_ID} ---")

if __name__ == '__main__':
    main()
