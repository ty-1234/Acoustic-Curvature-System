import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Set seed for reproducibility
np.random.seed(42)

# === Setup paths ===
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(parent_dir, "csv_data", "preprocessed", "preprocessed_training_dataset.csv")
timestamp = datetime.now().strftime("%-I:%M%p").lower()
output_dir = os.path.join(parent_dir, "neural_network", "model_outputs", "gpr_curvature")
os.makedirs(output_dir, exist_ok=True)

# === Load dataset and filter only active curvature segments ===
print("\n📥 Loading dataset...")
df = pd.read_csv(dataset_path)
df = df[df["Curvature_Active"] == 1]
print(f"Dataset shape after filtering active curvature segments: {df.shape}")

# === Features and Targets ===
feature_cols = [col for col in df.columns if col.startswith("FFT_")]
print(f"Number of FFT features: {len(feature_cols)}")
X = df[feature_cols]
y = df["Curvature_Label"]
groups = df["RunID"]
unique_groups = groups.unique()
print(f"Number of unique runs (groups): {len(unique_groups)}")

# === Scale features ===
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

# === Define GPR model ===
kernel = RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=1e-5)
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)

# === Grouped Cross-validation ===
# Adjust n_splits to match the number of unique groups
n_splits = min(3, len(unique_groups))  # Use either 3 or the number of groups, whichever is smaller
gkf = GroupKFold(n_splits=n_splits)
rmse_list, mae_list, maxerr_list, r2_list = [], [], [], []

print(f"\n📊 Evaluating GPR with GroupKFold (n_splits={n_splits})...")
start_time = time.time()

all_test_indices = []
all_predictions = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
    fold_start = time.time()
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"Fold {fold+1}: Training samples={len(X_train)}, Test samples={len(X_test)}")
    print(f"Training groups: {np.unique(groups.iloc[train_idx])}")
    print(f"Testing groups: {np.unique(groups.iloc[test_idx])}")
    
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)
    
    # Store test indices and predictions for CSV export
    all_test_indices.extend(test_idx)
    all_predictions.extend(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    maxerr = max_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    rmse_list.append(rmse)
    mae_list.append(mae)
    maxerr_list.append(maxerr)
    r2_list.append(r2)

    fold_time = time.time() - fold_start
    print(f"Fold {fold+1}: RMSE={rmse:.5f}, MAE={mae:.5f}, MaxError={maxerr:.5f}, R²={r2:.3f}")
    print(f"Fold {fold+1} completed in {fold_time:.2f} seconds\n")

cv_time = time.time() - start_time
print(f"Cross-validation completed in {cv_time:.2f} seconds")

# === Print mean performance metrics ===
print("\n=== Mean Cross-Validation Metrics ===")
print(f"Mean RMSE: {np.mean(rmse_list):.5f} ± {np.std(rmse_list):.5f}")
print(f"Mean MAE: {np.mean(mae_list):.5f} ± {np.std(mae_list):.5f}")
print(f"Mean MaxError: {np.mean(maxerr_list):.5f} ± {np.std(maxerr_list):.5f}")
print(f"Mean R²: {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")

# === Retrain on full dataset ===
print("\n🔄 Retraining on full dataset...")
full_train_start = time.time()
gpr.fit(X_scaled, y)
print(f"Full dataset training completed in {time.time() - full_train_start:.2f} seconds")

# Save the model
model_path = os.path.join(output_dir, "gpr_curvature_model.pkl")
joblib.dump(gpr, model_path)
print(f"Model saved to: {model_path}")

# === Save metrics ===
metrics = {
    "model": "GPR_Curvature",
    "rmse_mean": float(np.mean(rmse_list)),
    "rmse_std": float(np.std(rmse_list)),
    "mae_mean": float(np.mean(mae_list)),
    "mae_std": float(np.std(mae_list)),
    "max_error_mean": float(np.mean(maxerr_list)),
    "max_error_std": float(np.std(maxerr_list)),
    "r2_mean": float(np.mean(r2_list)),
    "r2_std": float(np.std(r2_list)),
    "kernel": str(gpr.kernel_),
    "training_points": len(X),
    "feature_count": len(feature_cols),
    "run_count": len(unique_groups),
    "n_splits_used": n_splits,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

metrics_df = pd.DataFrame({
    "Fold": list(range(1, len(rmse_list) + 1)),
    "RMSE": rmse_list,
    "MAE": mae_list,
    "MaxError": maxerr_list,
    "R2": r2_list
})

metrics_df.to_csv(os.path.join(output_dir, "gpr_per_fold_metrics.csv"), index=False)

with open(os.path.join(output_dir, "metrics_summary.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# === Plot: True vs Predicted Curvature ===
y_pred_all = gpr.predict(X_scaled)
plt.figure(figsize=(10, 8))
plt.scatter(y, y_pred_all, alpha=0.4, edgecolors='k', linewidths=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Perfect Prediction')

# Add regression line
z = np.polyfit(y, y_pred_all, 1)
p = np.poly1d(z)
plt.plot(np.sort(y), p(np.sort(y)), 'b-', linewidth=1.5, label=f'Best Fit: y={z[0]:.3f}x+{z[1]:.3f}')

# Add RMSE and R2 to plot
rmse_full = np.sqrt(mean_squared_error(y, y_pred_all))
r2_full = r2_score(y, y_pred_all)
plt.annotate(f'RMSE = {rmse_full:.5f}\nR² = {r2_full:.3f}', 
             xy=(0.05, 0.95), xycoords='axes fraction',
             bbox=dict(boxstyle='round', fc='white', alpha=0.8),
             fontsize=12)

plt.xlabel("True Curvature (mm$^{-1}$)", fontsize=14)
plt.ylabel("Predicted Curvature (mm$^{-1}$)", fontsize=14)
plt.title("Gaussian Process Regression - Curvature Prediction", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "curvature_prediction_plot.png"), dpi=300)
plt.close()

# === Plot residuals ===
residuals = y_pred_all - y
plt.figure(figsize=(10, 6))
plt.scatter(y, residuals, alpha=0.4, edgecolors='k', linewidths=0.5)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel("True Curvature (mm$^{-1}$)", fontsize=14)
plt.ylabel("Residuals (Predicted - True)", fontsize=14)
plt.title("GPR Model Residuals", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "residuals_plot.png"), dpi=300)
plt.close()

# === Export prediction data to CSV ===
# This exports a comprehensive dataset with true vs predicted values and error metrics
# Useful for: scatter plots, error analysis by run ID, and general performance assessment
def export_prediction_data():
    """
    Exports prediction data with actual vs predicted values and errors for curvature.
    """
    # Calculate curvature errors
    curv_abs_error = np.abs(y - y_pred_all)
    curv_rel_error = np.where(y != 0, curv_abs_error / np.abs(y), 0)
    
    # Create comprehensive dataframe with all prediction data
    predictions_df = pd.DataFrame({
        'RunID': df['RunID'].values,
        'Curvature_True': y.values,
        'Curvature_Predicted': y_pred_all,
        'Curvature_AbsError': curv_abs_error,
        'Curvature_RelError': curv_rel_error,
        'Curvature_Active': df['Curvature_Active'].values
    })
    
    # Export to CSV
    csv_path = os.path.join(output_dir, "gpr_prediction_data.csv")  # Add "gpr_" prefix
    predictions_df.to_csv(csv_path, index=False)
    print(f"✓ Prediction data exported to: {csv_path}")
    
    return predictions_df

# This exports predictions from each cross-validation fold to analyze model stability
# Useful for: comparing performance across folds and validating generalization ability
def export_cv_prediction_data(all_test_indices, all_predictions):
    """
    Exports cross-validation prediction data.
    """
    # Get all CV predictions matched to their original indices
    cv_indices = np.array(all_test_indices)
    cv_preds = np.array(all_predictions)
    
    # Sort by original index
    sort_idx = np.argsort(cv_indices)
    cv_indices = cv_indices[sort_idx]
    cv_preds = cv_preds[sort_idx]
    
    # Create dataframe
    cv_df = pd.DataFrame({
        'RunID': df.iloc[cv_indices]['RunID'].values,
        'Curvature_True': y.iloc[cv_indices].values,
        'Curvature_Predicted': cv_preds,
        'Curvature_AbsError': np.abs(y.iloc[cv_indices].values - cv_preds),
        'Fold': np.zeros_like(cv_indices)  # Will be filled with fold numbers
    })
    
    # Identify which fold each sample belongs to
    for fold, (_, test_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
        mask = np.isin(cv_indices, test_idx)
        cv_df.loc[mask, 'Fold'] = fold + 1
    
    # Export to CSV
    csv_path = os.path.join(output_dir, "gpr_cv_predictions.csv")  # Add "gpr_" prefix
    cv_df.to_csv(csv_path, index=False)
    print(f"✓ Cross-validation predictions exported to: {csv_path}")
    
    return cv_df

# This exports detailed error distribution information for statistical analysis
# Useful for: creating error histograms and analyzing error patterns by run ID
def export_error_distribution():
    """
    Exports data for error distribution analysis.
    """
    # Calculate errors
    curvature_errors = y - y_pred_all
    
    # Create dataframe with error distributions
    error_df = pd.DataFrame({
        'Curvature_Error': curvature_errors,
        'Curvature_AbsError': np.abs(curvature_errors),
        'Curvature_True': y.values,
        'RunID': df['RunID'].values
    })
    
    # Export to CSV
    csv_path = os.path.join(output_dir, "gpr_error_distribution.csv")  # Add "gpr_" prefix
    error_df.to_csv(csv_path, index=False)
    print(f"✓ Error distribution data exported to: {csv_path}")
    
    return error_df

# This exports a summary of all key performance metrics in a structured format
# Useful for: comparing this model with other models using standard metrics
def export_summary_metrics():
    """
    Exports all performance metrics in a structured format.
    """
    # Create dataframe with all metrics
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'R²', 'MAE', 'Max Error'],
        'Curvature': [rmse_full, r2_full, 
                      mean_absolute_error(y, y_pred_all), 
                      max_error(y, y_pred_all)]
    })
    
    # Export to CSV
    csv_path = os.path.join(output_dir, "gpr_performance_metrics.csv")  # Add "gpr_" prefix
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Performance metrics exported to: {csv_path}")
    
    return metrics_df

# This exports the model configuration and hyperparameters used
# Useful for: documenting model settings and reproducing the model setup
def export_model_parameters():
    """
    Exports model hyperparameters and configuration.
    """
    # Gather all parameters and metadata
    model_params = {
        'kernel': str(gpr.kernel_),
        'normalize_y': gpr.normalize_y,
        'feature_count': len(feature_cols),
        'total_samples': len(X),
        'feature_list': feature_cols,
        'timestamp': timestamp,
        'model_type': 'GaussianProcessRegressor'
    }
    
    # Convert to dataframe
    params_df = pd.DataFrame([model_params])
    
    # Export to CSV
    csv_path = os.path.join(output_dir, "gpr_model_parameters.csv")  # Add "gpr_" prefix
    params_df.to_csv(csv_path, index=False)
    print(f"✓ Model parameters exported to: {csv_path}")
    
    return params_df

# This exports the raw feature data with predictions for custom analyses
# Useful for: feature importance analysis and advanced visualization
def export_raw_feature_data():
    """
    Exports the raw feature data with predictions to enable custom analyses.
    """
    # Get original feature values (not scaled)
    features_df = X.copy()
    features_df['RunID'] = df['RunID'].values
    features_df['Curvature_True'] = y.values
    features_df['Curvature_Predicted'] = y_pred_all
    features_df['Curvature_Error'] = y.values - y_pred_all
    
    # Export to CSV
    csv_path = os.path.join(output_dir, "gpr_features_data.csv")  # Add "gpr_" prefix
    features_df.to_csv(csv_path, index=False)
    print(f"✓ Raw feature data exported to: {csv_path}")
    
    return features_df

# Export all data to CSV files with explanatory comments about model configuration
# Model uses only FFT-based features as inputs
# Only data points with Curvature_Active = 1 were used for training
# Features were scaled using MinMaxScaler
# GroupKFold cross-validation was used, grouped by RunID
print("\nExporting data for external visualization and analysis:")
export_prediction_data()
export_cv_prediction_data(all_test_indices, all_predictions)
export_error_distribution()
export_summary_metrics()
export_model_parameters()
export_raw_feature_data()

print(f"\n✅ GPR training complete. Files saved to: {output_dir}")
print(f"Mean cross-validation RMSE: {np.mean(rmse_list):.5f}")
print(f"Mean cross-validation R²: {np.mean(r2_list):.3f}")
print(f"Full dataset RMSE: {rmse_full:.5f}, R²: {r2_full:.3f}")
print(f"Total execution time: {time.time() - start_time:.2f} seconds")
print(f"See README.md in that directory for details on available files.")
