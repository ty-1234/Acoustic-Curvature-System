"""
ExtraTrees Regressor with Fixed Hyperparameters for Curvature Sensor

This script trains a multi-output ExtraTrees regressor to predict both position and curvature 
from FFT sensor data. It uses pre-optimized hyperparameters and outputs comprehensive 
evaluation metrics to CSV files for external visualization.

Author: Bipindra Rai
Date: 26 April 2025
"""

import os
import sys
import json
import time
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
from sklearn.ensemble import ExtraTreesRegressor

# Load dataset
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(base_dir, "csv_data", "preprocessed", "preprocessed_training_dataset.csv")
df = pd.read_csv(dataset_path)

# Drop rows where Position_cm is missing while Curvature_Active == 1
df = df[~(df["Position_cm"].isna() & (df["Curvature_Active"] == 1))]
df["Position_cm"] = df["Position_cm"].fillna(-1)

# FFT peak index feature - only include true frequency columns
fft_cols = [col for col in df.columns if col.startswith("FFT_") and "Hz" in col]
df["FFT_Peak_Index"] = (
    df[fft_cols]
    .idxmax(axis=1)
    .str.extract(r"(\d+)")
    .astype(float)
    .fillna(-1)
    .astype(int)
)

# Additional FFT-based features - extraction of frequency values
freqs = np.array([int(col.split('_')[1].split('Hz')[0]) for col in fft_cols])
df["FFT_Centroid"] = df[fft_cols].dot(freqs) / df[fft_cols].sum(axis=1)

# Frequency band calculations
low_freq_cols = [col for col in fft_cols if 200 <= int(col.split('_')[1].split('Hz')[0]) <= 800]
mid_freq_cols = [col for col in fft_cols if 800 < int(col.split('_')[1].split('Hz')[0]) <= 1400]
high_freq_cols = [col for col in fft_cols if int(col.split('_')[1].split('Hz')[0]) > 1400]

df["low_band_power"] = df[low_freq_cols].mean(axis=1)
df["mid_band_power"] = df[mid_freq_cols].mean(axis=1)
df["high_band_power"] = df[high_freq_cols].mean(axis=1)

# Feature and target selection
feature_cols = [col for col in df.columns if col.startswith("FFT_") or "Band" in col or "Ratio" in col or col in ["Curvature_Active", "FFT_Peak_Index", "FFT_Centroid", "low_band_power", "mid_band_power", "high_band_power"]]
X = df[feature_cols]
y = df[["Position_cm", "Curvature_Label"]]

# Updated sample weight with 0.01 for inactive points
sample_weight = np.where(y["Position_cm"] == -1, 0.01, 1.0)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Grouped cross-validation
groups = df["RunID"]
gkf = GroupKFold(n_splits=2)
train_idx, test_idx = next(gkf.split(X_scaled, y, groups))
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Pre-optimized hyperparameters (no Optuna optimization needed)
print("Using pre-optimized hyperparameters...")
best_params = {
    "n_estimators": 443,
    "max_depth": 15,
    "min_samples_split": 7,
    "min_samples_leaf": 5,
    "max_features": None,
    "bootstrap": True
}
print("Best hyperparameters:", best_params)

# Create a class to simulate the study object for compatibility with existing code
class MockStudy:
    def __init__(self, best_params):
        self.best_params = best_params
        
        # Create a mock trial to maintain compatibility with existing code
        class MockTrial:
            def __init__(self, params):
                self.number = 0
                self.value = 0  # Placeholder value
                self.params = params
                
        self.trials = [MockTrial(best_params)]

# Create mock study for compatibility with export functions
study = MockStudy(best_params)

# Train final model with the best parameters
base_model = ExtraTreesRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    bootstrap=best_params['bootstrap'],
    random_state=42
)
best_model = MultiOutputRegressor(base_model)
best_model.fit(X_train, y_train, sample_weight=sample_weight[train_idx])

# Evaluate best model on test fold
y_pred = best_model.predict(X_test)
rmse_pos = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]) ** 0.5
rmse_curv = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]) ** 0.5
r2_pos = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
r2_curv = r2_score(y_test.iloc[:, 1], y_pred[:, 1])

# Calculate additional metrics
mae_pos = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
mae_curv = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
maxerr_pos = max_error(y_test.iloc[:, 0], y_pred[:, 0])
maxerr_curv = max_error(y_test.iloc[:, 1], y_pred[:, 1])

# Use timestamped output directory in the correct location
timestamp = datetime.now().strftime("%-I:%M%p").lower()
model_output_dir = os.path.join(base_dir, "neural_network", "model_outputs", "extratrees", timestamp)
os.makedirs(model_output_dir, exist_ok=True)

# Save metrics to JSON
metrics = {
    "model": "ExtraTrees_Optuna",
    "rmse_position": rmse_pos,
    "rmse_curvature": rmse_curv,
    "r2_position": r2_pos,
    "r2_curvature": r2_curv,
    "mae_position": mae_pos,
    "mae_curvature": mae_curv,
    "maxerr_position": maxerr_pos,
    "maxerr_curvature": maxerr_curv,
    "best_params": best_params,
    "optimization_history": [
        {"trial": t.number, "value": t.value, "params": t.params}
        for t in study.trials
    ]
}

with open(os.path.join(model_output_dir, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# Save model
joblib.dump(best_model, os.path.join(model_output_dir, "extratrees_optuna_model.pkl"))
# Also save the scaler for future use
joblib.dump(scaler, os.path.join(model_output_dir, "feature_scaler.pkl"))

# Export data to CSV files for external visualization
def export_prediction_data():
    """
    Exports prediction data with actual vs predicted values and errors
    for both position and curvature targets.
    """
    # Calculate position errors
    pos_abs_error = np.abs(y_test.iloc[:, 0] - y_pred[:, 0])
    pos_rel_error = np.where(y_test.iloc[:, 0] != 0, 
                            pos_abs_error / np.abs(y_test.iloc[:, 0]), 
                            0)
    
    # Calculate curvature errors
    curv_abs_error = np.abs(y_test.iloc[:, 1] - y_pred[:, 1])
    curv_rel_error = np.where(y_test.iloc[:, 1] != 0, 
                             curv_abs_error / np.abs(y_test.iloc[:, 1]), 
                             0)
    
    # Create comprehensive dataframe with all prediction data
    predictions_df = pd.DataFrame({
        'RunID': df.iloc[test_idx]['RunID'].values,
        'Position_True': y_test.iloc[:, 0].values,
        'Position_Predicted': y_pred[:, 0],
        'Position_AbsError': pos_abs_error,
        'Position_RelError': pos_rel_error,
        'Curvature_True': y_test.iloc[:, 1].values,
        'Curvature_Predicted': y_pred[:, 1],
        'Curvature_AbsError': curv_abs_error,
        'Curvature_RelError': curv_rel_error,
        'Curvature_Active': df.iloc[test_idx]['Curvature_Active'].values
    })
    
    # Export to CSV
    csv_path = os.path.join(model_output_dir, "prediction_data.csv")
    predictions_df.to_csv(csv_path, index=False)
    print(f"✓ Prediction data exported to: {csv_path}")
    
    # Return for potential further use
    return predictions_df

def export_feature_importance():
    """
    Exports feature importance data from the trained model.
    """
    # Extract feature importances from each base estimator
    importances = []
    for estimator in best_model.estimators_:
        importances.append(estimator.feature_importances_)
    
    # Average importance across all estimators
    mean_importance = np.mean(importances, axis=0)
    
    # Create dataframe with feature names and importance values
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': mean_importance,
    }).sort_values('Importance', ascending=False)
    
    # Export to CSV
    csv_path = os.path.join(model_output_dir, "feature_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"✓ Feature importance data exported to: {csv_path}")
    
    return importance_df

def export_optimization_history():
    """
    Exports a placeholder optimization history (only contains the fixed best params).
    """
    # Create dataframe with optimization history (mock data with only one entry)
    trials_data = []
    
    for trial in study.trials:
        trial_data = {
            'trial_number': trial.number,
            'value': trial.value
        }
        
        # Add all parameters
        for param_name, param_value in trial.params.items():
            trial_data[param_name] = param_value
            
        trials_data.append(trial_data)
    
    # Convert to dataframe
    history_df = pd.DataFrame(trials_data)
    
    # Export to CSV
    csv_path = os.path.join(model_output_dir, "optimization_history.csv")
    history_df.to_csv(csv_path, index=False)
    print(f"✓ Optimization history exported to: {csv_path}")
    
    return history_df

def export_summary_metrics():
    """
    Exports all performance metrics in a structured format.
    """
    # Create dataframe with all metrics
    metrics_df = pd.DataFrame({
        'Metric': ['RMSE', 'R²', 'MAE', 'Max Error'],
        'Position': [rmse_pos, r2_pos, mae_pos, maxerr_pos],
        'Curvature': [rmse_curv, r2_curv, mae_curv, maxerr_curv]
    })
    
    # Export to CSV
    csv_path = os.path.join(model_output_dir, "performance_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ Performance metrics exported to: {csv_path}")
    
    return metrics_df

def export_error_distribution():
    """
    Exports data for error distribution analysis.
    """
    # Calculate errors
    position_errors = y_test.iloc[:, 0] - y_pred[:, 0]
    curvature_errors = y_test.iloc[:, 1] - y_pred[:, 1]
    
    # Create dataframe with error distributions
    error_df = pd.DataFrame({
        'Position_Error': position_errors,
        'Curvature_Error': curvature_errors,
        'Position_AbsError': np.abs(position_errors),
        'Curvature_AbsError': np.abs(curvature_errors),
        'Curvature_Active': df.iloc[test_idx]['Curvature_Active'].values
    })
    
    # Export to CSV
    csv_path = os.path.join(model_output_dir, "error_distribution.csv")
    error_df.to_csv(csv_path, index=False)
    print(f"✓ Error distribution data exported to: {csv_path}")
    
    return error_df

def export_model_parameters():
    """
    Exports model hyperparameters and configuration.
    """
    # Gather all parameters and metadata
    model_params = {
        'best_hyperparameters': best_params,
        'feature_count': len(feature_cols),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_list': feature_cols,
        'timestamp': timestamp,
        'model_type': 'ExtraTreesRegressor'
    }
    
    # Convert to dataframe
    params_df = pd.DataFrame([model_params])
    
    # Export to CSV
    csv_path = os.path.join(model_output_dir, "model_parameters.csv")
    params_df.to_csv(csv_path, index=False)
    print(f"✓ Model parameters exported to: {csv_path}")
    
    return params_df

def export_raw_feature_data():
    """
    Exports the raw feature data for test samples to enable custom analyses.
    """
    # Get original feature values (not scaled)
    test_features_df = X.iloc[test_idx].copy()
    test_features_df['RunID'] = df.iloc[test_idx]['RunID'].values
    
    # Add targets and predictions
    for col_idx, col_name in enumerate(['Position_cm', 'Curvature_Label']):
        test_features_df[f'{col_name}_True'] = y_test.iloc[:, col_idx].values
        test_features_df[f'{col_name}_Predicted'] = y_pred[:, col_idx]
    
    # Export to CSV
    csv_path = os.path.join(model_output_dir, "test_features_data.csv")
    test_features_df.to_csv(csv_path, index=False)
    print(f"✓ Raw feature data exported to: {csv_path}")
    
    return test_features_df

# Export all data to CSV files
print("\nExporting data for external visualization and analysis:")
export_prediction_data()
export_feature_importance()
export_optimization_history()
export_summary_metrics()
export_error_distribution()
export_model_parameters()
export_raw_feature_data()

# Create README file with overview of available data
readme_content = f"""# ExtraTrees Model with Fixed Hyperparameters ({timestamp})

## Model Performance Summary
- Position RMSE: {rmse_pos:.2f} cm, R²: {r2_pos:.3f}
- Curvature RMSE: {rmse_curv:.5f}, R²: {r2_curv:.3f}

## Available CSV Files for Visualization:

1. **prediction_data.csv**: Contains true and predicted values for both targets, along with absolute and relative errors.
   - Use for scatter plots comparing true vs. predicted values
   - Use for error analysis by run ID or curvature state

2. **feature_importance.csv**: Feature importance values from the trained model.
   - Use for bar charts showing most influential features

3. **optimization_history.csv**: Contains the fixed hyperparameters used (no optimization performed).

4. **performance_metrics.csv**: Summary metrics (RMSE, R², MAE, Max Error) for both targets.
   - Use for comparison bar charts between metrics and targets

5. **error_distribution.csv**: Detailed error distribution data.
   - Use for histograms showing error distributions
   - Use for analyzing errors by curvature state

6. **model_parameters.csv**: Model configuration and fixed hyperparameters.
   - Reference for model settings

7. **test_features_data.csv**: Raw feature values for test samples with predictions.
   - Use for custom feature analysis and visualization

## Fixed Hyperparameters
```
{json.dumps(best_params, indent=2)}
```

Author: Bipindra Rai
Date: 26 April 2025
"""

# Write README file
with open(os.path.join(model_output_dir, "README.md"), "w") as f:
    f.write(readme_content)

# Print summary
print("\n✅ Training complete with fixed hyperparameters")
print(f"Best Parameters: {best_params}")
print(f"Curvature RMSE: {rmse_curv:.5f}, R²: {r2_curv:.3f}")
print(f"Position RMSE: {rmse_pos:.2f} cm, R²: {r2_pos:.3f}")
print(f"\nAll data exported to CSVs in: {model_output_dir}")
print(f"See README.md in that directory for details on available files.")