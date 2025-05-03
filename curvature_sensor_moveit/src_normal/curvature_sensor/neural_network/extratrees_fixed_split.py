import os
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# === Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "csv_data", "preprocessed", "merged_tests")
output_dir = os.path.join(script_dir, "model_outputs", "extratrees_fixed_split")
os.makedirs(output_dir, exist_ok=True)

print(f"Looking for data in: {data_dir}")
if not os.path.exists(data_dir):
    print(f"ERROR: Directory not found. Please check the path.")
    exit(1)

# === Load and Create Train/Val Split ===
def load_and_create_split():
    # Define which test files to use for training and validation
    train_files = ['test_01.csv', 'test_02.csv']  # Using test_01 and test_02 for training
    val_file = 'test_03.csv'  # Using test_03 for validation
    
    train_dfs = []
    
    print("Loading datasets:")
    # Load training files
    for f in train_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["Source"] = f  # tag source file
            
            # Set Position_cm = -1 for Curvature_Active == 0
            df.loc[df['Curvature_Active'] == 0, 'Position_cm'] = -1.0
            
            train_dfs.append(df)
            print(f"- Training: {f} with {len(df)} rows")
        else:
            print(f"Warning: Training file {f} not found!")
    
    # Load validation file
    val_path = os.path.join(data_dir, val_file)
    if os.path.exists(val_path):
        val_df = pd.read_csv(val_path)
        val_df["Source"] = val_file
        val_df.loc[val_df['Curvature_Active'] == 0, 'Position_cm'] = -1.0
        print(f"- Validation: {val_file} with {len(val_df)} rows")
    else:
        print(f"ERROR: Validation file {val_file} not found!")
        return None, None
    
    # Combine training data
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # Print dataset overview
    print("\nDataset overview:")
    print(f"Training sources: {train_df['Source'].unique()}")
    print(f"Validation source: {val_df['Source'].unique()}")
    
    # Drop any rows where Curvature is NaN for both datasets
    train_df = train_df.dropna(subset=['Curvature'])
    val_df = val_df.dropna(subset=['Curvature'])
    
    # Filter out Curvature_Active == 2 rows completely from both sets
    print(f"Total training rows before filtering: {len(train_df)}")
    train_df = train_df[train_df['Curvature_Active'] != 2].copy()
    print(f"Total training rows after filtering: {len(train_df)}")
    
    print(f"Total validation rows before filtering: {len(val_df)}")
    val_df = val_df[val_df['Curvature_Active'] != 2].copy()
    print(f"Total validation rows after filtering: {len(val_df)}")
    
    return train_df, val_df

# === Custom Scorers for MultiOutput ===
def curvature_r2_scorer(y_true, y_pred):
    return r2_score(y_true[:, 0], y_pred[:, 0])

def position_r2_scorer(y_true, y_pred):
    position_mask = y_true[:, 1] != -1
    if sum(position_mask) > 0:
        return r2_score(y_true[position_mask, 1], y_pred[position_mask, 1])
    return 0

def combined_r2_scorer(y_true, y_pred):
    curvature_r2 = r2_score(y_true[:, 0], y_pred[:, 0])
    
    position_mask = y_true[:, 1] != -1
    if sum(position_mask) > 0:
        position_r2 = r2_score(y_true[position_mask, 1], y_pred[position_mask, 1])
    else:
        position_r2 = 0
    
    return (curvature_r2 + position_r2) / 2

# === Create Fixed Train/Val Split for BayesSearchCV ===
class FixedSplit:
    def __init__(self, X_train_indices, X_val_indices):
        self.X_train_indices = X_train_indices
        self.X_val_indices = X_val_indices
        
    def split(self, X, y=None, groups=None):
        yield self.X_train_indices, self.X_val_indices
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return 1

# === Main Function ===
def run_fixed_split_optimization():
    # Load and create train/val split
    train_df, val_df = load_and_create_split()
    if train_df is None or val_df is None:
        print("Error creating train/val split. Exiting.")
        return
    
    # Features and targets
    feature_cols = [
        'FFT_200Hz', 'FFT_400Hz', 'FFT_600Hz', 'FFT_800Hz', 'FFT_1000Hz',
        'FFT_1200Hz', 'FFT_1400Hz', 'FFT_1600Hz', 'FFT_1800Hz', 'FFT_2000Hz',
        'Low_Band_Mean', 'Mid_Band_Mean', 'High_Band_Mean',
        'Mid_to_Low_Band_Ratio', 'High_to_Mid_Band_Ratio',
        'FFT_Peak_Index', 'PC1', 'PC2'
    ]
    
    # Print shapes
    print(f"\nTraining set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    
    # Combine for feature extraction then separate for training
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    X_combined = combined_df[feature_cols].values
    y_combined = combined_df[['Curvature', 'Position_cm']].values
    
    # Create indices for train and validation
    train_indices = list(range(len(train_df)))
    val_indices = list(range(len(train_df), len(train_df) + len(val_df)))
    
    # Verify no overlap
    print(f"\nX_train indices: {len(train_indices)}")
    print(f"X_val indices: {len(val_indices)}")
    assert not set(train_indices).intersection(set(val_indices)), "Error: Train and validation indices overlap!"
    
    # Create fixed split
    fixed_split = FixedSplit(train_indices, val_indices)
    
    # Define the search space
    search_space = {
        'estimator__n_estimators': Integer(100, 300),
        'estimator__max_depth': Integer(10, 50),
        'estimator__max_features': Categorical(['sqrt', 'log2', None]),
        'estimator__min_samples_split': Integer(2, 10)
    }
    
    # Define the base estimator
    base_estimator = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    model = MultiOutputRegressor(base_estimator)
    
    # Define scoring
    scoring = {
        'curvature_r2': make_scorer(curvature_r2_scorer),
        'position_r2': make_scorer(position_r2_scorer),
        'combined_r2': make_scorer(combined_r2_scorer)
    }
    
    # Setup Bayesian optimization with fixed split
    opt = BayesSearchCV(
        model,
        search_space,
        n_iter=25,
        cv=fixed_split,
        scoring=scoring,
        refit='combined_r2',
        return_train_score=True,
        random_state=42,
        n_jobs=-1
    )
    
    # Run the optimization
    print("\nStarting Bayesian Optimization with fixed split...")
    opt.fit(X_combined, y_combined)
    
    # Get results
    print(f"\nBest parameters found: {opt.best_params_}")
    print(f"Best score: {opt.best_score_}")
    
    # Save best model
    joblib.dump(opt.best_estimator_, os.path.join(output_dir, "best_model.pkl"))
    
    # Extract and save detailed results
    results = pd.DataFrame(opt.cv_results_)
    results.to_csv(os.path.join(output_dir, "optimization_results.csv"), index=False)
    
    # Get the best model and evaluate on the validation set
    best_model = opt.best_estimator_
    
    # Separate the data for final evaluation
    X_train = X_combined[train_indices]
    y_train = y_combined[train_indices]
    X_val = X_combined[val_indices]
    y_val = y_combined[val_indices]
    
    # Make predictions on validation set
    y_pred = best_model.predict(X_val)
    
    # Calculate metrics
    curvature_rmse = np.sqrt(mean_squared_error(y_val[:, 0], y_pred[:, 0]))
    curvature_r2 = r2_score(y_val[:, 0], y_pred[:, 0])
    
    # Only calculate position metrics for rows where position is active
    position_mask = y_val[:, 1] != -1
    if sum(position_mask) > 0:
        position_rmse = np.sqrt(mean_squared_error(
            y_val[position_mask, 1], y_pred[position_mask, 1]))
        position_r2 = r2_score(
            y_val[position_mask, 1], y_pred[position_mask, 1])
    else:
        position_rmse = float('nan')
        position_r2 = float('nan')
    
    # Print final evaluation metrics
    print("\nFinal Evaluation on Validation Set (test_03):")
    print(f"  Curvature - RMSE: {curvature_rmse:.6f}, R²: {curvature_r2:.6f}")
    if not np.isnan(position_rmse):
        print(f"  Position  - RMSE: {position_rmse:.6f}, R²: {position_r2:.6f}")
    else:
        print("  Position  - No valid position data in validation set")
    
    # Save predictions
    val_df_copy = val_df.copy()
    val_df_copy['Pred_Curvature'] = y_pred[:, 0]
    val_df_copy['Pred_Position_cm'] = y_pred[:, 1]
    val_df_copy.to_csv(os.path.join(output_dir, "validation_predictions.csv"), index=False)
    
    # Save final report
    final_report = {
        "Training_Files": ['test_01.csv', 'test_02.csv'],
        "Validation_File": 'test_03.csv',
        "Best_Parameters": opt.best_params_,
        "Best_Score": float(opt.best_score_),
        "Validation_Metrics": {
            "Curvature_RMSE": float(curvature_rmse),
            "Curvature_R2": float(curvature_r2),
            "Position_RMSE": float(position_rmse) if not np.isnan(position_rmse) else None,
            "Position_R2": float(position_r2) if not np.isnan(position_r2) else None
        }
    }
    
    # Save the report
    with open(os.path.join(output_dir, "fixed_split_results.json"), 'w') as f:
        json.dump(final_report, f, indent=4)
    
    # Plot optimization history
    plot_optimization_history(opt, output_dir)
    
    print(f"\nOptimization complete. Results saved to {output_dir}")

def plot_optimization_history(optimizer, output_dir):
    """Plot the optimization history."""
    results = pd.DataFrame(optimizer.cv_results_)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(results['rank_test_combined_r2'])
    plt.xlabel('Iterations')
    plt.ylabel('Rank of Combined R²')
    plt.title('Optimization Progress')
    
    plt.subplot(1, 2, 2)
    plt.plot(results['mean_test_curvature_r2'], label='Curvature R²')
    plt.plot(results['mean_test_position_r2'], label='Position R²')
    plt.plot(results['mean_test_combined_r2'], label='Combined R²')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.title('Score Progression')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_history.png"))

if __name__ == "__main__":
    run_fixed_split_optimization()