"""
Multi-output Regression Model Training for Curvature Sensor.

This script trains and evaluates multiple regression models for a curvature sensor,
predicting both position and curvature from FFT and band-related features.
It handles preprocessing, cross-validation, hyperparameter tuning, and generates
evaluation metrics and visualization plots.

Models evaluated:
- Extra Trees Regressor
- XGBoost Regressor
- LightGBM Regressor
- Random Forest Regressor

Author: Bipindra Rai
Date: April 26, 2025
"""

import os
import sys
import json
import time
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# --- New Imports ---
from sklearn.metrics import mean_absolute_error, max_error

# === Setup paths ===
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(parent_dir, "csv_data", "preprocessed", "preprocessed_training_dataset.csv")
output_dir = os.path.join(parent_dir, "neural_network", "model_outputs", "multioutput_model_results")
os.makedirs(output_dir, exist_ok=True)

def load_and_preprocess_data():
    """
    Load the dataset and perform preprocessing operations.
    
    Returns:
        tuple: (DataFrame, features matrix, target matrix, sample_weights, groups)
            - DataFrame containing all data
            - X: Feature matrix
            - y: Target variables (Position_cm, Curvature_Label)
            - sample_weight: Weights for training
            - groups: RunID values for cross-validation
    """
    df = pd.read_csv(dataset_path)
    df = df[~(df["Position_cm"].isna() & (df["Curvature_Active"] == 1))]
    df["Position_cm"] = df["Position_cm"].fillna(-1)
    df["FFT_Peak_Index"] = (
        df[[col for col in df.columns if col.startswith("FFT_")]]
        .idxmax(axis=1)
        .str.extract(r"(\d+)")
        .astype(float)
        .fillna(-1)
        .astype(int)
    )
    
    print("\n📋 Unique RunIDs detected:", df["RunID"].unique())
    print(f"Total unique RunIDs: {df['RunID'].nunique()}")
    
    feature_cols = [col for col in df.columns if col.startswith("FFT_") or "Band" in col or "Ratio" in col or col == "Curvature_Active" or col == "FFT_Peak_Index"]
    X = df[feature_cols]
    y = df[["Position_cm", "Curvature_Label"]]
    sample_weight = np.where(y["Position_cm"] == -1, 0.01, 1.0)
    groups = df["RunID"]
    
    return df, X, y, sample_weight, groups


def create_model(model_name):
    """
    Create model and parameter grid for hyperparameter tuning.
    
    Args:
        model_name (str): Name of the model to create ('extratrees', 'xgb', 'lgbm', 'rf')
        
    Returns:
        tuple: (base_model, param_grid)
            - base_model: Initialized model instance
            - param_grid: Dictionary of parameters for RandomizedSearchCV
    """
    if model_name == "extratrees":
        base_model = ExtraTreesRegressor(n_estimators=100, random_state=42)
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [10, 20, 30, 50, None],
            "estimator__min_samples_split": [2, 4, 6, 8],
            "estimator__min_samples_leaf": [1, 2, 3],
            "estimator__max_features": ["sqrt", "log2", None],
            "estimator__bootstrap": [True, False]
        }
    elif model_name == "xgb":
        base_model = XGBRegressor(n_estimators=100, random_state=42)
        param_grid = {
            "estimator__max_depth": [3, 5, 7],
            "estimator__learning_rate": [0.05, 0.1, 0.2],
            "estimator__n_estimators": [100, 200, 300]
        }
    elif model_name == "lgbm":
        base_model = LGBMRegressor(random_state=42)
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__learning_rate": [0.05, 0.1, 0.2],
            "estimator__max_depth": [3, 5, 7, -1],
            "estimator__num_leaves": [31, 50, 100],
            "estimator__boosting_type": ["gbdt", "dart"]
        }
    elif model_name == "rf":
        base_model = RandomForestRegressor(n_estimators=100, random_state=42)
        param_grid = {
            "estimator__n_estimators": [100, 200, 300],
            "estimator__max_depth": [10, 20, 30, 50, None],
            "estimator__min_samples_split": [2, 4, 6, 8],
            "estimator__min_samples_leaf": [1, 2, 3],
            "estimator__max_features": ["sqrt", "log2", None],
            "estimator__bootstrap": [True, False]
        }
    return base_model, param_grid


def evaluate_model(model_name, X_scaled, y, groups, sample_weight):
    """
    Evaluate a model using hyperparameter tuning and cross-validation.
    
    Args:
        model_name (str): Name of the model to evaluate
        X_scaled (array): Scaled feature matrix
        y (DataFrame): Target variables
        groups (Series): GroupKFold groups
        sample_weight (array): Sample weights for training
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print(f"\n🚀 Training final model: {model_name}")
    base_model, param_grid = create_model(model_name)
    model = MultiOutputRegressor(base_model)
    
    print(f"\n🔧 Performing RandomizedSearchCV for {model_name}...")
    search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, random_state=42, n_jobs=-1, cv=3)
    search.fit(X_scaled, y)
    best_model = search.best_estimator_
    
    gkf = GroupKFold(n_splits=2)
    rmse_pos_list, rmse_curv_list, r2_pos_list, r2_curv_list = [], [], [], []
    mae_pos_list, mae_curv_list, maxerr_pos_list, maxerr_curv_list = [], [], [], []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_scaled, y, groups=groups)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        best_model.fit(X_train, y_train, sample_weight=sample_weight[train_idx])
        y_pred = best_model.predict(X_test)
        
        rmse_pos = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]) ** 0.5
        rmse_curv = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]) ** 0.5
        r2_pos = r2_score(y_test.iloc[:, 0], y_pred[:, 0])
        r2_curv = r2_score(y_test.iloc[:, 1], y_pred[:, 1])
        mae_pos = mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0])
        mae_curv = mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1])
        maxerr_pos = max_error(y_test.iloc[:, 0], y_pred[:, 0])
        maxerr_curv = max_error(y_test.iloc[:, 1], y_pred[:, 1])
        
        rmse_pos_list.append(rmse_pos)
        rmse_curv_list.append(rmse_curv)
        r2_pos_list.append(r2_pos)
        r2_curv_list.append(r2_curv)
        mae_pos_list.append(mae_pos)
        mae_curv_list.append(mae_curv)
        maxerr_pos_list.append(maxerr_pos)
        maxerr_curv_list.append(maxerr_curv)
    
    metrics = {
        "model": model_name,
        "rmse_position_mean": np.mean(rmse_pos_list),
        "rmse_curvature_mean": np.mean(rmse_curv_list),
        "r2_position_mean": np.mean(r2_pos_list),
        "r2_curvature_mean": np.mean(r2_curv_list),
        "mae_position_mean": np.mean(mae_pos_list),
        "mae_curvature_mean": np.mean(mae_curv_list),
        "maxerr_position_mean": np.mean(maxerr_pos_list),
        "maxerr_curvature_mean": np.mean(maxerr_curv_list)
    }
    
    print(f"\n📌 Final Evaluation Metrics for {model_name}:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    return metrics


def plot_metric_bar(metric_values, title, ylabel, filename, colors=None):
    """
    Create a bar plot for model comparison on a specific metric.
    
    Args:
        metric_values (Series): Values of the metric for each model
        title (str): Plot title
        ylabel (str): Y-axis label
        filename (str): Output filename
        colors (list, optional): Bar colors. Defaults to None.
    """
    plt.figure(figsize=(8,6))
    bars = plt.bar(metrics_df["model"], metric_values, color=colors if colors else 'skyblue')
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    plt.title(title, fontsize=14)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def main():
    """
    Main function to execute the model training and evaluation pipeline.
    """
    # Load and preprocess data
    df, X, y, sample_weight, groups = load_and_preprocess_data()
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models to evaluate
    models_to_run = ["extratrees", "xgb", "lgbm", "rf"]
    all_metrics = []
    
    # Evaluate all models
    for model_name in models_to_run:
        metrics = evaluate_model(model_name, X_scaled, y, groups, sample_weight)
        all_metrics.append(metrics)
    
    # Create summary dataframe and save results
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.sort_values("rmse_curvature_mean", inplace=True)
    metrics_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Generate visualization plots
    plot_metric_bar(metrics_df["rmse_curvature_mean"], 
                    "Model Comparison – RMSE Curvature", 
                    "RMSE (Curvature, mm⁻¹)", 
                    "rmse_curvature.png")
    
    plot_metric_bar(metrics_df["rmse_position_mean"], 
                    "Model Comparison – RMSE Position", 
                    "RMSE (Position, cm)", 
                    "rmse_position.png")
    
    plot_metric_bar(metrics_df["mae_curvature_mean"], 
                    "Model Comparison – MAE Curvature", 
                    "MAE (Curvature, mm⁻¹)", 
                    "mae_curvature.png")
    
    plot_metric_bar(metrics_df["mae_position_mean"], 
                    "Model Comparison – MAE Position", 
                    "MAE (Position, cm)", 
                    "mae_position.png")
    
    plot_metric_bar(metrics_df["maxerr_curvature_mean"], 
                    "Model Comparison – Max Error Curvature", 
                    "Max Error (Curvature, mm⁻¹)", 
                    "maxerr_curvature.png")
    
    plot_metric_bar(metrics_df["maxerr_position_mean"], 
                    "Model Comparison – Max Error Position", 
                    "Max Error (Position, cm)", 
                    "maxerr_position.png")
    
    print("\n✅ All models evaluated and high-quality graphs generated.")


if __name__ == "__main__":
    main()
