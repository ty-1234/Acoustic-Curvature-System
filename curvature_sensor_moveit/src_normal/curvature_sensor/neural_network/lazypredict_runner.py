"""
LazyPredict Runner Script

This script evaluates multiple machine learning models on curvature sensor data.
It performs both classification (section prediction) and regression (curvature value prediction) 
using the LazyPredict library, which automatically tests many models to find the best performers.

The script loads FFT features from a combined dataset CSV file, trains various models,
and saves the results ranked by performance.

Author: Bipindra Rai
Date: 2025-04-22

"""

import pandas as pd
import os
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lazypredict.Supervised import LazyClassifier, LazyRegressor
from tqdm import tqdm

def setup_paths():
    """
    Set up the correct file paths for data loading and result saving.
    
    Returns:
        tuple: Contains dataset path and results directory path
    """
    # Add parent directory to path for imports
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    # Set correct paths
    dataset_path = os.path.join(parent_dir, "csv_data", "combined_dataset.csv")
    results_dir = os.path.join(parent_dir, "csv_data")
    
    return dataset_path, results_dir

def run_classification(df, fft_cols):
    """
    Run classification models to predict the section from FFT features.
    
    Args:
        df (pd.DataFrame): The dataset containing features and section labels
        fft_cols (list): List of column names containing FFT features
    
    Returns:
        tuple: Contains (models_dataframe, execution_duration)
    """
    print("\n🎯 Running LazyClassifier (Section Classification)...")
    df_class = df.dropna(subset=["Section"])
    y_class = df_class["Section"].astype(str)
    X_class = df_class[fft_cols]

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_class, y_class, stratify=y_class, test_size=0.2, random_state=42
    )

    start = time.time()
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models_class, _ = clf.fit(Xc_train, Xc_test, yc_train, yc_test)
    duration_class = time.time() - start
    
    return models_class, duration_class

def run_regression(df, fft_cols):
    """
    Run regression models to predict curvature values from FFT features.
    
    Args:
        df (pd.DataFrame): The dataset containing features and curvature labels
        fft_cols (list): List of column names containing FFT features
    
    Returns:
        tuple: Contains (models_dataframe, execution_duration)
    """
    print("\n📐 Running LazyRegressor (Curvature Estimation)...")
    df_reg = df.dropna(subset=["Curvature_Label"])
    y_reg = df_reg["Curvature_Label"]
    X_reg = df_reg[fft_cols]

    scaler = MinMaxScaler()
    Xr_scaled = scaler.fit_transform(X_reg)

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        Xr_scaled, y_reg, test_size=0.2, random_state=42
    )

    start = time.time()
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models_reg, _ = reg.fit(Xr_train, Xr_test, yr_train, yr_test)
    duration_reg = time.time() - start
    
    return models_reg, duration_reg

def save_results(models_class, models_reg, results_dir):
    """
    Save classification and regression results to CSV files.
    
    Args:
        models_class (pd.DataFrame): Classification model results
        models_reg (pd.DataFrame): Regression model results
        results_dir (str): Directory path to save results
    
    Returns:
        tuple: Paths to the saved CSV files
    """
    os.makedirs(results_dir, exist_ok=True)
    class_results_path = os.path.join(results_dir, "section_classification_results.csv")
    reg_results_path = os.path.join(results_dir, "curvature_regression_results.csv")

    models_class.to_csv(class_results_path)
    models_reg.to_csv(reg_results_path)
    
    return class_results_path, reg_results_path

def main():
    """
    Main function to execute the LazyPredict evaluation workflow.
    
    Loads the dataset, extracts features, runs classification and regression models,
    and saves the results.
    """
    dataset_path, results_dir = setup_paths()
    print(f"🔍 Loading dataset from: {dataset_path}")

    # Load Dataset
    df = pd.read_csv(dataset_path)

    # Select FFT columns only
    fft_cols = [col for col in df.columns if col.startswith("FFT_")]
    
    # Run classification
    models_class, duration_class = run_classification(df, fft_cols)
    print(f"\n🏁 Finished LazyClassifier in {duration_class:.2f} seconds.")
    print("\n🏆 Top 5 Classifier Models (by Accuracy):")
    print(models_class.sort_values("Accuracy", ascending=False).head(5))

    # Run regression
    models_reg, duration_reg = run_regression(df, fft_cols)
    print(f"\n🏁 Finished LazyRegressor in {duration_reg:.2f} seconds.")
    print("\n🏆 Top 5 Regressor Models (by RMSE):")
    print(models_reg.sort_values("RMSE").head(5))

    # Save results
    class_path, reg_path = save_results(models_class, models_reg, results_dir)
    print("\n✅ Results saved to:")
    print(f"   - {class_path}")
    print(f"   - {reg_path}")

if __name__ == "__main__":
    main()
