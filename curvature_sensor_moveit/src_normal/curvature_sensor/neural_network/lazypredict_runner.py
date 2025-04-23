"""
LazyPredict Multi-Output Regression Script

This script evaluates multiple regression models on curvature sensor data.
It predicts both:
- The estimated position along the sensor (in cm)
- The curvature applied at that position

It wraps each model in MultiOutputRegressor and evaluates them using RMSE.
Only rows where curvature is actively applied (as indicated by Curvature_Active = 1) are used.
Features include raw FFT bands and engineered statistical + band ratio features.
Results are saved for further comparison.

Author: Bipindra Rai
Updated: 2025-04-23
"""

import pandas as pd
import os
import time
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from lazypredict.Supervised import REGRESSORS

def setup_paths():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    dataset_path = os.path.join(parent_dir, "csv_data", "preprocessed", "preprocessed_training_dataset.csv")
    results_dir = os.path.join(parent_dir, "csv_data", "lazy_csv")
    
    return dataset_path, results_dir

def prepare_data(df, fft_cols):
    # Use only rows where curvature is actively applied
    df = df[df["Curvature_Active"] == 1]  # Use only rows where curvature is actively applied

    # Include FFTs and engineered features including band ratios
    engineered_cols = [col for col in df.columns if col.startswith("FFT_") or "Band" in col or "Ratio" in col or "FFT_" in col]
    X = df[engineered_cols]
    y = df[["Position_cm", "Curvature_Label"]]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def run_multioutput_regressors(X_train, X_test, y_train, y_test):
    results = []

    print("\n📐 Running MultiOutput Regressors (Position + Curvature Estimation)...")
    for name, RegressorClass in REGRESSORS:
        try:
            model = MultiOutputRegressor(RegressorClass())
            start = time.time()
            model.fit(X_train, y_train)
            duration = time.time() - start

            y_pred = model.predict(X_test)

            # Manual RMSE calculation for maximum compatibility
            rmse_pos = mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]) ** 0.5
            rmse_curv = mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]) ** 0.5

            results.append({
                "Model": name,
                "RMSE_Position_cm": rmse_pos,
                "RMSE_Curvature": rmse_curv,
                "Time_Seconds": round(duration, 2)
            })

            print(f"✅ {name:30} | Pos RMSE: {rmse_pos:.3f} cm | Curv RMSE: {rmse_curv:.5f} | Time: {duration:.2f}s")

        except Exception as e:
            print(f"❌ {name:30} failed: {e}")

    return pd.DataFrame(results).sort_values(by="RMSE_Curvature")

def save_results(df, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "multioutput_regression_results.csv")
    df.to_csv(out_path, index=False)
    return out_path

def main():
    dataset_path, results_dir = setup_paths()
    print(f"🔍 Loading dataset from: {dataset_path}")

    df = pd.read_csv(dataset_path)
    fft_cols = [col for col in df.columns if col.startswith("FFT_")]

    X_train, X_test, y_train, y_test = prepare_data(df, fft_cols)
    results_df = run_multioutput_regressors(X_train, X_test, y_train, y_test)

    print("\n🏆 Top 5 Regressors (Sorted by Curvature RMSE):")
    print(results_df.head(5))

    out_path = save_results(results_df, results_dir)
    print(f"\n✅ Results saved to: {out_path}")

if __name__ == "__main__":
    main()