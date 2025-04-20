import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lazypredict.Supervised import LazyRegressor

# Add parent directory to path to handle relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# === Step 1: Load the merged dataset ===
# Use the parent directory path to correctly locate the CSV file
df = pd.read_csv(os.path.join(parent_dir, "csv_data", "combined_dataset.csv"))

# === Step 2: Select only the FFT feature columns for inputs (X) ===
# These are your input signals at different frequencies
fft_cols = [col for col in df.columns if col.startswith("FFT_")]
X = df[fft_cols]

# === Step 3: Set the target variable (y) ===
# This is the curvature value you want the model to predict
y = df["Curvature_Label"]

# === Step 4: Normalize the FFT features ===
# Many models perform better if inputs are scaled to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# === Step 5: Split the dataset into training and test sets ===
# We use 80% for training, 20% for testing
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=42
)

# === Step 6: Initialize LazyRegressor ===
# This runs many regression models and compares their performance
reg = LazyRegressor(verbose=1, ignore_warnings=True, custom_metric=None)

# === Step 7: Fit and compare models ===
# This line does all the work: trains many regressors and evaluates them
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# === Step 8: Display the results ===
# This will print a ranked list of models with their R^2 score, RMSE, etc.
print("\nTop performing regression models:")
print(models)

# === Step 9 (Optional): Save model results to CSV for your report ===
output_path = os.path.join(parent_dir, "csv_data", "merged", "lazypredict_model_results.csv")
models.to_csv(output_path)
print(f"Results saved to: {output_path}")
