"""
generate_test_csv.py

This script generates a synthetic sensor data CSV file for testing the neural network.
The CSV file will contain feature columns named 'feature_1' to 'feature_20' and a target column 'force'.

Author: Halden Banfield
Date: November 2024
"""

import numpy as np
import pandas as pd

# Configuration
NUM_SAMPLES = 1000  # Number of data samples to generate
NUM_FEATURES = 20    # Number of feature columns
NOISE_LEVEL = 0.5    # Standard deviation of Gaussian noise added to the target

# Function to generate synthetic data
def generate_synthetic_data(num_samples, num_features, noise_level):
    """
    Generates synthetic sensor data.

    Parameters:
        num_samples (int): Number of samples to generate.
        num_features (int): Number of feature columns.
        noise_level (float): Standard deviation of noise added to the target.

    Returns:
        pd.DataFrame: DataFrame containing synthetic features and target 'force'.
    """
    np.random.seed(42)  # For reproducibility
    X = np.random.randn(num_samples, num_features)
    weights = np.random.randn(num_features)
    y = X @ weights + np.random.randn(num_samples) * noise_level
    feature_columns = [f'feature_{i+1}' for i in range(num_features)]
    data = pd.DataFrame(X, columns=feature_columns)
    data['force'] = y
    return data

# Generate data
synthetic_data = generate_synthetic_data(NUM_SAMPLES, NUM_FEATURES, NOISE_LEVEL)

# Save to CSV
csv_filename = 'sensor_data_test.csv'
synthetic_data.to_csv(csv_filename, index=False)
print(f"Synthetic test data generated and saved to '{csv_filename}'.")
