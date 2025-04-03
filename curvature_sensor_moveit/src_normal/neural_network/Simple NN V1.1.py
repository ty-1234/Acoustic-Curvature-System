"""
Simple_NN_V2.py

This script builds, trains, evaluates, and visualizes a neural network
for predicting shear force measurements from tactile sensor data. It incorporates
best practices to address warnings and enhance model architecture.

Author: Halden Banfield
Date: November 2024
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import warnings

# Suppress TensorFlow warnings for cleaner output (optional)
# You can comment these lines out if you prefer to see all warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# ----------------------------- Configuration -----------------------------

# Path to the sensor data CSV file
DATA_FILE = 'sensor_data.csv'  # Ensure this file exists or set GENERATE_SYNTHETIC_DATA to True

# Flag to generate synthetic data if DATA_FILE does not exist
GENERATE_SYNTHETIC_DATA = False  # Set to True to enable synthetic data generation

# Synthetic data parameters (used only if GENERATE_SYNTHETIC_DATA is True)
NUM_SAMPLES = 1000  # Number of data samples to generate
NUM_FEATURES = 20    # Number of feature columns
NOISE_LEVEL = 0.5    # Standard deviation of Gaussian noise added to the target

# Neural Network parameters
HIDDEN_UNITS = 64
OUTPUT_DIM = 1
ACTIVATION = 'relu'
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'mse'
METRICS = ['mae']
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10

# Model saving path
MODEL_SAVE_PATH = 'tactile_sensor_model_v2.h5'

# ------------------------- Synthetic Data Generation -----------------------

def generate_synthetic_data(num_samples, num_features, noise_level):
    """
    Generates synthetic sensor data for testing purposes.

    Parameters:
        num_samples (int): Number of samples to generate.
        num_features (int): Number of feature columns.
        noise_level (float): Standard deviation of Gaussian noise added to the target.

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

# ----------------------------- Data Loading --------------------------------

def load_data(file_path=DATA_FILE, generate_synthetic=False):
    """
    Loads sensor data from a CSV file. If the file does not exist and
    generate_synthetic is True, synthetic data is generated and saved.

    Parameters:
        file_path (str): Path to the CSV file containing sensor data.
        generate_synthetic (bool): Whether to generate synthetic data if file is missing.

    Returns:
        pd.DataFrame: Loaded sensor data.
    """
    if not os.path.isfile(file_path):
        if generate_synthetic:
            print(f"Data file '{file_path}' not found. Generating synthetic data...")
            synthetic_data = generate_synthetic_data(NUM_SAMPLES, NUM_FEATURES, NOISE_LEVEL)
            synthetic_data.to_csv(file_path, index=False)
            print(f"Synthetic data generated and saved to '{file_path}'.")
            return synthetic_data
        else:
            raise FileNotFoundError(f"Data file '{file_path}' does not exist and synthetic data generation is disabled.")
    else:
        print(f"Loading data from '{file_path}'...")
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data

# --------------------------- Data Preprocessing ----------------------------

def preprocess_data(data, target_column='force'):
    """
    Preprocesses the sensor data by separating features and target, scaling features,
    and splitting into training and testing sets.

    Parameters:
        data (pd.DataFrame): The sensor data.
        target_column (str): The name of the target variable column.

    Returns:
        tuple: Scaled training and testing features and corresponding targets, along with the scaler.
    """
    # Check if target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

    # Feature matrix (all columns except target)
    X = data.drop(columns=[target_column]).values
    # Target vector
    y = data[target_column].values

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature Scaling: Standardize features to have zero mean and unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data preprocessing completed.")
    print(f"Training samples: {X_train_scaled.shape[0]}, Testing samples: {X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ---------------------------- Model Definition -----------------------------

def build_model(input_dim, hidden_units=HIDDEN_UNITS, output_dim=OUTPUT_DIM, activation=ACTIVATION):
    """
    Builds a simple feedforward neural network model with an explicit Input layer.

    Parameters:
        input_dim (int): Number of input features.
        hidden_units (int): Number of neurons in hidden layers.
        output_dim (int): Number of output neurons.
        activation (str): Activation function for hidden layers.

    Returns:
        keras.Model: Compiled Keras model.
    """
    model = keras.Sequential([
        Input(shape=(input_dim,)),  # Explicit Input layer to eliminate warnings
        layers.Dense(hidden_units, activation=activation),
        layers.Dense(hidden_units, activation=activation),
        layers.Dense(output_dim)  # No activation for regression
    ])

    # Compile the model
    model.compile(
        optimizer=OPTIMIZER,
        loss=LOSS_FUNCTION,
        metrics=METRICS
    )

    # Display the model architecture
    model.summary()
    return model

# ------------------------------- Training ----------------------------------

def train_model(model, X_train, y_train):
    """
    Trains the neural network model with early stopping.

    Parameters:
        model (keras.Model): The compiled Keras model.
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training targets.

    Returns:
        keras.History: Training history object.
    """
    # Define early stopping to prevent overfitting
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1
    )
    return history

# ------------------------------ Evaluation ---------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test dataset.

    Parameters:
        model (keras.Model): The trained Keras model.
        X_test (np.ndarray): Scaled testing features.
        y_test (np.ndarray): Testing targets.

    Returns:
        tuple: Test loss and test metrics.
    """
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest Loss (MSE): {test_loss:.4f}')
    print(f'Test MAE: {test_mae:.2f} N')
    return test_loss, test_mae

# ----------------------------- Visualization -------------------------------

def plot_results(history, y_test, y_pred):
    """
    Plots the training history and actual vs. predicted forces.

    Parameters:
        history (keras.History): Training history object.
        y_test (np.ndarray): Actual test targets.
        y_pred (np.ndarray): Predicted test targets.
    """
    # Plot training & validation loss values
    plt.figure(figsize=(14, 6))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Model Loss Over Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Subplot 2: Actual vs Predicted
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
    plt.title('Actual vs Predicted Forces')
    plt.xlabel('Actual Force (N)')
    plt.ylabel('Predicted Force (N)')
    # Plot diagonal line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.tight_layout()

    # Show plots
    plt.show()

# ----------------------------- Save Model ----------------------------------

def save_model(model, save_path=MODEL_SAVE_PATH):
    """
    Saves the trained Keras model to a file.

    Parameters:
        model (keras.Model): The trained Keras model.
        save_path (str): Path to save the model file.
    """
    model.save(save_path)
    print(f"Model saved to '{save_path}'.")

# -------------------------------- Main -------------------------------------

def main():
    """
    Main function to execute the neural network workflow.
    """
    # Step 1: Load Data
    try:
        data = load_data(file_path=DATA_FILE, generate_synthetic=GENERATE_SYNTHETIC_DATA)
    except FileNotFoundError as e:
        print(e)
        return

    # Step 2: Preprocess Data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data, target_column='force')

    # Step 3: Build Model
    input_dim = X_train.shape[1]  # Number of features
    model = build_model(input_dim=input_dim, hidden_units=HIDDEN_UNITS, output_dim=OUTPUT_DIM, activation=ACTIVATION)

    # Step 4: Train Model
    history = train_model(model, X_train, y_train)

    # Step 5: Evaluate Model
    test_loss, test_mae = evaluate_model(model, X_test, y_test)

    # Step 6: Make Predictions
    y_pred = model.predict(X_test).flatten()

    # Step 7: Visualize Results
    plot_results(history, y_test, y_pred)

    # Step 8: Save Model
    save_model(model, save_path=MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
