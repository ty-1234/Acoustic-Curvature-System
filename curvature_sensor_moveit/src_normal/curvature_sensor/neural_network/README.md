# Neural Network Models for Curvature Sensing

![Neural Network Models Banner](https://via.placeholder.com/800x200?text=Curvature+Sensor+Neural+Network+Models)

## Overview

The Neural Network module contains scripts for training, evaluating, and comparing machine learning models that predict curvature and position from acoustic FFT data. This module implements various regression techniques to achieve high-precision predictions and exports comprehensive evaluation metrics and visualizations.

## Features

- **Multi-output Regression**: Simultaneous prediction of both curvature and position
- **Model Comparison**: Evaluation of multiple regression algorithms with LazyPredict
- **Hyperparameter Optimization**: Pre-optimized model parameters for ExtraTrees regressors
- **Comprehensive Evaluation**: Detailed metrics and visualizations for model performance
- **CSV Export**: Structured data exports for external analysis and visualization
- **Cross-validation**: Group-based validation to ensure generalization across different runs

## Development Process

The models were developed and refined through the following chronological process:

1. **Initial Model Comparison** (`lazypredict_runner.py`): Evaluated dozens of regression algorithms to identify the most promising approaches
2. **Multi-output Implementation** (`train_multioutput.py`): Applied the best performing models to simultaneously predict both curvature and position
3. **Hyperparameter Optimization** (`train_extratrees_optuna.py`): Fine-tuned the ExtraTrees model using Optuna for optimal performance
4. **Final Model Training** (`best_peram_extratrees_optuna.py`): Trained the final model with optimized parameters for production use
5. **Alternative Models** (`og_train_gpr_curvature.py`): Developed Gaussian Process Regression as an alternative approach

## Models Implemented

- **Various Regression Models**: Through LazyPredict integration, evaluated dozens of algorithms
- **Multi-output Regressors**: Applied best performers for simultaneous curvature and position prediction
- **ExtraTrees Regressor**: Optimized ensemble of extremely randomized trees
- **Gaussian Process Regression**: Non-parametric probabilistic model with uncertainty estimation

## Requirements

- Python 3.6+
- scikit-learn 1.0+
- pandas, numpy
- matplotlib, seaborn (for visualizations)
- joblib (for model persistence)
- optuna (for hyperparameter optimization)
- lazypredict (for automated model evaluation)

All dependencies can be installed using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```


## Usage

### Model Comparison (Step 1)

To evaluate multiple regression algorithms automatically:

```bash
python lazypredict_runner.py
```

This evaluates dozens of regression models and ranks them by performance on the curvature prediction task.

### Multi-output Implementation (Step 2)

To train models that predict both curvature and position:

```bash
python train_multioutput.py
```

This script applies the best-performing algorithms from the LazyPredict evaluation for multi-output regression.

### Hyperparameter Optimization (Step 3)

To optimize the ExtraTrees hyperparameters:

```bash
python train_extratrees_optuna.py
```

This uses Optuna to search for optimal hyperparameters for the ExtraTrees regressor.

### Final Model Training (Step 4)

To train the optimized ExtraTrees model:

```bash
python best_peram_extratrees_optuna.py
```

This will train a model with pre-optimized hyperparameters, evaluate performance, and export the results to `model_outputs/extratrees/BEST/`.

### Gaussian Process Model Training (Alternative)

To train the Gaussian Process Regression model:

```bash
python og_train_gpr_curvature.py
```

This script focuses on curvature prediction using GPR and outputs results to `model_outputs/gpr_curvature/`.

## Output Files

Each model training script generates standardized output files:

- **Trained Model**: `.pkl` files containing the trained model
- **Feature Scaler**: Scaling parameters for preprocessing new data
- **Prediction Data**: CSV files with true vs. predicted values
- **Performance Metrics**: Structured metrics in CSV format
- **Error Analysis**: Error distributions and statistics
- **Feature Importance**: For tree-based models, relative importance of features
- **Visualizations**: PNG files with performance plots

## Model Selection

The ExtraTrees model (`extratrees_optuna_model.pkl`) provides the best overall performance for both curvature and position prediction, with:
- Low RMSE for curvature prediction (typically <0.005 mm⁻¹)
- High R² scores (>0.95 for curvature)
- Excellent position tracking capability

## Integration

The trained models from this module are used by the real-time prediction system. The `extratrees_optuna_model.pkl` and `feature_scaler.pkl` files should be copied to the model directory for real-time inference.

## Development Workflow

1. **Model Exploration**: Use LazyPredict to evaluate multiple algorithms and identify promising approaches
2. **Multi-output Implementation**: Develop models capable of predicting both curvature and position
3. **Hyperparameter Tuning**: Optimize model parameters using Optuna for ExtraTrees regression
4. **Final Model Training**: Train the final model with optimized parameters
5. **Evaluation**: Generate comprehensive performance metrics
6. **Export**: Save models, scalers, metrics, and visualizations for further use
7. **Comparison**: Compare different modeling approaches using standardized metrics

## Directory Structure

```
neural_network/
├── lazypredict_runner.py              # Step 1: Multiple model evaluation
├── train_multioutput.py               # Step 2: Multi-output regression variants
├── train_extratrees_optuna.py         # Step 3: Hyperparameter optimization
├── best_peram_extratrees_optuna.py    # Step 4: ExtraTrees model training
├── og_train_gpr_curvature.py          # GPR model training (alternative)
│
├── model_outputs/                     # Model outputs and results
│   ├── extratrees/                    # ExtraTrees model outputs
│   │   └── BEST/                      # Best model configuration
│   │       ├── extratrees_optuna_model.pkl  # Trained model
│   │       ├── feature_scaler.pkl     # Feature scaling parameters
│   │       ├── metrics.json           # Performance metrics
│   │       └── et_*.csv               # Detailed CSV exports
│   │
│   └── gpr_curvature/                 # GPR model outputs
│       ├── gpr_curvature_model.pkl    # Trained GPR model
│       ├── scaler.pkl                 # Feature scaling parameters
│       └── gpr_*.csv                  # Detailed CSV exports
│
└── extra.vs.GPR/                      # Comparison visualizations
    ├── true_vs_predicted.png          # Model prediction comparisons
    ├── residuals.png                  # Error residual plots
    ├── comparison_metrics_split.png   # Performance metric comparison
    └── error_*.png                    # Error distribution visualizations
```

## Further Development

The modular design of this system allows for easy extension:

1. Add new models by creating additional training scripts
2. Implement alternative feature engineering approaches 
3. Extend metrics and visualizations for more detailed analysis
4. Experiment with deep learning approaches for improved accuracy

## Author

**Bipindra Rai**  
Final Year Engineering Project  
Class of 2025

---

© 2025 Bipindra Rai. All Rights Reserved.
