# Curvature Sensing System

![Curvature Sensing System Banner](https://via.placeholder.com/800x200?text=Curvature+Sensing+System)

## Overview

The Curvature Sensing System is an innovative solution that uses acoustic frequency analysis to measure both curvature and position simultaneously. By analyzing how multi-tone audio signals are modified when passing through a sensing channel, the system can detect subtle changes in curvature with high precision while also tracking position along the sensor.

This project demonstrates the complete pipeline from data collection and processing through model training to real-time prediction, utilizing signal processing, machine learning, and robotics integration.

## Key Features

- **Acoustic-based sensing** for non-contact curvature and position detection
- **Multi-tone frequency analysis** (200Hz-2000Hz) for robust feature extraction
- **Robot-assisted data collection** for precise calibration and validation
- **Machine learning models** for accurate prediction of both curvature and position
- **Real-time prediction system** with live visualization and data logging
- **Comprehensive data pipeline** from raw signal collection to trained models

## Directory Structure

The project is organized into four main modules, each handling a specific part of the workflow:

```
curvature_sensor/
├── scripts/              # Data collection and preprocessing scripts
├── csv_data/             # Repository for all data files
├── neural_network/       # Model training and evaluation scripts
├── real_time_prediction/ # Real-time inference system
└── requirements.txt      # Project dependencies
```

## Workflow

The Curvature Sensing System follows a structured workflow from data collection to real-time prediction:

### 1. Data Collection Phase

1. **Generate Audio Test Signal**
   - Use 2000hz_wav_creator.py to create WAV files with test tones (200Hz-2000Hz)
   - These signals provide consistent acoustic excitation for the curvature sensor

2. **Collect Sensor Data**
   - Play the generated WAV file through speakers
   - Run curvature_data_collector.py to record and process audio via microphone
   - Extract FFT features in real-time and save to `csv_data/raw/raw_audio_*.csv`

3. **Robot Control and Position Data**
   - Simultaneously run curvature_ros.py on the ROS-enabled PC
   - Control the robot to move systematically through test positions (0-5cm)
   - Record position data with timestamps to `csv_data/raw/raw_robot_*.csv`

### 2. Data Processing Phase

1. **Merge Data Sources**
   - Run csv_merger.py to combine audio and position data based on timestamps
   - Creates aligned data files in `csv_data/merged/merged_*.csv`

2. **Combine Multiple Experiments**
   - Run all_merger.py to aggregate all test runs into a single dataset
   - Creates combined_dataset.csv with experiment identifiers

3. **Feature Engineering**
   - Run preprocess_training_features.py to generate advanced features
   - Creates preprocessed_training_dataset.csv with:
     - Statistical features (means, standard deviations)
     - Frequency band aggregations
     - Position information
     - Curvature labels

### 3. Model Development Phase

1. **Model Evaluation**
   - Use lazypredict_runner.py to evaluate multiple algorithm types
   - Identifies the most promising approaches for curvature prediction

2. **Hyperparameter Optimization**
   - Run train_extratrees_optuna.py to optimize model parameters
   - Systematically searches for optimal hyperparameters using cross-validation

3. **Final Model Training**
   - Train the optimized ExtraTrees model with best_peram_extratrees_optuna.py
   - Train GPR model with og_train_gpr_curvature.py for comparison
   - Export models, evaluation metrics, and visualizations to model_outputs

4. **Model Comparison**
   - Run graphmaker_compare.py to compare model performance
   - Creates detailed visualizations in extra.vs.GPR

### 4. Real-Time Prediction Phase

1. **System Setup**
   - Copy trained models to model
   - Configure audio settings for microphone input

2. **Launch Prediction System**
   - Run orchestrator.py to start the real-time system
   - Orchestrator coordinates frequency generation, feature extraction, and prediction

3. **Live Operation**
   - The system runs continuously, displaying curvature and position predictions
   - Real-time visualization via the GUI interface
   - Predictions are logged to prediction_logs

### 5. Analysis and Iteration

1. **Performance Analysis**
   - Analyze prediction logs to evaluate real-world performance
   - Compare with expected values to identify areas for improvement

2. **System Refinement**
   - Adjust feature engineering parameters
   - Collect additional training data for challenging scenarios
   - Retrain models with enhanced datasets

## Module Details

### Scripts Module

The scripts directory contains all Python scripts for data collection, robot control, and data preprocessing:

- **Main Interface**: main.py provides a menu-driven interface to all system functions
- **Data Collection**: 
  - `2000hz_wav_creator.py` generates reference audio signals
  - `curvature_data_collector.py` records and processes audio data
  - curvature_ros.py controls the Franka robot arm for position data
- **Data Processing**:
  - `csv_merger.py` combines audio and position data by timestamp
  - `all_merger.py` merges multiple experiment runs
  - `preprocess_training_features.py` creates ML-ready features
- **Utilities**:
  - `curvature_fft_utils.py` provides FFT processing functions

### CSV Data Module

The csv_data directory serves as the central data repository:

- **Raw Data**: Collected sensor data in `raw/`
  - Audio FFT features in `raw_audio_*.csv`
  - Robot position data in `raw_robot_*.csv`
- **Processed Data**:
  - Timestamp-synchronized data in `merged/*.csv`
  - Combined dataset in `combined_dataset.csv`
  - ML-ready features in preprocessed_training_dataset.csv

### Neural Network Module

The neural_network directory contains all scripts for model training and evaluation:

- **Model Development**:
  - lazypredict_runner.py for initial model comparison
  - train_multioutput.py for multi-output regression
  - `train_extratrees_optuna.py` for hyperparameter optimization
  - best_peram_extratrees_optuna.py for final ExtraTrees model
  - og_train_gpr_curvature.py for Gaussian Process Regression
- **Model Analysis**:
  - graphmaker_compare.py for comparative visualization
- **Outputs**:
  - Trained models in `model_outputs/`
  - Performance metrics and visualizations
  - Comprehensive CSV exports for analysis

### Real-Time Prediction Module

The real_time_prediction directory contains the live inference system:

- **Core Components**:
  - `orchestrator.py` coordinates all real-time components
  - `rt_frequency_gen.py` generates reference tones
  - `rt_feature_extraction.py` processes audio input
  - `rt_model.py` makes predictions using trained models
  - `rt_gui.py` visualizes predictions in real-time
  - rt_logger.py logs predictions for later analysis
- **Supporting Files**:
  - Trained models in `model/`
  - Prediction logs in `prediction_logs/`

## Installation

### Requirements

- Python 3.6+ (3.10+ recommended for real-time prediction)
- ROS (Robot Operating System) for robot control features
- PortAudio for audio processing

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/username/curvature-sensor.git
   cd curvature-sensor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For robot control, ensure ROS is properly installed with required packages:
   ```bash
   sudo apt install ros-noetic-moveit
   sudo apt install ros-noetic-franka-ros  # If using Franka robot
   ```

4. Verify audio device configuration for your system.

## Usage

### Data Collection

1. Generate reference WAV file:
   ```bash
   python scripts/2000hz_wav_creator.py
   ```

2. Run data collection (audio and robot) in separate terminals:
   ```bash
   python scripts/curvature_data_collector.py
   python scripts/curvature_ros.py
   ```

3. Merge and process data:
   ```bash
   python scripts/csv_merger.py
   python scripts/all_merger.py
   python scripts/preprocess_training_features.py
   ```

### Model Training

1. Run initial model comparison:
   ```bash
   python neural_network/lazypredict_runner.py
   ```

2. Test multi-output regression:
   ```bash
   python neural_network/train_multioutput.py
   ```

3. Optimize model hyperparameters:
   ```bash
   python neural_network/train_extratrees_optuna.py
   ```

4. Train final optimized model:
   ```bash
   python neural_network/best_peram_extratrees_optuna.py
   ```

5. Compare model performance:
   ```bash
   python neural_network/graphmaker_compare.py
   ```

### Real-Time Operation

1. Copy trained models to real-time module:
   ```bash
   cp neural_network/model_outputs/extratrees/BEST/extratrees_optuna_model.pkl real_time_prediction/model/
   cp neural_network/model_outputs/extratrees/BEST/feature_scaler.pkl real_time_prediction/model/
   ```

2. Launch real-time prediction system:
   ```bash
   python real_time_prediction/orchestrator.py
   ```

## Technical Details

### Curvature Sensing Method

The system uses multi-tone audio signals (200Hz-2000Hz) as a sensing medium. When the acoustic channel undergoes curvature, the frequency response changes in characteristic ways that can be detected through FFT analysis. These changes are then mapped to curvature and position values using machine learning models.

### Data Collection Methodology

1. **Controlled Environment**: Laboratory setting with consistent conditions
2. **Calibrated Test Objects**: Multiple known curvature values (0.005-0.05 mm⁻¹)
3. **Robot-Assisted Positioning**: Precise and repeatable placement along the sensor
4. **Multiple Test Runs**: Statistical validity through repeated measurements

### Machine Learning Approach

1. **Feature Engineering**: Statistical and frequency-domain features
2. **Model Selection**: ExtraTrees Regressor for optimal performance
3. **Multi-output Regression**: Simultaneous prediction of position and curvature
4. **Cross-validation**: Group-based validation to ensure model generalization

### Real-Time System

1. **Frequency Generation**: Outputs consistent multi-tone signals
2. **FFT Analysis**: Real-time extraction of frequency features
3. **Prediction Pipeline**: Pre-processing, scaling, and inference
4. **GUI Visualization**: User-friendly display of sensing results
5. **Data Logging**: Continuous recording of predictions

## Future Work

- Enhance noise robustness through adaptive filtering
- Improve position tracking accuracy
- Implement real-time model retraining capabilities
- Integrate with robotic control systems
- Develop mobile/embedded version for portable applications

---
*Author: Bipindra Rai*