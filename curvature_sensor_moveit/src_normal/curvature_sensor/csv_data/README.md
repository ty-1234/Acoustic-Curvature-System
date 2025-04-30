# Curvature Sensing System: Data Repository

![Data Banner](https://via.placeholder.com/800x200?text=Curvature+Sensor+Data+Repository)

## Overview

The csv_data directory is the central data repository for the Curvature Sensing System, containing all experimental data from data collection through preprocessing. This directory houses the complete data pipeline from raw sensor recordings to feature-engineered datasets that are consumed by the neural network module for model training and evaluation.

## Directory Structure

```
csv_data/
├── raw/                      # Raw data from data collection phase
│   ├── raw_audio_*.csv       # FFT frequency data from microphone recordings
│   └── raw_robot_*.csv       # Position data from robot arm movements
│
├── merged/                   # Synchronized data files
│   └── merged_*.csv          # Audio and robot data matched by timestamp
│
├── preprocessed/             # ML-ready processed data
│   └── preprocessed_training_dataset.csv  # Feature-engineered dataset for neural networks
│
└── combined_dataset.csv      # All experimental data combined into a single dataset
```

## Data Flow Process

This directory represents the complete data pipeline for the curvature sensing system:

1. **Data Collection** (inputs to the `raw/` directory)
    - The scripts module collects raw sensor data through `curvature_data_collector.py`
    - Robot positional data is recorded through `curvature_ros.py`

2. **Data Merging** (creates files in the `merged/` directory)
    - `csv_merger.py` synchronizes audio and robot data based on timestamps
    - The resulting files maintain curvature value and test identification

3. **Data Aggregation** (creates `combined_dataset.csv`)
    - `all_merger.py` combines data from all experiments into a single dataset
    - Adds RunID and Curvature_Label columns for identification and machine learning

4. **Data Preprocessing** (creates files in `preprocessed/` directory)
    - `preprocess_training_features.py` engineers features for machine learning
    - Creates the final training dataset used by neural network models

## File Content Details

### Raw Data Files

Raw data files contain initial measurements from data collection:

* **Audio FFT Files**: Contain frequency domain features extracted from microphone recordings:
  ```
  FFT_200Hz,FFT_400Hz,...,FFT_2000Hz,Curvature,Timestamp
  0.00032,0.00063,...,0.07643,0.01,2025-04-17T17:53:22.911506
  ```

* **Robot Position Files**: Contain physical location of the robot end effector:
  ```
  Section,PosX,PosY,PosZ,Timestamp
  0cm,0.446576,0.311206,0.663847,2025-04-17T17:52:27.861814
  ```

### Merged and Combined Files

These files synchronize the sensor and position data:

* **Merged Files**: Align audio and robot data by timestamp:
  ```
  FFT_200Hz,...,FFT_2000Hz,Section,PosX,PosY,PosZ,Curvature,Timestamp
  0.00021,...,0.08597,2cm,0.447734,0.310688,0.642657,0.01,2025-04-17T17:53:05.264010
  ```

* **Combined Dataset**: Aggregates all experimental data with experiment identifiers:
  ```
  FFT_200Hz,...,FFT_2000Hz,Section,PosX,PosY,PosZ,Curvature_Label,RunID,Timestamp
  ```

### Preprocessed Dataset

The `preprocessed_training_dataset.csv` file contains engineered features specifically designed for machine learning:

```
FFT_200Hz,...,FFT_Mean,FFT_Std,FFT_Peak_Index,FFT_Centroid,Low_Band_Mean,Mid_Band_Mean,High_Band_Mean,Position_cm,Curvature_Label,RunID,Curvature_Active
```

This dataset includes:
- Raw FFT frequency features
- Statistical features (mean, standard deviation, etc.)
- Frequency band aggregations
- Peak detection features
- Position and curvature labels for supervised learning
- Experiment identification columns

## Integration with Neural Network Module

The preprocessed dataset in this directory serves as the primary input to the neural_network module:

1. **Input for Model Training**: The `preprocessed_training_dataset.csv` file is the primary training data source for all machine learning models.

2. **Dataset Characteristics**:
    - Multiple curvature values ranging from 0.005 mm⁻¹ to 0.05 mm⁻¹
    - Position values from 0cm to 5cm in 1cm increments
    - Engineered features for better model performance

3. **Neural Network Integration**:
    - Scripts like `train_extratrees_optuna.py` and `train_gpr_curvature.py` load this dataset
    - Data is split into training and testing sets for model evaluation
    - Models are trained to predict both curvature and position values
    - Trained models are exported for use in real-time prediction

## Data Collection Methodology

The data in this repository was collected through a systematic experimental procedure:

1. **Controlled Environment**: Data collection in a laboratory setting with constant environmental conditions
2. **Multiple Test Objects**: Various curvature values using calibrated test objects
3. **Robot-Assisted Precision**: Franka robot arm used for precise and repeatable positioning
4. **Multiple Runs**: Each curvature value tested multiple times for statistical validity
5. **Multi-tone Audio**: Testing used reference tones from 200Hz to 2000Hz in 200Hz steps

## Author

**Bipindra Rai**  
Final Year Computer Science Project  
Class of 2025

---

© 2025 Bipindra Rai. All Rights Reserved.