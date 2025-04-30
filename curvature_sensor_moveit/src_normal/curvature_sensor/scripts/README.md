# Curvature Sensing System: Scripts Module

![Scripts Banner](https://via.placeholder.com/800x200?text=Curvature+Sensor+Scripts+Module)

## Overview

The Scripts module forms the core data acquisition and processing pipeline for the Curvature Sensing System. This collection of Python scripts handles everything from audio signal generation to robot control, sensor data collection, and preprocessing required for machine learning applications.

## Module Contents

| Script Name | Description |
|-------------|-------------|
| main.py | Central hub with a menu-driven interface for accessing all system functions |
| 2000hz_wav_creator.py | Generates multi-tone WAV files (200Hz-2000Hz) for acoustic excitation |
| curvature_data_collector.py | Records audio and performs real-time FFT analysis |
| curvature_ros.py | Controls the Franka robot arm for systematic data collection |
| csv_merger.py | Combines audio FFT data with robot positional data by timestamp |
| all_merger.py | Merges all experiment datasets into a single combined dataset |
| preprocess_training_features.py | Processes raw data into engineered features for ML training |
| curvature_fft_utils.py | Utility functions for FFT processing and frequency analysis |

## Data Collection Workflow

The scripts are designed to be used in a specific sequence to create a complete dataset:

1. **Generate Test Signals** - Create WAV files with consistent frequency content
2. **Collect Sensor Data** - Record FFT data from acoustic interactions
3. **Control Robot** - Move robot arm through test positions and record location data
4. **Merge Data Sources** - Combine audio and position data using timestamps
5. **Prepare Training Data** - Engineer features for machine learning models

## Dependencies

### Required Python Packages
```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
sounddevice>=0.4.3
PyAudio>=0.2.11
matplotlib>=3.5.0
```

### System Requirements
- Python 3.6+
- Audio recording capabilities
- When using robot control: ROS (Robot Operating System) installation

### ROS Dependencies (for Robot Control)
- rospy
- tf
- moveit_commander
- geometry_msgs
- controller_manager_msgs
- shape_msgs
- actionlib_msgs

## Installation

1. Ensure Python 3.6+ is installed on your system
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. For robot control features, install ROS and required ROS packages

## Usage

### Using the Menu Interface

The simplest way to access all functions is through the menu-driven interface:

```bash
python scripts/main.py
```

This will present a menu with options for running each script:
```
🔊 Curvature Sensing System Control Hub
═════════════════════════════════════════
Please select a task to run:
1. Generate WAV (2000hz_wav_creator.py)
2. Collect Data (curvature_data_collector.py)
3. ROS Control (curvature_ros.py)
4. Merge CSV Files (csv_merger.py)
5. Merge All Files Into Combined Dataset (all_merger.py)
6. Preprocess Data (preprocess_training_features.py)
═════════════════════════════════════════
```

### Running Individual Scripts

Alternatively, scripts can be run individually:

#### Generate WAV Files
```bash
python scripts/2000hz_wav_creator.py
```
This creates a `300s.wav` file with test tones from 200Hz to 2000Hz.

#### Collect FFT Data
```bash
python scripts/curvature_data_collector.py
```
Records audio via the system's microphone, extracts FFT features, and saves to CSV.

#### Control Robot
```bash
python scripts/curvature_ros.py
```
Controls the Franka robot arm to perform systematic movements for data collection.

#### Merge Data Files
```bash
python scripts/csv_merger.py
```
Combines audio and positional data by matching timestamps.

#### Create Combined Dataset
```bash
python scripts/all_merger.py
```
Merges all experimental data into a single dataset file.

#### Preprocess Features
```bash
python scripts/preprocess_training_features.py
```
Creates engineered features required for machine learning models.

## File Structure and Data Flow

```
curvature_sensor/
├── scripts/
│   ├── main.py                    # Menu interface
│   ├── 2000hz_wav_creator.py      # Step 1: Generate WAV files
│   ├── curvature_data_collector.py # Step 2: Record audio data
│   ├── curvature_ros.py           # Step 3: Control robot
│   ├── csv_merger.py              # Step 4: Merge data by timestamp
│   ├── all_merger.py              # Step 5: Create combined dataset
│   ├── preprocess_training_features.py # Step 6: Engineer features
│   └── curvature_fft_utils.py     # Utilities for FFT processing
│
├── csv_data/                      # Data directory
│   ├── raw/                       # Step 2-3 output
│   │   ├── raw_audio_*.csv        # Audio FFT data
│   │   └── raw_robot_*.csv        # Robot position data
│   │
│   ├── merged/                    # Step 4 output
│   │   └── merged_*.csv           # Timestamp-aligned files
│   │
│   ├── combined_dataset.csv       # Step 5 output
│   │
│   └── preprocessed/              # Step 6 output
│       └── preprocessed_training_dataset.csv
```

## File Naming Conventions

The system uses a specific naming convention for data files:

- **Audio Data**: `raw_audio_{curvature}[test ID].csv`
- **Robot Data**: `raw_robot_{curvature}[test ID].csv`
- **Merged Data**: `merged_{curvature}[test ID].csv`

Where:
- `{curvature}` is the curvature value with decimal points replaced by underscores (e.g., `0_01818`)
- `[test ID]` is an optional identifier in square brackets (e.g., `[test 1]`)

## Script Details

### 2000hz_wav_creator.py

Generates a multi-tone WAV file with frequencies from 200Hz to 2000Hz (in 200Hz steps).

**Key Functions:**
- `generate_multi_tone()`: Creates a combined sine wave signal
- `main()`: Sets parameters and saves the WAV file

**Output:**
- `300s.wav`: 5-minute WAV file with test tones

### curvature_data_collector.py

Records audio and extracts FFT features in real-time.

**Key Functions:**
- `check_microphone()`: Validates microphone functionality
- `audio_callback()`: Processes audio blocks and extracts FFT features
- `main()`: Handles user input and recording control

**Input:**
- User-provided curvature value (from radius)

**Output:**
- `raw_audio_{curvature}.csv`: FFT features with timestamps

### curvature_ros.py

Controls the Franka robot arm to move through data collection positions.

**Key Functions:**
- `move_Downwards()`: Performs systematic movements through test sections
- `set_force_publisher()`: Controls force application
- `open_gripper()/close_gripper()`: Manages the gripper

**Output:**
- `raw_robot_{curvature}.csv`: Robot position data with timestamps

### csv_merger.py

Merges audio and robot data using timestamp alignment.

**Key Functions:**
- `merge_csvs()`: Merges file pairs based on nearest timestamps

**Input:**
- `raw_audio_{curvature}.csv`
- `raw_robot_{curvature}.csv`

**Output:**
- `merged_{curvature}.csv`: Combined audio and position data

### all_merger.py

Combines all merged files into a single dataset.

**Key Functions:**
- `parse_filename()`: Extracts curvature and run ID from filenames

**Input:**
- All files in merged

**Output:**
- `combined_dataset.csv`: Complete dataset with all experiments

### preprocess_training_features.py

Processes raw data to create engineered features for ML training.

**Key Functions:**
- `process_curvature_data()`: Performs feature engineering

**Input:**
- `combined_dataset.csv`

**Output:**
- `preprocessed_training_dataset.csv`: Prepared ML features

### curvature_fft_utils.py

Utility functions for FFT processing.

**Key Functions:**
- `extract_fft_features()`: Extracts frequency domain features from signals

## Integration with Other Modules

- The preprocessed data from this module is used by the `neural_network` module for model training
- Trained models are used by the `real_time_prediction` module for live sensing

## Troubleshooting

### Audio Recording Issues
- **No sound detected**: Check if your microphone is properly connected and selected
- **Low amplitude**: Increase microphone gain or check the audio source volume
- **CSV not saved**: Ensure the script has write permissions to the output directory

### Robot Control Issues
- **Missing ROS packages**: Install required ROS dependencies
- **Controller errors**: Use the recovery mechanism in curvature_ros.py
- **Connection failures**: Verify network connection to the robot

### Data Processing Issues
- **Timestamp misalignment**: Ensure system clocks are synchronized
- **Missing features**: Check for NaN values in the raw data
- **Merging failures**: Verify file naming conventions match expected patterns

## Authors

**Bipindra Rai**  

**Tariq** (ROS Framework)
