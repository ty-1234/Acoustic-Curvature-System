# Curvature Sensing System: Scripts Module

## Overview

The `scripts` folder contains essential Python scripts for the Curvature Sensing System. These scripts handle data collection, processing, robot control, and feature extraction. Below is a detailed description of the included scripts.

## Folder Structure

```
scripts/
├── main.py                    # Central hub with a menu-driven interface
├── curvature_data_collector.py # Records audio and performs real-time FFT analysis
├── curvature_ros.py           # Controls the Franka robot arm for data collection
├── csv_sync.py                # Synchronizes and merges audio and robot data
├── curvature_fft_utils.py     # Utility functions for FFT processing
├── frequency_gen.py           # Generates multi-tone signals for acoustic excitation
├── trimmer.py                 # Trims raw data files for preprocessing
├── requirements.txt           # Python dependencies for the scripts
```

## Script Details

### 1. `main.py`
- **Description**: Central hub script with a menu-driven interface to access all system functions.
- **Usage**: Run this script to interact with the system through a menu.
- **Key Features**:
  - Provides options to run individual scripts for data collection, processing, and analysis.

### 2. `curvature_data_collector.py`
- **Description**: Records audio and extracts FFT features in real-time.
- **Key Functions**:
  - `check_microphone()`: Validates microphone functionality.
  - `audio_callback()`: Processes audio blocks and extracts FFT features.
  - `main()`: Handles user input and recording control.
- **Output**: `raw_audio_{curvature}.csv` (FFT features with timestamps).

### 3. `curvature_ros.py`
- **Description**: Controls the Franka robot arm for systematic data collection.
- **Key Functions**:
  - `move_downwards()`: Performs systematic movements through test sections.
  - `set_force_publisher()`: Controls force application.
  - `open_gripper()/close_gripper()`: Manages the gripper.
- **Output**: `raw_robot_{curvature}.csv` (Robot position data with timestamps).

### 4. `csv_sync.py`
- **Description**: Synchronizes and merges audio and robot data based on timestamps.
- **Key Functions**:
  - `merge_csvs()`: Merges file pairs based on nearest timestamps.
- **Output**: `merged_{curvature}.csv` (Combined audio and position data).

### 5. `curvature_fft_utils.py`
- **Description**: Provides utility functions for FFT processing and frequency analysis.
- **Key Functions**:
  - `extract_fft_features()`: Extracts frequency domain features from signals.

### 6. `frequency_gen.py`
- **Description**: Generates multi-tone signals for acoustic excitation.
- **Key Features**:
  - Produces consistent reference frequencies for sensing.

### 7. `trimmer.py`
- **Description**: Trims raw data files for preprocessing.
- **Key Features**:
  - Removes unnecessary data segments to prepare for further processing.

### 8. `requirements.txt`
- **Description**: Lists the Python dependencies required for the scripts.
- **Installation**:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

### Using the Menu Interface
Run the following command to access the menu-driven interface:
```bash
python scripts/main.py
```

### Running Individual Scripts
Each script can also be run individually. For example:
- Collect FFT data:
  ```bash
  python scripts/curvature_data_collector.py
  ```
- Merge CSV files:
  ```bash
  python scripts/csv_sync.py
  ```

## Authors
- **Bipindra Rai**: Core functionality and system design.
- **Tariq**: ROS integration and robot control.