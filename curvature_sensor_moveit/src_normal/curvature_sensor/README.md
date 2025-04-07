# Curvature Sensing Pipeline Control Hub

## Overview

`main.py` is the primary script for the curvature sensor project. It serves as the **entry point** for executing various tasks related to curvature sensing, data collection, processing, and visualization. The script is modular and configurable, allowing users to dynamically load and execute specific modules based on their needs.

---

## How It Works

The `main.py` script provides a **menu-based interface** that allows users to select and execute specific tasks. Each task is implemented in a separate Python script located in the `scripts` directory, making the system modular and easy to maintain.

1. **Dynamic Module Import**: The script dynamically imports the required modules for each task using Python's `importlib.import_module`. This ensures that only the necessary code is loaded into memory, optimizing performance.
2. **Menu-Based Interface**: Users can select a task from the menu by entering the corresponding number.
3. **Task Execution**: Based on the user's input, the script dynamically imports the corresponding module and executes its `main()` function.

---

## Menu Options

When you run `main.py`, you will see the following menu:

### Explanation of Menu Options

1. **Generate WAV**:
   - **Script**: `2000hz_wav_creator.py`
   - **Description**: Generates WAV files for testing.
   - **Purpose**: Creates audio files with specific frequencies for sensor testing.

2. **Collect Data**:
   - **Script**: `curvature_data_collector.py`
   - **Description**: Records audio data and extracts FFT features for curvature sensing.
   - **Purpose**: Saves FFT features and timestamps to a CSV file.

3. **ROS Control**:
   - **Script**: `curvature_ros.py`
   - **Description**: Controls the robot using ROS and logs positional data.
   - **Purpose**: Moves the robot and collects data for curvature experiments.

4. **Merge CSV Files**:
   - **Script**: `csv_merger.py`
   - **Description**: Merges raw CSV files into a unified dataset.
   - **Purpose**: Combines audio and robot data for easier analysis.

5. **Preprocess Data**:
   - **Script**: `curvature_data_processor.py`
   - **Description**: Normalizes and preprocesses curvature data.
   - **Purpose**: Prepares data for machine learning or statistical analysis.

6. **Visualize FFT Data**:
   - **Script**: `fft_visualizer.py`
   - **Description**: Visualizes FFT data (from the merged folder) using boxplots, line plots, and heatmaps.
   - **Purpose**: Provides insights into FFT data through visualizations.

---

## Usage

### Running the Script

1. Navigate to the directory containing `main.py`:
   ```bash
   cd /Users/bipinrai/git/ass_245/curvature_sensor_moveit/src_normal/curvature_sensor
