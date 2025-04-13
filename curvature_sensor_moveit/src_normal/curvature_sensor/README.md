# Curvature Sensing System
[TODO: License: MIT] 

A comprehensive system for robotic curvature sensing through audio signal processing and machine learning.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [System Architecture](#system-architecture)
- [Usage](#usage)
- [Step-by-Step Workflow](#step-by-step-workflow)
- [Project Structure](#project-structure)
- [Pipeline Description](#pipeline-description)
- [Development and Extension](#development-and-extension)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The Curvature Sensing System is a modular framework that processes audio signals from robot-surface interactions to detect and analyze surface curvatures. It combines robotic control, audio processing, data collection, and visualization into a unified pipeline for tactile sensing research.

## Features

- **Audio Signal Generation**: Create test WAV files with controlled frequencies
- **Data Collection**: Record and extract FFT features from audio signals in real-time
- **ROS Integration**: Control robot movements and collect positional data
- **Data Processing**: Multi-stage pipeline for merging and preprocessing sensor data
- **Visualization**: Generate insightful plots and visualizations from FFT data
- **Modular Architecture**: Easy to extend with additional processing modules

## Requirements

### Software Dependencies
- Python 3.6+
- ROS (Robot Operating System)
- Python packages (see `requirements.txt`):
  - numpy
  - scipy
  - pandas
  - scikit-learn
  - matplotlib
  - pyaudio
  - tf
  - moveit_commander

### Hardware Requirements
- Compatible robot arm (tested with Franka Emika)
- Audio recording equipment
- Test surfaces with varying curvatures

## System Architecture

This system operates across two computers:

1. **Laptop**: Handles audio playback, recording, and FFT processing
2. **PC with ROS**: Controls the robot arm (Franka Emika) and manages physical interactions

The workflow involves generating test signals, playing/recording audio on the laptop, and simultaneously controlling the robot on the PC. Data from both systems is later merged using timestamps.

## Usage

The system is controlled through a menu-based interface in `main.py`:

```bash
python main.py
```

### Menu Options

1. **Generate WAV**:
   - **Script**: `2000hz_wav_creator.py`
   - **Description**: Generates WAV files for testing
   - **Purpose**: Creates audio files with specific frequencies for sensor testing

2. **Collect Data**:
   - **Script**: `curvature_data_collector.py`
   - **Description**: Records audio data and extracts FFT features
   - **Purpose**: Saves FFT features and timestamps to a CSV file

3. **ROS Control**:
   - **Script**: `curvature_ros.py`
   - **Description**: Controls the robot using ROS and logs positional data
   - **Purpose**: Moves the robot and collects data for curvature experiments

4. **Merge CSV Files**:
   - **Script**: `csv_merger.py`
   - **Description**: Merges raw CSV files into a unified dataset
   - **Purpose**: Combines audio and robot data for easier analysis

5. **Preprocess Data**:
   - **Script**: `curvature_data_processor.py`
   - **Description**: Normalizes and preprocesses curvature data
   - **Purpose**: Prepares data for machine learning or statistical analysis

6. **Visualize FFT Data**:
   - **Script**: `fft_visualizer.py`
   - **Description**: Visualizes FFT data using boxplots, line plots, and heatmaps
   - **Purpose**: Provides insights into FFT data through visualizations

## Step-by-Step Workflow

### Step 1: Generate Test Signal (Once Only)
- **Script**: `scripts/2000hz_wav_creator.py`
- **Run on**: Either machine
- **Description**: Generates a 300-second WAV file containing test tones from 200 Hz to 2000 Hz (every 200 Hz)
- **Output**: `300.0s.wav`
- **Action**: Copy this WAV file to the audio laptop and prepare it for playback

### Step 2: Start Audio Playback Manually
- **Run on**: Laptop
- **Description**: Manually open and play `300.0s.wav` using any audio player (e.g., VLC)
- **Purpose**: This reference audio will propagate through the sensor

### Step 3: Start Audio Recording & FFT Logging
- **Script**: `scripts/curvature_data_collector.py`
- **Run on**: Laptop
- **Description**:
  - Records audio via microphone while the WAV file is playing
  - Performs sliding FFT (windowed at 10,000 samples, stepped by 1,000)
  - Extracts amplitude features at 200–2000 Hz (every 200 Hz)
  - Saves each row with real-world timestamp and known curvature value
- **Output**: `csv_data/raw/raw_audio_<curvature>.csv`

### Step 4: Start Robot Movement and Clamp Pressing
- **Script**: `scripts/curvature_ros.py`
- **Run on**: PC (ROS machine)
- **Description**:
  - Moves the Franka robot from section `0cm` to `6cm`
  - At each step:
    - Opens gripper, moves down, applies curvature, logs position
    - Records the timestamp and XYZ pose for each clamp press
    - Publishes section labels (useful for ROS-subbed tools)
- **Output**: `csv_data/raw/raw_robot_<curvature>.csv`

### Step 5: Wait Until Robot Finishes Pressing All Sections
- Once the robot finishes the full sequence, you'll have a complete positional CSV from the ROS PC
- **Action**: At this point, stop the audio recording on the laptop to finalize the audio-side FFT CSV

### Step 6: Merge CSVs by Timestamp
- **Script**: `scripts/csv_merger.py`
- **Run on**: Either machine
- **Description**:
  - Loads both `raw_audio_<curvature>.csv` and `raw_robot_<curvature>.csv`
  - Matches each FFT data row to the nearest robot data point using ISO-format timestamps
  - Creates a merged dataset ready for ML training
- **Output**: `csv_data/merged/merged_<curvature>.csv`

## Project Structure

```
curvature_sensor/
├── main.py                # Main entry point with menu interface
├── requirements.txt       # Python dependencies
├── 300.0s.wav            # Example WAV file
├── csv_data/             # Data storage directory
│   ├── merged/           # Combined audio and robot data
│   ├── preprocessed/     # Processed data ready for analysis
│   └── raw/              # Raw collected data
├── scripts/              # Processing modules
│   ├── 2000hz_wav_creator.py
│   ├── curvature_data_collector.py
│   ├── curvature_ros.py
│   ├── csv_merger.py
│   ├── curvature_data_processor.py
│   ├── curvature_fft_utils.py
│   └── fft_visualizer.py
└── src_normal/           # Additional resources
    └── curvature_sensor/ # Duplicate resources
```

## Pipeline Description

The curvature sensing pipeline consists of the following steps:

1. **Data Generation/Collection**:
   - Generate test WAV files OR
   - Collect real audio data during robot-surface interaction

2. **Data Processing**:
   - Extract FFT features from audio data
   - Merge with robot positional data
   - Normalize and preprocess for analysis

3. **Analysis and Visualization**:
   - Generate visualizations of FFT data
   - Analyze correlations between surface curvature and audio features

## Development and Extension

The system is designed to be modular and easily extendable:

1. Create a new script in the `scripts/` directory
2. Implement your functionality with a `main()` function
3. Add a new menu option in `main.py` that imports and calls your script

## Troubleshooting

- **Import errors**: Ensure all scripts are in the correct directory and that Python can find them
- **Audio recording issues**: Check audio device permissions and configurations
- **ROS communication errors**: Verify ROS master is running and network configuration is correct
- **Two-computer synchronization**: Make sure both machines have synchronized clocks for accurate timestamp matching
- **File transfer issues**: If moving files between machines, verify file paths and permissions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[TODO: Add MIT License details]

## Contact

[TODO: Add contact information]



