# Curvature Sensing System

## Overview

This project implements a complete pipeline for curvature sensing using audio-based FFT features and robot-assisted data collection. The system includes scripts for data collection, synchronization, preprocessing, and machine learning model training and evaluation.

---

## Folder Structure

```
curvature_sensor/
├── csv_data/                  # All experimental data (raw, merged, preprocessed)
│   ├── raw/                   # Raw FFT and robot CSVs
│   ├── merged/                # Timestamp-aligned sensor and robot data
│   ├── preprocessed/          # Feature-engineered, ML-ready datasets
│   └── README.md              # Data format and flow documentation
├── machine learning models/   # ML model training scripts and outputs
│   ├── lightGBM_mode.py
│   ├── mlp_regression.py
│   ├── svr_regression.py
│   ├── gpr.py
│   ├── models/                # Saved models and metrics
│   └── README.md
├── scripts/                   # Data collection, processing, and utility scripts
│   ├── main.py
│   ├── curvature_data_collector.py
│   ├── curvature_ros.py
│   ├── new_merger.py
│   ├── curvature_fft_utils.py
│   ├── frequency_gen.py
│   ├── trimmer.py
│   └── requirements.txt
├── requirements.txt           # Top-level Python dependencies
└── README.md                  # Project documentation (this file)
```

---

## Scripts

- **[main.py](scripts/main.py)**: Menu-driven interface to access all system functions.
- **[curvature_data_collector.py](scripts/curvature_data_collector.py)**: Records audio and extracts FFT features in real-time.
- **[curvature_ros.py](scripts/curvature_ros.py)**: Controls the Franka robot arm for systematic data collection.
- **[new_merger.py](scripts/csv_sync.py)**: Synchronizes and merges audio and robot data by timestamp.
- **[curvature_fft_utils.py](scripts/curvature_fft_utils.py)**: Utility functions for FFT processing and feature extraction.
- **[frequency_gen.py](scripts/frequency_gen.py)**: Generates multi-tone signals for acoustic excitation.
---

## Machine Learning Models

All model scripts are in [machine learning models/](machine%20learning%20models/):

- **[lightGBM_mode.py](machine%20learning%20models/lightGBM_mode.py)**: Multi-output regression using LightGBM.
- **[mlp_regression.py](machine%20learning%20models/mlp_regression.py)**: Multi-layer perceptron regression for position and curvature.
- **[svr_regression.py](machine%20learning%20models/svr_regression.py)**: Support Vector Regression for multi-output prediction.
- **[gpr.py](machine%20learning%20models/gpr.py)**: Gaussian Process Regression for curvature and position.
- **[models/](machine%20learning%20models/models/)**: Stores trained models and output metrics.

---

## CSV Data

See [csv_data/README.md](csv_data/README.md) for full details.

- **raw/**: Contains raw sensor and robot data (`raw_audio_*.csv`, `raw_robot_*.csv`).
- **merged/**: Contains merged files aligning audio and robot data by timestamp.

---

## Usage

### Menu Interface

```sh
python scripts/main.py
```

### Run Individual Scripts

- Collect FFT data:
  ```sh
  python scripts/curvature_data_collector.py
  ```
- Merge CSV files:
  ```sh
  python scripts/csv_sync.py
  ```
- Train a model (example for LightGBM):
  ```sh
  python "machine learning models/lightGBM_mode.py"
  ```

---

## Requirements

Install dependencies with:

```sh
pip install -r requirements.txt
```

---


