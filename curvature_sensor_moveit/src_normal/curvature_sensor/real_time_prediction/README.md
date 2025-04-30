# Real-Time Curvature and Position Sensing System

![Curvature Sensing Banner](https://via.placeholder.com/800x200?text=Curvature+and+Position+Sensing+System)

## Overview

The Real-Time Curvature and Position Sensing System is a novel approach to measuring both curvature and position using acoustic frequency analysis. This system uses multi-tone audio signals and machine learning to detect subtle changes in curvature and precisely track position, providing real-time feedback with high precision.

This project was developed as a final year engineering project to demonstrate the application of signal processing, machine learning, and real-time systems in creating innovative sensors for robotics and human-computer interaction.

## Features

- **Multi-tone Frequency Generation**: Outputs precise reference frequencies (200Hz-2000Hz)
- **Real-time FFT Analysis**: Extracts frequency domain features from microphone input
- **Machine Learning Prediction**: Uses a trained model to simultaneously predict curvature and position values
- **Live Visualization**: Shows curvature and position predictions through an intuitive GUI interface
- **Data Logging**: Records all predictions for later analysis and validation

## System Requirements

- Python 3.10 or higher
- PortAudio library (for sound device access)
- Audio input/output capability

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/username/curvature-sensor.git
    cd curvature-sensor
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure model files are present in the model directory:
    - `extratrees_optuna_model.pkl` - The trained prediction model
    - `feature_scaler.pkl` - Feature scaling parameters

## Usage

### Physical Setup

Before running the system, ensure your hardware is properly connected:

1. **Connect Speakers/Headphones**: Ensure your computer's audio output device is connected and functioning properly. This will output the multi-tone frequency signals.

2. **Connect Microphone**: Ensure an external microphone is properly connected to your computer. This will capture the acoustic signals after they have passed through the sensing module.

3. **Position the Sensing Module**: Place the physical curvature sensing module between the speakers and microphone, ensuring the acoustic channel is properly aligned.

4. **Check Audio Settings**: Verify your system recognizes both the speakers and microphone in your operating system's audio settings.

### Running the System

After completing the physical setup:

1. Open a terminal/command prompt in the project directory
2. Run the orchestrator script:

```bash
python orchestrator.py
```

3. The system will:
   - Generate multi-tone frequencies through your speakers
   - Capture the modified signal through your microphone
   - Process the signal using FFT analysis
   - Make real-time predictions of curvature and position
   - Display results in the GUI

### Monitoring and Data Collection

The GUI provides real-time information:

- **Microphone Information**: Confirms which microphone is being used for data capture
- **System Status**: Shows whether the system is active or inactive
- **Curvature Value**: Displays the current measured curvature in mm⁻¹
- **Position Value**: Shows the current detected position in cm

All prediction data is automatically logged to CSV files in the prediction_logs directory with timestamps for later analysis. Each session creates a new log file with a date-time stamp in the filename.

### Troubleshooting

If the system shows "INACTIVE" status:
- Check that the sensing module is properly positioned
- Verify audio connections are secure
- Ensure ambient noise levels are not too high

For optimal results, use the system in a relatively quiet environment with consistent lighting conditions.

## System Architecture

The system consists of several integrated components:

1. **Frequency Generator** (`rt_frequency_gen.py`): Produces consistent multi-tone signals for sensing.

2. **Feature Extractor** (`rt_feature_extraction.py`): Captures audio and performs FFT analysis to extract frequency domain features.

3. **Prediction Model** (`rt_model.py`): Uses a pre-trained machine learning model to predict both curvature and position values.

4. **Data Logger** (`rt_logger.py`): Records predictions to CSV files with timestamps.

5. **Visualization** (`rt_gui.py`): Displays curvature and position predictions in real-time through a user-friendly interface.

6. **Orchestrator** (`orchestrator.py`): Coordinates all components and ensures proper system operation.

## Project Structure

```
real_time_prediction/
├── orchestrator.py         # Main entry point
├── rt_feature_extraction.py # Feature extraction from audio
├── rt_frequency_gen.py     # Multi-tone generator
├── rt_gui.py               # Real-time visualization
├── rt_logger.py            # Data logging functionality
├── rt_model.py             # ML model prediction
├── requirements.txt        # Dependencies
├── README.md               # This file
├── model/                  # Trained models and scalers
├── prediction_logs/        # Logged prediction data
└── utils/                  # Utilities
     └── rt_fft_utils.py     # FFT processing utilities
```

## Development

The project uses several key libraries:
- `numpy` and `pandas` for data processing
- `scikit-learn` for ML model integration
- `sounddevice` for audio input/output
- `tkinter` for GUI implementation

## Future Work

- Enhance noise robustness through adaptive filtering
- Improve position tracking accuracy
- Implement real-time model retraining capabilities
- Integrate with robotic systems

## Author

**Bipindra Rai**  
Final Year Engineering Project  
Class of 2025  

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

© 2025 Bipindra Rai. All Rights Reserved.