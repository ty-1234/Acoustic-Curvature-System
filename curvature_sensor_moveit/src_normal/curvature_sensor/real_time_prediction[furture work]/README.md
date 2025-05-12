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
- **Health Monitoring**: Continuously monitors system components and handles failures gracefully
- **Fault Tolerance**: Recovers from transient errors and performs controlled shutdown if critical failures occur

## System Requirements

- Python 3.10 or higher
- PortAudio library (for sound device access)
- Audio input/output capability

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
   - Load the trained model and feature scaler
   - Check available audio devices
   - Start the prediction thread
   - Initialize the frequency generator (audio output)
   - Start the feature extractor (audio capture and processing)
   - Launch the GUI interface
   - Begin continuous health monitoring
   - Make predictions every 100ms

### Monitoring and Data Collection

The GUI provides real-time information:

- **Microphone Information**: Confirms which microphone is being used for data capture
- **System Status**: Shows whether the system is active or inactive
- **Curvature Value**: Displays the current measured curvature in mm⁻¹
- **Position Value**: Shows the current detected position in cm

All prediction data is automatically logged to CSV files in the prediction_logs directory with timestamps for later analysis. Each session creates a new log file with a date-time stamp in the filename.

### System Shutdown

The system can be terminated in several ways:
- Closing the GUI window
- Pressing Ctrl+C in the terminal
- Automatic shutdown if critical component failures are detected

In all cases, a controlled shutdown sequence ensures all components are properly stopped, threads are terminated, and log files are properly closed.

### Troubleshooting

If the system shows "INACTIVE" status:
- Check that the sensing module is properly positioned
- Verify audio connections are secure
- Ensure ambient noise levels are not too high

For optimal results, use the system in a relatively quiet environment with consistent lighting conditions.

## System Architecture

The system consists of several integrated components coordinated by a central orchestrator:

1. **Orchestrator** (`orchestrator.py`): The central controller that:
   - Coordinates all system components
   - Manages the startup and shutdown sequence
   - Monitors component health
   - Handles errors and system recovery
   - Maintains the 100ms prediction cycle

2. **Frequency Generator** (`rt_frequency_gen.py`): Produces consistent multi-tone signals for sensing.

3. **Feature Extractor** (`rt_feature_extraction.py`): Captures audio and performs FFT analysis to extract frequency domain features.

4. **Prediction Module** (`rt_model.py`): Uses a pre-trained machine learning model to predict both curvature and position values.

5. **Data Logger** (`rt_logger.py`): Records predictions to CSV files with timestamps.

6. **Visualization** (`rt_gui.py`): Displays curvature and position predictions in real-time through a user-friendly interface.

The system employs a multi-threaded architecture with event-based synchronization to maintain real-time performance. Each component runs in a dedicated thread, coordinated by the orchestrator to ensure proper timing and data flow.

## Project Structure

```
real_time_prediction/
├── orchestrator.py         # Main entry point and system coordinator
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
- `threading` for parallel processing and component coordination
- `joblib` for model loading

## Future Work

- Enhance noise robustness through adaptive filtering
- Improve position tracking accuracy
- Implement real-time model retraining capabilities
- Integrate with robotic systems

## Author

**Bipindra Rai**  
Final Year Engineering Project  
Class of 2025  

---

© 2025 Bipindra Rai. All Rights Reserved.