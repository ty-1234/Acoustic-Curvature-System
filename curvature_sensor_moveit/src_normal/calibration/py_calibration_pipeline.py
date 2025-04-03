import numpy as np
import pandas as pd
import sounddevice as sd
from std_msgs.msg import Float64MultiArray
import rospy

# Generate a filename with current date and time
dateTimeStr = pd.Timestamp.now().strftime('%Y-%m-%d %H-%M-%S')
excelFilename = f'Noise_0.0dB_Load_20_{dateTimeStr}.xlsx'

# Initialize lists to store data
timestamps = []
filtered_forces = []
locations = []
fft_data = []

inputVarNames = ['FFT300', 'FFT500', 'FFT700', 'FFT900']

Fs = 22050 * 2  # Reduced sample rate
L = 10000  # Window length
step_size = 1000  # Increased step size for overlapping windows

# Kalman filter parameters
A = 1
H = 1
Q = 0.01
R = 0.6
initial_state = 0
initial_covariance = 1

# Initialize Kalman filter state and covariance
current_state = initial_state
current_covariance = initial_covariance

# Initialize ROS node
rospy.init_node('fft_publisher', anonymous=True)

# Publisher for the FFT results
fft_pub = rospy.Publisher('/fft_results', Float64MultiArray, queue_size=10)

# Calibration matrix (example values, replace with actual calibration data)
calibration_matrix = np.array([1.0, 1.0, 1.0, 1.0])

# Function to apply calibration to FFT values
def apply_calibration(fft_values, calibration_matrix):
    return fft_values * calibration_matrix

# Record the start time
start_time = pd.Timestamp.now()
while (pd.Timestamp.now() - start_time).total_seconds() < 20:
    audio_data = sd.rec(L, samplerate=Fs, channels=1, blocking=True)
    audio_data = audio_data.flatten()  # Flatten the audio data to a 1D array

    # Perform overlapping FFT
    start_idx = 0
    end_idx = L
    while end_idx <= len(audio_data):
        window_data = audio_data[start_idx:end_idx]

        # Perform FFT on the windowed data
        Fn = Fs / 2
        FTy = np.fft.fft(window_data, L) / L
        Fv = np.linspace(0, 1, int(L / 2) + 1) * Fn
        x = np.abs(FTy[:int(L / 2) + 1]) * 2

        a = Fv[:, np.newaxis]
        M = np.column_stack((a, x))

        # Extract the FFT values at specific frequencies
        target_frequencies = [300, 500, 700, 900]
        FFT_values = np.zeros(len(target_frequencies))

        for i, target_frequency in enumerate(target_frequencies):
            index = np.argmin(np.abs(M[:, 0] - target_frequency))
            FFT_values[i] = M[index, 1]

        # Apply calibration to FFT values
        calibrated_FFT_values = apply_calibration(FFT_values, calibration_matrix)

        FFT300, FFT500, FFT700, FFT900 = calibrated_FFT_values

        # Calculate the filtered Force value using Kalman filter
        measurement = 10  # Replace this with your actual Force value
        predicted_state = A * current_state
        predicted_covariance = A * current_covariance * A + Q
        kalman_gain = predicted_covariance * H / (H * predicted_covariance * H + R)
        current_state = predicted_state + kalman_gain * (measurement - H * predicted_state)
        current_covariance = (1 - kalman_gain * H) * predicted_covariance
        filtered_Force = current_state

        # Display the filtered Force value
        print(filtered_Force)

        # Append data to lists
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        timestamps.append(timestamp)
        filtered_forces.append(filtered_Force)
        fft_data.append(calibrated_FFT_values.tolist())
        # locations.append(Location)

        # Create and publish ROS message
        fft_msg = Float64MultiArray(data=[FFT300, FFT500, FFT700, FFT900, filtered_Force])
        fft_pub.publish(fft_msg)

        # Move to the next window
        start_idx += step_size
        end_idx += step_size

# Create a DataFrame from the collected data
data = pd.DataFrame({'Timestamp': timestamps, 'Kalman force': filtered_forces})

# Add FFT data to the DataFrame
fft_df = pd.DataFrame(fft_data, columns=inputVarNames)
data = pd.concat([data, fft_df], axis=1)

# Write the DataFrame to an Excel file
data.to_excel(excelFilename, sheet_name='ForceData', index=False)
print(f'Data written to {excelFilename}')
