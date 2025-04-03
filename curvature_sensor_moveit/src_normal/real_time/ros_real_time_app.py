import rospy
import numpy as np
import pandas as pd
import sounddevice as sd
from std_msgs.msg import Float64MultiArray

rospy.init_node('fft_publisher', anonymous=True)
fft_pub = rospy.Publisher('/fft_results', Float64MultiArray, queue_size=10)

timestamps = []
filtered_forces = []
locations = []

inputVarNames = ['FFT300', 'FFT500', 'FFT700', 'FFT900']

Fs = 22050 * 2  # Reduced sample rate
L = 10000  # Window length
step_size = 10000  # Step size for overlapping windows

A, H, Q, R = 1, 1, 0.01, 0.6
current_state, current_covariance = 0, 1

def get_location_data():
    return np.random.random()  # Placeholder for location data

def calculate_force():
    return np.random.random()  # Placeholder for Force value

start_time = pd.Timestamp.now()
while (pd.Timestamp.now() - start_time).total_seconds() < 20:
    try:
        audio_data = sd.rec(L, samplerate=Fs, channels=1, blocking=True).flatten()
    except Exception as e:
        rospy.logerr(f"Audio recording failed: {e}")
        continue

    start_idx, end_idx = 0, L
    while end_idx <= len(audio_data):
        window_data = audio_data[start_idx:end_idx]
        Fn = Fs / 2
        FTy = np.fft.fft(window_data, L) / L
        Fv = np.linspace(0, 1, int(L / 2) + 1) * Fn
        x = np.abs(FTy[:int(L / 2) + 1]) * 2

        M = np.column_stack((Fv[:, np.newaxis], x))
        target_frequencies = [300, 500, 700, 900]
        FFT_values = [M[np.argmin(np.abs(M[:, 0] - f)), 1] for f in target_frequencies]
        FFT300, FFT500, FFT700, FFT900 = FFT_values

        Force = calculate_force()
        predicted_state = A * current_state
        predicted_covariance = A * current_covariance * A + Q
        kalman_gain = predicted_covariance * H / (H * predicted_covariance * H + R)
        current_state = predicted_state + kalman_gain * (Force - H * predicted_state)
        current_covariance = (1 - kalman_gain * H) * predicted_covariance
        filtered_Force = current_state

        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        Location = get_location_data()
        timestamps.append(timestamp)
        filtered_forces.append(filtered_Force)
        locations.append(Location)

        fft_msg = Float64MultiArray(data=[timestamp, filtered_Force, Location])
        try:
            fft_pub.publish(fft_msg)
        except rospy.ROSException as e:
            rospy.logerr(f"Publishing failed: {e}")

        start_idx += step_size
        end_idx += step_size

data = pd.DataFrame({'Timestamp': timestamps, 'Kalman force': filtered_forces, 'OrForce': locations})
# Optional: Write data to an Excel file if needed.
