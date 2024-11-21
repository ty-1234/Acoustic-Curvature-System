import numpy as np
import pandas as pd
import sounddevice as sd

dateTimeStr = pd.Timestamp.now().strftime('%Y-%m-%d %H-%M-%S')
excelFilename = f'Noise_0.0dB_Load_20_{dateTimeStr}.xlsx'

timestamps = []
filtered_forces = []

Fs = 22050 * 2
L = 10000
step_size = 10000

A, H, Q, R = 1, 1, 0.01, 0.6
current_state, current_covariance = 0, 1

def calculate_force():
    return np.random.random()  # Placeholder for Force

start_time = pd.Timestamp.now()
while (pd.Timestamp.now() - start_time).total_seconds() < 20:
    try:
        audio_data = sd.rec(L, samplerate=Fs, channels=1, blocking=True).flatten()
    except Exception as e:
        print(f"Error recording audio: {e}")
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

        Force = calculate_force()
        predicted_state = A * current_state
        predicted_covariance = A * current_covariance * A + Q
        kalman_gain = predicted_covariance * H / (H * predicted_covariance * H + R)
        current_state = predicted_state + kalman_gain * (Force - H * predicted_state)
        current_covariance = (1 - kalman_gain * H) * predicted_covariance
        filtered_Force = current_state

        timestamps.append(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        filtered_forces.append(filtered_Force)

        start_idx += step_size
        end_idx += step_size

data = pd.DataFrame({'Timestamp': timestamps, 'Filtered Force': filtered_forces})
data.to_excel(excelFilename, index=False)
print(f"Data written to {excelFilename}")
