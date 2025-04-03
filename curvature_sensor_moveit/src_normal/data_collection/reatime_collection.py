import rospy
import numpy as np
import pandas as pd
import sounddevice as sd
from std_msgs.msg import Float64MultiArray

# Initialize ROS node
rospy.init_node('fft_publisher', anonymous=True)

# Publisher for the FFT results
fft_pub = rospy.Publisher('/fft_results', Float64MultiArray, queue_size=10)

# Prompt the user to enter an integer
user_integer = int(input('Enter loadcell value: '))

Fs = 22050  # Reduced sample rate
L = 10000  # Window length
step_size = 1000  # Increased step size for overlapping windows

# Empty list to store the results
results = []

# Record the start time
start_time = pd.Timestamp.now()
while (pd.Timestamp.now() - start_time).total_seconds() < 50:
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

        M = np.column_stack((Fv, x))

        # Extract the FFT values at specific frequencies
        target_frequencies = [300, 500, 700, 900]
        FFT_values = np.zeros(len(target_frequencies))

        for i, target_frequency in enumerate(target_frequencies):
            index = np.argmin(np.abs(M[:, 0] - target_frequency))
            FFT_values[i] = M[index, 1]

        FFT300, FFT500, FFT700, FFT900 = FFT_values

        # Display the FFT values
        print(f"FFT300: {FFT300}")
        print(f"FFT500: {FFT500}")
        print(f"FFT700: {FFT700}")
        print(f"FFT900: {FFT900}")

        # Create a new row of results for the current iteration
        current_time = pd.Timestamp.now()
        new_row = [user_integer, FFT300, FFT500, FFT700, FFT900, current_time]
        results.append(new_row)

        # Publish the results via ROS
        fft_msg = Float64MultiArray(data=[user_integer, FFT300, FFT500, FFT700, FFT900, current_time.timestamp()])
        fft_pub.publish(fft_msg)

        # Move to the next window
        start_idx += step_size
        end_idx += step_size

# Convert the list of results to a DataFrame
results_df = pd.DataFrame(results, columns=['Force', 'FFT300', 'FFT500', 'FFT700', 'FFT900', 'Time'])

# Write the results DataFrame to a CSV file
# Using mode='a' to append if the file already exists; if you want to overwrite, use mode='w'
results_df.to_csv('C.csv', mode='a', index=False)
