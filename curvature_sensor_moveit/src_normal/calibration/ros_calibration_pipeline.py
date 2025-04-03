import rospy
import numpy as np
import pandas as pd
import sounddevice as sd
from std_msgs.msg import Float64MultiArray

rospy.init_node('fft_publisher', anonymous=True)
fft_pub = rospy.Publisher('/fft_results', Float64MultiArray, queue_size=10)

try:
    user_integer = int(input("Enter loadcell value: "))
except ValueError:
    rospy.logerr("Invalid loadcell value. Exiting.")
    exit()

Fs, L, step_size = 22050, 10000, 1000

start_time = pd.Timestamp.now()
while (pd.Timestamp.now() - start_time).total_seconds() < 50:
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

        current_time = pd.Timestamp.now()
        fft_msg = Float64MultiArray(data=[user_integer] + FFT_values + [current_time.timestamp()])
        try:
            fft_pub.publish(fft_msg)
        except rospy.ROSException as e:
            rospy.logerr(f"Publishing failed: {e}")

        start_idx += step_size
        end_idx += step_size
