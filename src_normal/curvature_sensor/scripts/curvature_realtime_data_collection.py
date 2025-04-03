#!/usr/bin/env python3
import numpy as np
import pandas as pd
import sounddevice as sd
import os
import rospy
from std_msgs.msg import String

# 📦 Import FFT utility function
from scripts.curvature_fft_utils import extract_fft_features

# ================================
# CONFIGURABLE INPUTS
# ================================

curvature_value = float(input("Enter known curvature value for this test object (e.g., 0.01818): "))
curvature_str = str(curvature_value).replace(".", "_")

# Save path
output_dir = 'csv_data/raw'
output_filename = f"raw_{curvature_str}.csv"
output_path = os.path.join(output_dir, output_filename)

print(f"\n📐 Curvature set to {curvature_value} mm⁻¹")
print(f"📁 Saving to: {output_path}\n")

# Audio parameters
Fs = 22050          # Sample rate
L = 10000           # FFT window length
step_size = 1000    # Overlap
record_seconds = 200  # Total time

# FFT frequencies (200–2000 Hz)
target_frequencies = list(range(200, 2001, 200))

# ================================
# STATE MANAGEMENT
# ================================

curving_state = "rest"  # Default
current_section_label = "Unknown"  # Default section label

# ROS callback to update the current section label
def section_callback(msg):
    global current_section_label
    current_section_label = msg.data

# ================================
# RECORDING LOOP
# ================================

rospy.init_node('curvature_logger', anonymous=True)
rospy.Subscriber('/curving_section', String, section_callback)

results = []
print("🎙️ Beginning audio recording...")

start_time = pd.Timestamp.now()

while (pd.Timestamp.now() - start_time).total_seconds() < record_seconds:
    audio_data = sd.rec(L, samplerate=Fs, channels=1, blocking=True)
    audio_data = audio_data.flatten()

    start_idx = 0
    while start_idx + L <= len(audio_data):
        window_data = audio_data[start_idx:start_idx + L]

        # FFT feature extraction
        FFT_values = extract_fft_features(
            signal=window_data,
            sample_rate=Fs,
            window_size=L,
            target_freqs=target_frequencies
        )

        timestamp = pd.Timestamp.now()

        # If using pose data:
        if latest_pose:
            posX = latest_pose.pose.position.x
            posY = latest_pose.pose.position.y
            posZ = latest_pose.pose.position.z
        else:
            posX = posY = posZ = None  # Placeholder

        new_row = [current_section_label] + FFT_values + [curving_state, curvature_value, posX, posY, posZ, timestamp]
        results.append(new_row)

        start_idx += step_size

# ================================
# SAVE RESULTS
# ================================

os.makedirs(output_dir, exist_ok=True)

columns = (
    ["Position_cm"] +
    [f"FFT{f}" for f in target_frequencies] +
    ["Curving_State", "Curvature", "PosX", "PosY", "PosZ", "Timestamp"]
)

df = pd.DataFrame(results, columns=columns)
df.to_csv(output_path, index=False)

print(f"\n✅ Recording complete.")
print(f"💾 Data saved to: {output_path}")
