import numpy as np
import pandas as pd
import sounddevice as sd
import os
import rospy
from std_msgs.msg import String

# 📦 Import FFT utility function
from scripts.curvature_fft_utils import extract_fft_features

def main():
    # ================================
    # CONFIGURABLE INPUTS
    # ================================
    curvature_value = float(input("Enter known curvature value for this test object (e.g., 0.01818): "))
    curvature_str = str(curvature_value).replace(".", "_")

    output_dir = 'csv_data/raw'
    output_filename = f"raw_{curvature_str}.csv"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\n📐 Curvature set to {curvature_value} mm⁻¹")
    print(f"📁 Saving to: {output_path}\n")

    Fs = 22050
    L = 10000
    step_size = 1000
    record_seconds = 200
    target_frequencies = list(range(200, 2001, 200))

    # Global ROS section label
    current_section_label = "Unknown"

    def section_callback(msg):
        nonlocal current_section_label
        current_section_label = msg.data

    rospy.init_node('curvature_logger', anonymous=True)
    rospy.Subscriber('/curving_section', String, section_callback)

    results = []
    print("🎙️ Beginning audio recording...")

    start_time = pd.Timestamp.now()

    while (pd.Timestamp.now() - start_time).total_seconds() < record_seconds:
        audio_data = sd.rec(L, samplerate=Fs, channels=1, blocking=True)
        audio_data = audio_data.flatten()

        FFT_values = extract_fft_features(
            signal=audio_data,
            sample_rate=Fs,
            window_size=L,
            target_freqs=target_frequencies
        )

        timestamp = pd.Timestamp.now()
        new_row = [current_section_label] + FFT_values + ["curved", curvature_value, timestamp]
        results.append(new_row)

    os.makedirs(output_dir, exist_ok=True)

    columns = (
        ["Position_cm"] +
        [f"FFT{f}" for f in target_frequencies] +
        ["Curving_State", "Curvature", "Timestamp"]
    )

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Recording complete.")
    print(f"💾 Data saved to: {output_path}")

if __name__ == "__main__":
    main()
