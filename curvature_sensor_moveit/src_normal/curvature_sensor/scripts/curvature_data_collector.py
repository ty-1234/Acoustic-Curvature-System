import numpy as np
import pandas as pd
import sounddevice as sd
import os
import sys
from datetime import datetime

# Add the parent directory to sys.path for module resolution
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from curvature_fft_utils import extract_fft_features


def main():
    # ===== User Input for Curvature Value =====
    curvature_value = float(input("Enter the curvature value of your test object (in mm⁻¹): "))
    curvature_str = str(curvature_value).replace(".", "_")

    # Updated directory to be relative to the script's location
    output_dir = os.path.join(os.path.dirname(__file__), "..", "csv_data", "raw")
    output_filename = f"raw_audio_{curvature_str}.csv"
    output_path = os.path.join(output_dir, output_filename)

    print(f"\n📐 Curvature: {curvature_value} mm⁻¹")
    print(f"💾 Output path: {output_path}\n")

    # ===== Audio Setup =====
    Fs = 22050
    L = 10000
    step_size = 1000
    target_frequencies = list(range(200, 2001, 200))

    results = []
    print("🎙️ Starting audio recording... Press Ctrl+C to stop.")

    try:
        start_time = pd.Timestamp.now()

        while True:  # Infinite loop until manually stopped
            try:
                # Attempt to record audio
                audio_data = sd.rec(L, samplerate=Fs, channels=1, blocking=True)
                audio_data = audio_data.flatten()
            except Exception as e:
                print(f"❌ Microphone not detected or not working: {e}")
                return  # Exit the program if the microphone is not working

            start_idx = 0
            while start_idx + L <= len(audio_data):
                window_data = audio_data[start_idx:start_idx + L]

                FFT_values = extract_fft_features(
                    signal=window_data,
                    sample_rate=Fs,
                    window_size=L,
                    target_freqs=target_frequencies
                )

                timestamp = pd.Timestamp.now().isoformat()
                new_row = FFT_values + [curvature_value, timestamp]
                results.append(new_row)

                start_idx += step_size

    except KeyboardInterrupt:
        print("\n🛑 Recording manually stopped by user.")

    # ===== Save =====
    try:
        os.makedirs(output_dir, exist_ok=True)
        columns = [f"FFT_{f}Hz" for f in target_frequencies] + ["Curvature", "Timestamp"]
        df = pd.DataFrame(results, columns=columns)
        df.to_csv(output_path, index=False)
        print(f"\n✅ Done. CSV saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving CSV: {e}")


if __name__ == "__main__":
    main()
