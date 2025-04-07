import numpy as np
import pandas as pd
import sounddevice as sd
import os
import sys
import time
import logging
from datetime import datetime
from collections import deque

# Add the parent directory to sys.path for module resolution
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from curvature_fft_utils import extract_fft_features

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    # ===== User Input for Radius =====
    while True:
        try:
            radius_value = float(input("Enter the radius of curvature of your test object (in mm): "))
            if radius_value <= 0:
                raise ValueError("Radius must be a positive number.")
            break
        except ValueError as e:
            logging.error(f"Invalid input: {e}")

    curvature_value = 1 / radius_value  # Convert radius to curvature
    curvature_str = str(curvature_value).replace(".", "_")

    # Updated directory to be relative to the script's location
    output_dir = os.path.join(os.path.dirname(__file__), "..", "csv_data", "raw")
    output_filename = f"raw_audio_{curvature_str}.csv"
    output_path = os.path.join(output_dir, output_filename)

    logging.info(f"📐 Radius: {radius_value} mm")
    logging.info(f"📐 Curvature: {curvature_value} mm⁻¹")
    logging.info(f"💾 Output path: {output_path}")

    # ===== Audio Setup =====
    Fs = 22050  # Sample rate
    L = 10000  # FFT window size
    step_size = 1000  # Step size for sliding window
    target_frequencies = list(range(200, 2001, 200))  # Target FFT frequencies

    # Rolling buffer to store audio data
    buffer = deque(maxlen=L + step_size)
    results = []

    def audio_callback(indata, frames, time, status):
        try:
            if status:
                logging.warning(f"⚠️ Audio stream status: {status}")
            buffer.extend(indata[:, 0])  # Add audio data to the buffer

            # Process data if enough samples are available
            if len(buffer) >= L:
                window_data = np.array(buffer)[:L]  # Extract the FFT window
                FFT_values = extract_fft_features(
                    signal=window_data,
                    sample_rate=Fs,
                    window_size=L,
                    target_freqs=target_frequencies
                )
                timestamp = pd.Timestamp.now().isoformat()
                new_row = FFT_values + [curvature_value, timestamp]
                results.append(new_row)

                # Provide feedback every 10 samples
                if len(results) % 10 == 0:
                    logging.info(f"🔢 Samples collected: {len(results)}")

                # Remove processed samples from the buffer
                for _ in range(step_size):
                    buffer.popleft()
        except Exception as e:
            logging.error(f"Error in audio callback: {e}")

    logging.info("🎙️ Starting audio recording... Press Ctrl+C to stop.")

    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Start the audio stream
        with sd.InputStream(callback=audio_callback, samplerate=Fs, channels=1, blocksize=step_size):
            while True:
                time.sleep(0.1)  # Prevent CPU burn by sleeping briefly

    except KeyboardInterrupt:
        logging.info("\n🛑 Recording manually stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        # ===== Save =====
        try:
            if results:
                columns = [f"FFT_{f}Hz" for f in target_frequencies] + ["Curvature", "Timestamp"]
                df = pd.DataFrame(results, columns=columns)
                df.to_csv(output_path, index=False)
                logging.info(f"\n✅ Done. CSV saved to {output_path}")
            else:
                logging.warning("No data was recorded. CSV file not created.")
        except Exception as e:
            logging.error(f"❌ Error saving CSV: {e}")


if __name__ == "__main__":
    main()
