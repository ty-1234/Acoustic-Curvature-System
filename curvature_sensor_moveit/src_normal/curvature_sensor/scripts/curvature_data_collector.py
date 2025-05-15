"""
Script to collect audio data and extract FFT features for curvature sensing.

This script records audio from the system's microphone, processes it using FFT to extract
frequency-domain features, and saves the results to a CSV file with curvature information.

Author: Bipindra Rai
Date: 2025-04-17
"""

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


def check_microphone():
    """
    Validates that a microphone is available and functioning.
    Displays information about the selected microphone.
    
    Returns:
        bool: True if a valid microphone is detected, False otherwise.
    """
    try:
        # Display microphone information for monitoring
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        device_info = devices[default_input]
        
        logging.info("")
        logging.info("=" * 70)
        logging.info(f"🎤  MICROPHONE: {device_info['name']}")
        logging.info(f"🔊  SAMPLE RATE: {int(device_info['default_samplerate'])} Hz")
        logging.info(f"🆔  DEVICE ID: {default_input}")
        logging.info("=" * 70)
        logging.info("")
        
        # Verify the device has input channels
        if device_info['max_input_channels'] < 1:
            logging.error("❌ Selected device has no input channels.")
            return False
        
        # Optional: Record a short test sample to verify signal
        duration = 0.5  # seconds
        logging.info("🔊 Testing microphone with a short recording...")
        test_recording = sd.rec(
            int(duration * 22050), 
            samplerate=22050, 
            channels=1, 
            blocking=True
        )
        
        # Check if recording contains only silence
        if np.abs(test_recording).max() < 0.01:
            logging.warning("⚠️ Very low audio levels detected. Check microphone connection and volume.")
            
        return True
        
    except Exception as e:
        logging.error(f"❌ Error accessing microphone: {e}")
        return False


def main():
    """
    Main function that handles user input, audio recording, and data processing.
    
    The function performs the following steps:
    1. Verifies microphone availability and functionality
    2. Prompts the user for the radius of curvature of the test object
    3. Sets up audio recording parameters and buffers
    4. Records audio data and processes it using FFT
    5. Saves the processed data to a CSV file
    """
    # ===== Verify Microphone Availability =====
    if not check_microphone():
        logging.error("❌ Cannot proceed without a working microphone.")
        return
    
    # ===== User Input for Radius =====
    while True:
        try:
            radius_value_cm = float(input("Enter the radius of curvature of your test object (in cm): "))
            if radius_value_cm <= 0:
                # Radius must be positive for physical validity
                raise ValueError("Radius must be a positive number.")
            
            # Convert from cm to mm for internal calculations
            radius_value = radius_value_cm * 10  # 1 cm = 10 mm
            break
        except ValueError as e:
            logging.error(f"Invalid input: {e}")

    # Convert radius to curvature (curvature = 1/radius)
    curvature_value = 1 / radius_value
    # Format the curvature value for filename (replace decimal point with underscore)
    curvature_str = str(curvature_value).replace(".", "_")

    # Set up the output directory path relative to the project root
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "csv_data", "raw")
    output_filename = f"raw_audio_{curvature_str}.csv"
    output_path = os.path.join(output_dir, output_filename)

    # Log the input parameters and output path
    logging.info(f"📐 Radius: {radius_value_cm} cm ({radius_value} mm)")
    logging.info(f"📐 Curvature: {curvature_value} mm⁻¹")
    logging.info(f"💾 Output path: {output_path}")

    # ===== Audio Setup =====
    Fs = 22050  # Sample rate in Hz
    L = 10000   # FFT window size (number of samples)
    step_size = 1000  # Step size for sliding window
    # Generate list of target frequencies for FFT analysis (200Hz to 2000Hz in steps of 200Hz)
    target_frequencies = list(range(200, 2001, 200))

    # Initialize a rolling buffer to store audio data
    buffer = deque(maxlen=L + step_size)
    # List to store processed results
    results = []

    def audio_callback(indata, frames, time, status):
        """
        Callback function for audio stream processing.
        
        This function is called by the sounddevice library for each audio block.
        It buffers the audio data, performs FFT analysis when enough data is available,
        and stores the results.
        
        Parameters:
        -----------
        indata : ndarray
            Input audio data as a numpy array
        frames : int
            Number of frames in the current block
        time : CData
            Timestamp information
        status : CallbackFlags
            Status flags indicating stream status
        """
        try:
            # Check for any issues with the audio stream
            if status:
                logging.warning(f"⚠️ Audio stream status: {status}")
            
            # Add new audio data to the buffer
            buffer.extend(indata[:, 0])

            # Process data if enough samples are available
            if len(buffer) >= L:
                # Extract a window of audio data for FFT analysis
                window_data = np.array(buffer)[:L]
                
                # Extract FFT features for the target frequencies
                FFT_values = extract_fft_features(
                    signal=window_data,
                    sample_rate=Fs,
                    window_size=L,
                    target_freqs=target_frequencies
                )
                
                # Record timestamp for the current sample
                timestamp = pd.Timestamp.now().isoformat()
                
                # Combine FFT values with curvature and timestamp
                new_row = FFT_values + [curvature_value, timestamp]
                results.append(new_row)

                # Provide periodic feedback on data collection
                if len(results) % 10 == 0:
                    logging.info(f"🔢 Samples collected: {len(results)}")

                # Remove processed samples from the buffer to advance the window
                for _ in range(step_size):
                    buffer.popleft()
        except Exception as e:
            # Catch and log any errors during processing
            logging.error(f"Error in audio callback: {e}")

    # Log audio recording start
    logging.info("🎙️ Starting audio recording... Press Ctrl+C to stop.")

    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Start the audio input stream with the callback function
        with sd.InputStream(callback=audio_callback, samplerate=Fs, channels=1, blocksize=step_size):
            # Keep the script running until interrupted
            while True:
                time.sleep(0.1)  # Prevent CPU burn by sleeping briefly

    except KeyboardInterrupt:
        # Handle manual termination (Ctrl+C)
        logging.info("\n🛑 Recording manually stopped by user.")
    except Exception as e:
        # Handle any other unexpected errors
        logging.error(f"Unexpected error: {e}")
    finally:
        # ===== Save the collected data =====
        try:
            if results:
                # Create column names for the DataFrame
                columns = [f"FFT_{f}Hz" for f in target_frequencies] + ["Curvature", "Timestamp"]
                # Create a DataFrame from the collected results
                df = pd.DataFrame(results, columns=columns)
                # Save to CSV file
                df.to_csv(output_path, index=False)
                logging.info(f"\n✅ Done. CSV saved to {output_path}")
            else:
                # Warn if no data was collected
                logging.warning("No data was recorded. CSV file not created.")
        except Exception as e:
            # Handle errors during saving
            logging.error(f"❌ Error saving CSV: {e}")


if __name__ == "__main__":
    main()
