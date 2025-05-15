"""
Script to collect audio data and extract FFT features for curvature sensing.

This script records audio from the system's microphone, processes it using FFT to extract
frequency-domain features, and saves the results to a CSV file with curvature information.

MODIFIED to target frequencies from 100 Hz to 16,000 Hz and use an appropriate sample rate.

Author: Bipindra Rai
Date: 2025-04-17 (Modified: 2025-05-15)
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
# This assumes curvature_fft_utils.py is in a directory one level up from this script's directory
# Adjust if your project structure is different.
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from curvature_fft_utils import extract_fft_features # Assumes this utility handles the FFT calculation
except ImportError:
    logging.error("❌ Failed to import extract_fft_features from curvature_fft_utils.")
    logging.error("Ensure curvature_fft_utils.py is in the correct path (e.g., parent directory).")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def check_microphone(expected_sample_rate=44100):
    """
    Validates that a microphone is available and functioning at the expected sample rate.
    Displays information about the selected microphone and performs a brief test recording.
    
    Parameters:
    -----------
    expected_sample_rate : int, optional
        The sample rate (in Hz) the script intends to use for recording. 
        Defaults to 44100.
        
    Returns:
        bool: True if a valid microphone is detected and passes a basic test, False otherwise.
    """
    try:
        devices = sd.query_devices()
        default_input_device_index = sd.default.device[0]
        
        if default_input_device_index == -1:
            logging.error("❌ No default input device found. Please configure your system's audio input.")
            logging.info("Available input devices:")
            found_input = False
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    logging.info(f"  ID {i}: {device['name']} (Max Input Channels: {device['max_input_channels']}, Default SR: {device['default_samplerate']})")
                    found_input = True
            if not found_input:
                logging.info("  No input devices found with input channels.")
            return False

        device_info = devices[default_input_device_index]
        
        logging.info("")
        logging.info("=" * 70)
        logging.info(f"🎤  USING MICROPHONE: {device_info['name']}")
        logging.info(f"📋  DEVICE INFO:")
        logging.info(f"    Default Sample Rate (from device): {int(device_info['default_samplerate'])} Hz")
        logging.info(f"    Max Input Channels: {device_info['max_input_channels']}")
        logging.info(f"    Script Target Sample Rate: {expected_sample_rate} Hz")
        logging.info(f"🆔  DEVICE ID: {default_input_device_index}")
        logging.info("=" * 70)
        logging.info("")
        
        if device_info['max_input_channels'] < 1:
            logging.error("❌ Selected default device has no input channels.")
            return False
        
        # Optional: Record a short test sample to verify signal and sample rate compatibility
        duration = 0.5  # seconds
        logging.info(f"🔊 Testing microphone with a short recording at {expected_sample_rate} Hz...")
        try:
            test_recording = sd.rec(
                int(duration * expected_sample_rate), 
                samplerate=expected_sample_rate, 
                channels=1, 
                blocking=True # Waits until recording is finished
            )
            sd.wait() # Explicitly wait for the blocking recording to complete
        except sd.PortAudioError as pae:
            logging.error(f"❌ PortAudioError during test recording: {pae}")
            logging.error(f"   This might indicate an issue with the selected sample rate ({expected_sample_rate} Hz) "
                          f"or device configuration. Check if your microphone supports this sample rate.")
            return False
        except Exception as e_rec: # Catch other potential recording errors
            logging.error(f"❌ An unexpected error occurred during test recording: {e_rec}", exc_info=True)
            return False


        if np.abs(test_recording).max() < 0.01: # Check if recording contains mostly silence
            logging.warning("⚠️ Very low audio levels detected during test. Check microphone connection, volume, and ensure it's not muted.")
        else:
            logging.info("✅ Microphone test recording successful (signal detected).")
            
        return True
        
    except Exception as e:
        logging.error(f"❌ Error accessing or testing microphone: {e}", exc_info=True)
        return False


def main():
    """
    Main function that handles user input for curvature, sets up audio recording parameters,
    records audio data in a loop, processes it using FFT for specified frequencies,
    and saves the extracted features along with metadata to a CSV file.
    """
    # ===== Audio Setup =====
    # Sample rate for recording. Must be at least twice the maximum target frequency (Nyquist theorem).
    # 44100 Hz is a common CD-quality rate and sufficient for frequencies up to 22050 Hz.
    Fs = 44100  # Sample rate in Hz
    
    # ===== Verify Microphone Availability =====
    if not check_microphone(expected_sample_rate=Fs):
        logging.error("❌ Cannot proceed without a working microphone or if microphone test failed. Please check audio settings.")
        return
    
    # ===== User Input for Radius =====
    while True:
        try:
            radius_value_cm_str = input("Enter the radius of curvature of your test object (in cm), or 'q' to quit: ")
            if radius_value_cm_str.lower() == 'q':
                logging.info("Exiting script as per user request.")
                return
            radius_value_cm = float(radius_value_cm_str)
            if radius_value_cm <= 0:
                raise ValueError("Radius must be a positive number.")
            radius_value_mm = radius_value_cm * 10  # Convert cm to mm
            break
        except ValueError as e:
            logging.error(f"Invalid input: {e}. Please enter a positive number or 'q'.")

    # Calculate curvature (inverse of radius)
    curvature_value = 1 / radius_value_mm # Curvature in mm^-1
    # Create a string representation of curvature for use in filenames (more robust for various float formats)
    curvature_str_for_filename = f"{curvature_value:.6f}".replace(".", "_").rstrip('0').rstrip('_') 

    # Set up the output directory path.
    # Assumes this script is in a subfolder (e.g., 'scripts') and 'csv_data/raw' is relative to the parent project directory.
    script_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(script_dir) 
    output_dir = os.path.join(project_root, "csv_data", "raw")
    
    # Construct the simplified filename without automatic versioning
    output_filename = f"raw_audio_{curvature_str_for_filename}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Optional: Add a warning if the file already exists
    if os.path.exists(output_path):
        logging.warning(f"⚠️ File {output_filename} already exists and will be overwritten.")

    logging.info(f"📐 Radius: {radius_value_cm} cm ({radius_value_mm} mm)")
    logging.info(f"📐 Curvature: {curvature_value:.6f} mm⁻¹")
    logging.info(f"💾 Output path: {output_path}")

    # ===== Audio Recording Parameters =====
    L = 10000   # FFT window size (number of audio samples per FFT calculation).
                # Frequency Resolution = Fs / L. With Fs=44100Hz, L=10000 -> Res = 4.41 Hz.
                # Window Duration = L / Fs. With Fs=44100Hz, L=10000 -> Duration = ~0.226 seconds.
    step_size = 1000  # Step size for the sliding window (number of new samples before next FFT).
                      # Hop Duration = step_size / Fs. With Fs=44100Hz, step=1000 -> Hop = ~0.0226 seconds.
    
    # Target frequencies for FFT analysis (100 Hz to 16,000 Hz in 100 Hz steps)
    target_frequencies = list(range(100, 16001, 100))

    # Initialize a double-ended queue (deque) as a rolling buffer.
    # `maxlen=L` ensures the buffer never exceeds the FFT window size.
    # Old samples are automatically discarded from the left when new samples are added to the right if full.
    buffer = deque(maxlen=L) 
    results = [] # List to store processed FFT results and metadata
    is_running = True # Flag to control the main recording loop

    def audio_callback(indata, frames, time_info, status):
        """
        Callback function called by sounddevice for each new block of audio data.
        
        It appends new audio data to a buffer. When the buffer reaches the desired
        FFT window size (L), it performs FFT analysis and stores the features.
        It then slides the window by removing `step_size` old samples.
        
        Parameters:
        -----------
        indata : np.ndarray
            Input audio data block (frames x channels). We assume mono (1 channel).
        frames : int
            Number of frames in `indata` (should match `blocksize` of the stream).
        time_info : CData
            Timestamp information from the stream.
        status : sd.CallbackFlags
            Status flags indicating if any stream errors occurred.
        """
        nonlocal is_running # Allow modification of the outer scope's is_running flag
        try:
            if status:
                logging.warning(f"⚠️ Audio stream status: {status}")
            
            # `indata` is a 2D array (frames, channels). We use the first channel [:, 0].
            # Extend the buffer from the right with new samples.
            buffer.extend(indata[:, 0])

            # Process data only when the buffer is completely filled to the window size L.
            if len(buffer) == L:
                # Convert the deque buffer to a NumPy array for FFT processing.
                window_data = np.array(buffer) 
                
                # Call utility function to extract FFT magnitudes for the target frequencies.
                FFT_values = extract_fft_features(
                    signal=window_data,
                    sample_rate=Fs,
                    window_size=L, 
                    target_freqs=target_frequencies
                )
                
                # Get a precise timestamp for this data sample. Use UTC for consistency.
                timestamp = pd.Timestamp.now(tz='UTC').isoformat()
                
                # Combine FFT values, the current curvature, and timestamp into a new row.
                new_row = FFT_values + [curvature_value, timestamp]
                results.append(new_row)

                # Provide periodic feedback on the number of samples collected.
                if len(results) % 20 == 0: 
                    logging.info(f"🔢 Samples collected: {len(results)}")

                # Implement the sliding window:
                # Remove `step_size` oldest samples from the left end of the buffer
                # to make space for the next block of `step_size` new samples.
                for _ in range(step_size):
                    if buffer: # Ensure buffer is not empty before popping
                        buffer.popleft()
            
        except Exception as e:
            # Log any error occurring within the callback and signal the main loop to stop.
            logging.error(f"Error in audio callback: {e}", exc_info=True)
            is_running = False 

    logging.info("🎙️ Starting audio recording... Press Ctrl+C to stop.")
    
    # Ensure the output directory exists before starting the stream.
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as oe:
        logging.error(f"❌ Could not create output directory {output_dir}: {oe}")
        return

    stream = None # Initialize stream variable to ensure it's defined in `finally`
    try:
        # Open the audio input stream.
        # The `audio_callback` will be called every time `blocksize` (i.e., `step_size`) new frames are available.
        stream = sd.InputStream(
            callback=audio_callback, 
            samplerate=Fs, 
            channels=1, # Mono audio
            blocksize=step_size 
        )
        stream.start() # Start the audio stream
        
        # Keep the main thread alive while the callback processes audio in the background.
        # The `is_running` flag can be set to False by the callback (on error) or by Ctrl+C.
        while is_running: 
            time.sleep(0.1) # Sleep briefly to reduce CPU usage in this loop

    except KeyboardInterrupt:
        logging.info("\n🛑 Recording manually stopped by user (Ctrl+C).")
    except Exception as e:
        logging.error(f"❌ Unexpected error during recording stream setup or main loop: {e}", exc_info=True)
    finally:
        is_running = False # Ensure the recording/callback loop condition is false
        
        if stream is not None:
            logging.info("Stopping audio stream...")
            if stream.active: # Check if stream is actually active before trying to stop
                stream.stop()
            stream.close()
            logging.info("Audio stream closed.")
        
        # Save the collected data to a CSV file.
        if results:
            logging.info(f"Collected {len(results)} samples in total.")
            # Define column headers for the CSV file.
            columns = [f"FFT_{f}Hz" for f in target_frequencies] + ["Curvature", "Timestamp"]
            df = pd.DataFrame(results, columns=columns)
            try:
                df.to_csv(output_path, index=False)
                logging.info(f"\n✅ Done. CSV saved to {output_path}")
            except Exception as e:
                logging.error(f"❌ Error saving CSV: {e}", exc_info=True)
        else:
            logging.warning("⚠️ No data was recorded. CSV file not created.")

if __name__ == "__main__":
    main()

"""
Summary of Commenting Enhancements:

1. **`check_microphone` function:**
   - Added more detail to the docstring about the `expected_sample_rate` parameter.
   - Clarified logging messages.
   - Added comments for `sd.wait()` and catching `PortAudioError`.

2. **`main` function:**
   - Clarified the purpose of `Fs` (Sample Rate).
   - Added more detailed comments for `L` (FFT window size), explaining frequency resolution and window duration.
   - Added more detailed comments for `step_size`, explaining hop duration.
   - Clarified the `buffer = deque(maxlen=L)` initialization and its behavior.

3. **`audio_callback` function:**
   - Expanded the docstring to better explain its role, parameters, and the sliding window mechanism.
   - Added comments clarifying that `indata` is (frames, channels) and we take `[:, 0]`.
   - Added comments explaining that processing happens when `len(buffer) == L`.
   - Clarified the purpose of popping `step_size` samples for the sliding window.

4. **Stream Handling in `main`:**
   - Added comments explaining that `blocksize=step_size` means the callback is invoked with that many new frames.
   - Added a comment explaining the `while is_running: time.sleep(0.1)` loop.
   - Ensured the `finally` block comments clearly state actions like stopping and closing the stream.

5. **General:**
   - Ensured consistent terminology (e.g., "frames" vs. "samples" where appropriate).
   - Added `exc_info=True` to error logs for more detailed tracebacks during debugging.
   - Improved formatting of some multi-line comments.

These changes should make the script's operation, particularly the audio processing pipeline, more transparent and easier to understand for future developers or maintainers.
"""