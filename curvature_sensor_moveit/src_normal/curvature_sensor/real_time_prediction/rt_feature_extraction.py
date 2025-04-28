"""
Real-time feature extraction from microphone data for curvature prediction.

This script captures audio data from the microphone in real-time, processes it
through FFT analysis, and feeds the resulting features to a pre-trained model
for real-time prediction of curvature and position.

Author: Bipindra Rai
Date: 2025-04-20
"""

import numpy as np
import sounddevice as sd
import time
import logging
from collections import deque
import threading
from sklearn.preprocessing import MinMaxScaler
from utils.rt_fft_utils import extract_fft_features
import os
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class RealTimeFeatureExtractor:
    """
    Extracts audio features in real-time from microphone input for curvature prediction.
    
    This class handles continuous audio capture, FFT processing, and feature extraction
    to provide scaled features for the prediction model.
    """
    
    def __init__(self, sample_rate=22050, window_size=10000, step_size=1000, 
                 target_frequencies=None, buffer_size=5):
        """
        Initialize the real-time feature extractor.
        
        Parameters:
        -----------
        sample_rate : int
            Audio sampling rate in Hz
        window_size : int
            Size of the FFT window in samples
        step_size : int
            Step size for the sliding window in samples
        target_frequencies : list
            List of target frequencies to extract from FFT
        buffer_size : int
            Number of feature sets to buffer for processing
        """
        # Audio parameters
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.step_size = step_size
        
        # Target frequencies for FFT analysis (200Hz to 2000Hz in steps of 200Hz)
        self.target_frequencies = target_frequencies or list(range(200, 2001, 200))
        
        # Audio buffer to hold samples for processing
        self.audio_buffer = deque(maxlen=window_size + step_size)
        
        # Feature buffer to hold extracted features
        self.feature_buffer = deque(maxlen=buffer_size)
        
        # Feature related variables
        self.feature_ready = threading.Event()
        self.latest_features = None
        
        # Control flags
        self.is_running = False
        self.stream = None
        self.processing_thread = None
        
        # Additional feature columns needed for model
        self.fft_cols = [f"FFT_{f}Hz" for f in self.target_frequencies]
        
        # For calculating frequency bands
        self.low_freq_indices = [i for i, f in enumerate(self.target_frequencies) 
                                if 200 <= f <= 800]
        self.mid_freq_indices = [i for i, f in enumerate(self.target_frequencies) 
                               if 800 < f <= 1400]
        self.high_freq_indices = [i for i, f in enumerate(self.target_frequencies) 
                                if f > 1400]
        
        logger.info(f"Feature extractor initialized with {len(self.target_frequencies)} frequency bands")
        
    def extract_fft_features(self, signal):
        """
        Extract FFT features from the audio signal for the target frequencies.
        
        Parameters:
        -----------
        signal : ndarray
            Audio signal data
            
        Returns:
        --------
        dict
            Dictionary containing FFT values for target frequencies and derived features
        """
        # Apply Hanning window to reduce spectral leakage
        windowed_signal = signal * np.hanning(len(signal))
        
        # Get FFT values using the utility function
        fft_values = extract_fft_features(
            windowed_signal, 
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            target_freqs=self.target_frequencies
        )
        
        # Initialize feature dictionary with FFT values for target frequencies
        features = {}
        
        # Store FFT values in the features dictionary
        for i, freq in enumerate(self.target_frequencies):
            features[f"FFT_{freq}Hz"] = fft_values[i]
        
        # Create numpy array of FFT values for derived features
        fft_array = np.array(fft_values)
        
        # Calculate FFT peak index (which frequency has maximum amplitude)
        peak_index = np.argmax(fft_array)
        features["FFT_Peak_Index"] = peak_index
        
        # Calculate spectral centroid
        freqs = np.array(self.target_frequencies)
        if np.sum(fft_array) > 0:  # Avoid division by zero
            features["FFT_Centroid"] = np.sum(freqs * fft_array) / np.sum(fft_array)
        else:
            features["FFT_Centroid"] = 0
            
        # Calculate frequency band powers
        features["low_band_power"] = np.mean(fft_array[self.low_freq_indices]) if self.low_freq_indices else 0
        features["mid_band_power"] = np.mean(fft_array[self.mid_freq_indices]) if self.mid_freq_indices else 0
        features["high_band_power"] = np.mean(fft_array[self.high_freq_indices]) if self.high_freq_indices else 0
        
        # Add curvature active flag (always 1 for real-time prediction)
        features["Curvature_Active"] = 1
            
        return features
        
    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for audio stream processing.
        """
        if status and status.input_overflow:
            logger.warning("Audio input overflow")
            
        # Add audio data to buffer
        self.audio_buffer.extend(indata[:, 0])
        
    def process_audio(self):
        """
        Process audio data from buffer and extract features.
        This runs in a separate thread.
        """
        while self.is_running:
            # Process data if enough samples are available
            if len(self.audio_buffer) >= self.window_size:
                # Extract a window of audio data
                window_data = np.array(list(self.audio_buffer)[:self.window_size])
                
                # Extract FFT features
                features = self.extract_fft_features(window_data)
                
                # Store the features
                self.latest_features = features
                self.feature_buffer.append(features)
                
                # Signal that new features are ready
                self.feature_ready.set()
                
                # Remove processed samples to advance window
                for _ in range(self.step_size):
                    if self.audio_buffer:
                        self.audio_buffer.popleft()
            
            # Sleep to prevent CPU overuse
            time.sleep(0.01)
            
    def start(self):
        """Start the feature extraction thread."""
        if self.is_running:
            logger.warning("Feature extraction thread is already running")
            return True
            
        self.is_running = True
        
        # Clear any existing data
        self.audio_buffer.clear()
        self.feature_buffer.clear()
        
        # Get and print microphone information before starting the stream
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            device_info = devices[default_input]
            
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"🎤  MICROPHONE: {device_info['name']}")
            logger.info(f"🔊  SAMPLE RATE: {int(device_info['default_samplerate'])} Hz")
            logger.info(f"🆔  DEVICE ID: {default_input}")
            logger.info("=" * 70)
            logger.info("")
        except Exception as e:
            logger.warning(f"🎤  Could not query audio device information: {e}")
        
        # Start audio stream
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.step_size
        )
        self.stream.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Real-time feature extraction started")
        
        return True  # Explicitly return True on successful start
    
    def stop(self):
        """
        Stop audio capture and feature extraction.
        """
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Stop audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        # Wait for processing thread to finish
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            
        logger.info("Real-time feature extraction stopped")
        
    def get_latest_features(self, timeout=1.0):
        """
        Get the latest extracted features.
        
        Parameters:
        -----------
        timeout : float
            Maximum time to wait for features
            
        Returns:
        --------
        dict
            Dictionary of extracted features or None if no features available
        """
        # Wait for feature ready event with timeout
        if self.feature_ready.wait(timeout):
            self.feature_ready.clear()
            return self.latest_features
        return None
    
    def prepare_features_for_model(self, features, scaler=None):
        """
        Prepare features for model prediction by creating a feature vector and scaling.
        
        Parameters:
        -----------
        features : dict
            Dictionary of extracted features
        scaler : sklearn.preprocessing.MinMaxScaler
            Scaler to use for feature normalization
            
        Returns:
        --------
        ndarray
            Scaled feature vector ready for model prediction
        """
        if not features:
            return None
        
        # Print scaler information only once (at DEBUG level)
        if scaler and not hasattr(self, '_debug_printed'):
            try:
                self._debug_printed = True
                logger.debug(f"Scaler expects {scaler.n_features_in_} features")
                if hasattr(scaler, 'feature_names_in_'):
                    logger.debug(f"Expected feature names: {scaler.feature_names_in_}")
            except Exception as e:
                logger.debug(f"Could not extract feature information from scaler: {e}")
        
        # Get FFT values as numpy array for calculating statistics
        fft_values = np.array([features.get(col, 0) for col in self.fft_cols])
        
        # Create a dictionary with all features in the exact order
        feature_dict = {}
        
        # 1. FFT frequency values (10 features)
        for col in self.fft_cols:
            feature_dict[col] = features.get(col, 0)
        
        # 2. FFT statistics (5 features)
        feature_dict["FFT_Mean"] = np.mean(fft_values)
        feature_dict["FFT_Std"] = np.std(fft_values)
        feature_dict["FFT_Min"] = np.min(fft_values)
        feature_dict["FFT_Max"] = np.max(fft_values)
        feature_dict["FFT_Range"] = np.max(fft_values) - np.min(fft_values)
        
        # 3. Band means (3 features)
        low_band = fft_values[self.low_freq_indices] if self.low_freq_indices else np.array([0])
        mid_band = fft_values[self.mid_freq_indices] if self.mid_freq_indices else np.array([0])
        high_band = fft_values[self.high_freq_indices] if self.high_freq_indices else np.array([0])
        
        feature_dict["Low_Band_Mean"] = np.mean(low_band)
        feature_dict["Mid_Band_Mean"] = np.mean(mid_band)
        feature_dict["High_Band_Mean"] = np.mean(high_band)
        
        # 4. Band ratios (2 features)
        low_mean = np.mean(low_band)
        mid_mean = np.mean(mid_band)
        high_mean = np.mean(high_band)
        
        # Avoid division by zero
        if abs(low_mean) < 1e-10:
            low_mean = 1e-10
        if abs(mid_mean) < 1e-10:
            mid_mean = 1e-10
        
        feature_dict["Mid_to_Low_Band_Ratio"] = mid_mean / low_mean
        feature_dict["High_to_Mid_Band_Ratio"] = high_mean / mid_mean
        
        # 5. Add FFT_Peak_Index
        feature_dict["FFT_Peak_Index"] = features.get("FFT_Peak_Index", 0)
        
        # 6. Set Curvature_Active to ALWAYS be 1 for deployment
        feature_dict["Curvature_Active"] = 1
        
        # 7. Add remaining features
        feature_dict["FFT_Centroid"] = features.get("FFT_Centroid", 0)
        feature_dict["low_band_power"] = features.get("low_band_power", 0)
        feature_dict["mid_band_power"] = features.get("mid_band_power", 0)
        feature_dict["high_band_power"] = features.get("high_band_power", 0)
        
        # Create pandas DataFrame with proper column names to avoid the warning
        if scaler and hasattr(scaler, 'feature_names_in_'):
            # Convert dictionary to DataFrame with scaler's expected column names
            feature_df = pd.DataFrame([feature_dict], columns=scaler.feature_names_in_)
            
            # Scale features using the scaler
            try:
                scaled_features = scaler.transform(feature_df)
                return scaled_features
            except Exception as e:
                logger.error(f"Error scaling features: {e}")
                return None
        else:
            # Fallback to array-based approach if no scaler or column names
            feature_list = list(feature_dict.values())
            feature_array = np.array(feature_list).reshape(1, -1)
            
            if scaler:
                try:
                    feature_array = scaler.transform(feature_array)
                except Exception as e:
                    logger.error(f"Error scaling features: {e}")
                    
            return feature_array

# Function to test the real-time feature extraction
def test_feature_extractor():
    """
    Test function to demonstrate real-time feature extraction.
    """
    extractor = RealTimeFeatureExtractor()
    
    try:
        extractor.start()
        logger.info("Feature extraction started. Press Ctrl+C to stop.")
        
        # Run for a set amount of time, printing features
        for _ in range(10):
            features = extractor.get_latest_features()
            if features:
                # Just print the first few features to avoid overwhelming output
                logger.info(f"Features extracted: {list(features.items())[:3]}...")
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        extractor.stop()
        logger.info("Feature extraction stopped")

if __name__ == "__main__":
    # Run the test function if this script is executed directly
    test_feature_extractor()