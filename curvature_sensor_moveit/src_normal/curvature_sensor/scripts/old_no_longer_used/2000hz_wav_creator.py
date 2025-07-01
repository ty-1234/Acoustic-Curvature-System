"""
WAV File Generator for Curvature Sensing Experiment.

This script generates a multi-tone WAV file containing sine waves at frequencies from 
200 Hz to 2000 Hz (in steps of 200 Hz). The generated file is used for playback during 
curvature sensing experiments to provide a consistent acoustic input signal.

Author: Bipindra Rai
Date: 2025-04-17
"""

import os
import numpy as np
import scipy.io.wavfile as wav

def generate_multi_tone(frequencies, duration, sample_rate=44100):
    """
    Generate a combined sine wave signal for the given list of frequencies.
    
    This function creates a multi-tone audio signal by summing sine waves at the
    specified frequencies. The amplitude of each tone is scaled to prevent clipping
    when combined.
    
    Parameters
    ----------
    frequencies : list
        List of frequencies in Hz to include in the signal
    duration : float
        Duration of the signal in seconds
    sample_rate : int, optional
        Sampling rate in Hz (default: 44100)
        
    Returns
    -------
    np.ndarray
        Normalized multi-tone waveform as a NumPy array of int16 values
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    
    for freq in frequencies:
        signal += 0.25 * np.sin(2 * np.pi * freq * t)  # scaled to prevent clipping

    # Normalize to int16 range
    signal = (signal / np.max(np.abs(signal)) * 32767).astype(np.int16)
    return signal

def main():
    """
    Main function to generate and save the multi-tone WAV file.
    
    This function sets up the parameters for the audio generation, creates the
    multi-tone signal, and saves it as a WAV file in the parent directory of the script.
    The filename includes the duration of the audio.
    """
    duration = 300.0  # seconds
    sample_rate = 44100
    frequencies = list(range(200, 2001, 200))  # [200, 400, ..., 2000]

    print(f"Generating multi-tone signal for frequencies: {frequencies}")
    multi_tone_signal = generate_multi_tone(frequencies, duration, sample_rate)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(script_dir, f"{duration}s.wav")

    wav.write(filename, sample_rate, multi_tone_signal)
    print(f"✔ Sound wave saved as '{filename}'")

if __name__ == "__main__":
    main()
