import os
import numpy as np
import scipy.io.wavfile as wav

def generate_multi_tone(frequencies, duration, sample_rate=44100):
    """
    Generates a combined sine wave signal for the given list of frequencies.

    Args:
        frequencies (list): List of frequencies in Hz.
        duration (float): Duration of the signal in seconds.
        sample_rate (int): Sampling rate in Hz.

    Returns:
        np.ndarray: Normalized multi-tone waveform.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    
    for freq in frequencies:
        signal += 0.25 * np.sin(2 * np.pi * freq * t)  # scaled to prevent clipping

    # Normalize to int16 range
    signal = (signal / np.max(np.abs(signal)) * 32767).astype(np.int16)
    return signal

def main():
    duration = 300.0  # seconds
    sample_rate = 44100
    frequencies = list(range(200, 2001, 200))  # [200, 400, ..., 2000]

    print(f"Generating multi-tone signal for frequencies: {frequencies}")
    multi_tone_signal = generate_multi_tone(frequencies, duration, sample_rate)

    # Get the directory where main.py is located
    main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(main_dir, f"{duration}s.wav")

    wav.write(filename, sample_rate, multi_tone_signal)
    print(f"✔ Sound wave saved as '{filename}'")

if __name__ == "__main__":
    main()
