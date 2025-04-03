import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

file_path = file_path = "C:/Users/giuli/Documents/Audacity/example_piano.aup3"

def get_fft_from_wav(file_path):
    """
    Reads a .wav file, computes its FFT, and returns the frequency components and amplitudes.

    Args:
        file_path (str): Path to the .wav file.

    Returns:
        freqs (np.ndarray): Array of frequency bins.
        amplitudes (np.ndarray): Array of amplitude values corresponding to each frequency.
    """
    # Read the .wav file
    sample_rate, data = wavfile.read(file_path)
    
    # Handle stereo files by taking only one channel
    if len(data.shape) > 1:
        data = data[:, 0]

    # Perform FFT
    fft_result = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
    amplitudes = np.abs(fft_result[:len(fft_result) // 2])  # Only positive frequencies

    return freqs[:len(freqs) // 2], amplitudes

def plot_fft(freqs, amplitudes):
    """
    Plots the FFT results.

    Args:
        freqs (np.ndarray): Array of frequency bins.
        amplitudes (np.ndarray): Array of amplitude values corresponding to each frequency.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, amplitudes)
    plt.title("FFT of Audio Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()
