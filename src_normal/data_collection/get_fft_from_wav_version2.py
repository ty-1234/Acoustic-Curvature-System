import numpy as np
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt

# defining the directory path
# directory_path = os.path.join(os.getcwd(), "src_normal/data_collection")

# defining file name   ------> select the file you want to analyze
file_name = "wav_60.0s.wav"

file_path = os.path.join(file_name)

# Check existence of the file within the directory
if os.path.exists(file_path):
    print(f"The file {file_name} exists.")
else:
    print(f"The file {file_name} does not exist in the specified directory.")


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
    Plots the FFT results, limiting to 1000 Hz.

    Args:
        freqs (np.ndarray): Array of frequency bins.
        amplitudes (np.ndarray): Array of amplitude values corresponding to each frequency.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, amplitudes)
    plt.title("FFT of Audio Signal (0-1000 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1000)  # Limit x-axis to 1000 Hz
    plt.grid()
    plt.show()


# Main execution
freqs, amplitudes = get_fft_from_wav(file_path)
plot_fft(freqs, amplitudes)
