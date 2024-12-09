import os
import numpy as np
from scipy.io import wavfile
import csv
import sounddevice as sd


def record_audio(file_path, duration=5, sample_rate=44100, channels=1):
    """
    Records audio from the default microphone for a given duration and saves it as a .wav file.

    Args:
        file_path (str): Path to save the recorded .wav file.
        duration (int or float): Duration in seconds to record.
        sample_rate (int): Sampling rate in Hz.
        channels (int): Number of channels to record (1 for mono, 2 for stereo).
    """
    print(f"Recording audio for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()  # Wait until recording is finished
    wavfile.write(file_path, sample_rate, recording)
    print(f"Finished recording. Audio saved to {file_path}")


def get_fft_from_wav(file_path):
    """
    Reads a .wav file, computes its FFT, and returns the frequency bins and amplitudes.

    Args:
        file_path (str): Path to the .wav file.

    Returns:
        freqs (np.ndarray): Array of frequency bins.
        amplitudes (np.ndarray): Array of amplitude values corresponding to each frequency.
        sample_rate (int): The sample rate of the audio file.
    """
    # Read the .wav file
    sample_rate, data = wavfile.read(file_path)

    # Handle stereo files by taking only one channel
    if len(data.shape) > 1:
        data = data[:, 0]

    # Perform FFT
    fft_result = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
    # Only take the positive half of the spectrum
    half_length = len(fft_result) // 2
    freqs = freqs[:half_length]
    amplitudes = np.abs(fft_result[:half_length])

    return freqs, amplitudes, sample_rate


def compare_amplitude_spectra(file_ref, file_real, output_csv):
    """
    Compares the amplitude spectra of two .wav files and stores the frequency,
    amplitude from ref file, amplitude from real-time file, and their difference into a CSV file.

    Args:
        file_ref (str): Path to the reference .wav file.
        file_real (str): Path to the recorded .wav file.
        output_csv (str): Path to the output .csv file.
    """
    # Get FFT from both files
    freqs_ref, amps_ref, sr_ref = get_fft_from_wav(file_ref)
    freqs_real, amps_real, sr_real = get_fft_from_wav(file_real)

    # Check if sample rates match
    if sr_ref != sr_real:
        print("Warning: The sample rates of the two files do not match. Results may be inaccurate.")

    # If the lengths differ, we will truncate to the minimum length for comparison
    min_len = min(len(freqs_ref), len(freqs_real))
    freqs_ref = freqs_ref[:min_len]
    amps_ref = amps_ref[:min_len]
    freqs_real = freqs_real[:min_len]
    amps_real = amps_real[:min_len]

    # Compute the difference in amplitude
    amp_diff = amps_ref - amps_real

    # Write the results to a CSV file
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["Frequency(Hz)", "Ref_Amplitude", "Real_Amplitude", "Difference"])
        # Write data rows
        for f, a_ref, a_real, diff in zip(freqs_ref, amps_ref, amps_real, amp_diff):
            writer.writerow([f, a_ref, a_real, diff])

    print(f"Comparison complete. Results saved to {output_csv}")


if __name__ == "__main__":
    # Reference file path
    reference_file = "wav_60.0s.wav"  # Update with your reference wav file
    # Real-time recorded file name
    realtime_file = "realtime_recording.wav"
    # Output CSV file
    output_file = "fft_comparison.csv"

    # Check existence of reference file
    if not os.path.exists(reference_file):
        print(f"The reference file {reference_file} does not exist.")
    else:
        # Record a new real-time wav file
        duration = 5  # seconds of recording
        record_audio(realtime_file, duration=duration)

        # Now compare the FFTs of the reference and the newly recorded file
        compare_amplitude_spectra(reference_file, realtime_file, output_file)
