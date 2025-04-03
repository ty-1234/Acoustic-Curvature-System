import numpy as np
import scipy.io.wavfile as wav

# Function to generate a sine wave for given frequencies and duration
def generate_multi_tone(frequencies, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    for freq in frequencies:
        signal += 0.25 * np.sin(2 * np.pi * freq * t)  # Reduce amplitude to avoid clipping
    return signal

# Get user input for duration and frequencies
duration = float(input("Enter the duration of the sound wave (seconds): "))
frequencies_input = input("Enter the frequencies (Hz) separated by commas (e.g., 200,400,600): ")
frequencies = [float(freq.strip()) for freq in frequencies_input.split(',')]

# Generate the multi-tone wave
sample_rate = 44100
multi_tone_signal = generate_multi_tone(frequencies, duration, sample_rate)

# Normalize the signal to fit in the range of int16
multi_tone_signal = (multi_tone_signal / np.max(np.abs(multi_tone_signal)) * 32767).astype(np.int16)

# Save the signal to a .wav file
output_filename = f"wav_{duration}s.wav"
wav.write(output_filename, sample_rate, multi_tone_signal)

print(f"Sound wave saved as {output_filename}")
