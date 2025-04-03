import pyaudio
import numpy as np
import wave
import time
import threading
import csv

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "output.wav"
CSV_OUTPUT_FILENAME = "output.csv"

def record_audio(stop_event, frames):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    print("Recording...")
    while not stop_event.is_set():
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

def save_audio(frames, filename):
    audio = pyaudio.PyAudio()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def analyze_audio(filename):
    # Load the wave file
    wf = wave.open(filename, 'rb')
    n_frames = wf.getnframes()
    signal_wave = wf.readframes(n_frames)
    wf.close()
    
    # Convert to numpy array
    signal = np.frombuffer(signal_wave, dtype=np.int16)
    
    # Perform FFT (Fast Fourier Transform) to extract frequency components
    fft_spectrum = np.fft.fft(signal)
    frequency = np.fft.fftfreq(len(fft_spectrum), 1.0/RATE)
    
    # Get the magnitude of frequencies
    magnitude = np.abs(fft_spectrum)
    
    # Display top frequency components for analysis
    idx = np.argsort(magnitude)[::-1]
    print("Top frequency components:")
    top_frequencies = []
    for i in range(10):
        freq = frequency[idx[i]]
        mag = magnitude[idx[i]]
        print(f"Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f}")
        top_frequencies.append((freq, mag))
    
    # Save frequency components to CSV
    save_to_csv(top_frequencies, CSV_OUTPUT_FILENAME)

def save_to_csv(data, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frequency (Hz)", "Magnitude"])
        writer.writerows(data)

if __name__ == "__main__":
    stop_event = threading.Event()
    frames = []
    
    # Start recording thread
    record_thread = threading.Thread(target=record_audio, args=(stop_event, frames))
    record_thread.start()
    
    # Let it record for RECORD_SECONDS
    time.sleep(RECORD_SECONDS)
    stop_event.set()
    record_thread.join()
    
    # Save recorded audio to a file
    save_audio(frames, WAVE_OUTPUT_FILENAME)
    
    # Analyze the saved audio file and save results to CSV
    analyze_audio(WAVE_OUTPUT_FILENAME)
