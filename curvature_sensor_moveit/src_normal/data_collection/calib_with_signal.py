import rospy
import numpy as np
from scipy.io.wavfile import write
from scipy.io import wavfile
import sounddevice as sd
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt

class CalibWithSignal:
    def __init__(self):
        rospy.init_node("calibration_with_signal_node", anonymous=True)
        self.response_pub = rospy.Publisher("/sensor_responses", Float64MultiArray, queue_size=10)
        self.sample_rate = 44100  # Hz
        self.duration = 1.0  # seconds
        self.window_size = 1024
        self.step_size = 512

    def record_response(self):
        """
        Records audio data from the sensor.
        """
        rospy.loginfo("Recording sensor response...")
        response = sd.rec(int(self.sample_rate * self.duration), samplerate=self.sample_rate, channels=1, blocking=True)
        rospy.loginfo("Recording complete.")
        return response.flatten()

    def save_to_wav(self, response, file_path="recorded_response.wav"):
        """
        Saves the recorded response to a .wav file.
        """
        write(file_path, self.sample_rate, response.astype(np.int16))
        rospy.loginfo(f"Response saved to {file_path}")

    def compute_fft(self, response):
        """
        Computes the FFT of the recorded response.
        """
        fft = np.fft.fft(response)
        freqs = np.fft.fftfreq(len(fft), 1 / self.sample_rate)
        amplitudes = np.abs(fft[:len(fft) // 2])
        return freqs[:len(freqs) // 2], amplitudes

    def plot_fft(self, freqs, amplitudes):
        """
        Plots the FFT of the response.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(freqs, amplitudes)
        plt.title("FFT of Recorded Response")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    def perform_calibration(self):
        """
        Records, saves, and computes FFT for sensor response during calibration.
        """
        rospy.loginfo("Starting calibration...")
        response = self.record_response()
        self.save_to_wav(response, "recorded_response.wav")

        freqs, amplitudes = self.compute_fft(response)
        self.plot_fft(freqs, amplitudes)

        response_msg = Float64MultiArray(data=amplitudes.tolist())
        self.response_pub.publish(response_msg)

if __name__ == "__main__":
    calib = CalibWithSignal()
    calib.perform_calibration()
