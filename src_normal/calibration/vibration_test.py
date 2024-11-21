import rospy
import numpy as np
from scipy.io import wavfile
from calib_with_signal import CalibWithSignal
from std_msgs.msg import Float64MultiArray

class VibrationTest(CalibWithSignal):
    def perform_vibration_test_with_wav(self, wav_file):
        """
        Uses a .wav file as input to simulate vibration and analyze the response.
        """
        rospy.loginfo(f"Starting vibration test with .wav file: {wav_file}")
        sample_rate, response = self.read_wav_file(wav_file)
        freqs, fft_results = self.analyze_response(response, sample_rate)

        rospy.loginfo(f"Vibration Test FFT Results: Frequencies={freqs}, Amplitudes={fft_results}")
        response_msg = Float64MultiArray(data=fft_results.tolist())
        self.response_pub.publish(response_msg)

if __name__ == "__main__":
    wav_file_path = "example_vibration.wav"  # Replace with your .wav file path
    vibration_test = VibrationTest()
    vibration_test.perform_vibration_test_with_wav(wav_file_path)
