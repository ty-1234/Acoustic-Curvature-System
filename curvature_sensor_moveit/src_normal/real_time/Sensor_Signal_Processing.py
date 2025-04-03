import rospy
import numpy as np
import sounddevice as sd
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
import pandas as pd

# Initialize ROS node
rospy.init_node('sensor_signal_handler', anonymous=True)

# Publisher for signal responses
response_pub = rospy.Publisher('/sensor_responses', Float64MultiArray, queue_size=10)

# Signal generation parameters
SAMPLE_RATE = 44100  # Hz
DURATION = 1.0       # seconds
FREQ = 1000          # Hz, change this as needed for different tests

# FFT parameters
WINDOW_SIZE = 1024  # Number of samples per FFT window
STEP_SIZE = 512     # Overlap step size for FFT
FFT_FREQS = [300, 500, 700, 900]  # Frequencies to analyze

# Function to generate and send a sine wave through the sensor
def generate_signal():
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * FREQ * t)  # Sine wave at target frequency
    return signal

# Function to record and process the response signal
def measure_response(signal):
    # Play the signal and record the response simultaneously
    response = sd.playrec(signal, samplerate=SAMPLE_RATE, channels=1, blocking=True)
    response = response.flatten()  # Convert to 1D array

    # Perform FFT analysis
    fft_results = []
    start_idx = 0
    while start_idx + WINDOW_SIZE <= len(response):
        window = response[start_idx:start_idx + WINDOW_SIZE]
        fft = np.fft.fft(window, n=WINDOW_SIZE)
        freqs = np.fft.fftfreq(WINDOW_SIZE, d=1/SAMPLE_RATE)
        fft_amplitudes = np.abs(fft[:WINDOW_SIZE // 2])

        # Extract amplitudes at target frequencies
        extracted_amplitudes = []
        for target_freq in FFT_FREQS:
            idx = np.argmin(np.abs(freqs[:WINDOW_SIZE // 2] - target_freq))
            extracted_amplitudes.append(fft_amplitudes[idx])

        fft_results.append(extracted_amplitudes)
        start_idx += STEP_SIZE

    return np.mean(fft_results, axis=0)  # Average across all windows

# Main function to handle signal generation, transmission, and response measurement
def main():
    rospy.loginfo("Sensor Signal Handler Node Started.")
    try:
        while not rospy.is_shutdown():
            # Generate signal
            signal = generate_signal()

            # Measure response
            fft_results = measure_response(signal)
            
            # Log and publish results
            rospy.loginfo(f"FFT Results: {fft_results}")
            response_msg = Float64MultiArray(data=fft_results)
            response_pub.publish(response_msg)

            # Pause between iterations
            rospy.sleep(2.0)

    except rospy.ROSInterruptException:
        rospy.loginfo("Sensor Signal Handler Node Stopped.")

if __name__ == "__main__":
    main()
