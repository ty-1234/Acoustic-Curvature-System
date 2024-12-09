import rospy
import csv
import numpy as np
import pandas as pd
import sounddevice as sd

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, WrenchStamped
from threading import Lock



class DataCollectorNode:
    def __init__(self):
        rospy.init_node("data_collector_node", anonymous=True)

        # File setup
        self.csv_file = "robot_data.csv"
        self.lock = Lock()
        self._initialize_csv()

        # Publisher for the FFT results
        self.fft_pub = rospy.Publisher('/fft_results', Float64MultiArray, queue_size=10)

        # Subscriptions
        self.pose_sub = rospy.Subscriber("/robot_pose", PoseStamped, self.pose_callback)
        self.wrench_sub = rospy.Subscriber("/end_effector_wrench", WrenchStamped, self.wrench_callback)

        # Data storage
        self.latest_pose = None
        self.latest_wrench = None

        self.Fs = 22050  # Reduced sample rate
        self.L = 10000  # Window length
        self.step_size = 1000  # Increased step size for overlapping windows

        # Empty list to store the results
        results = []

        def recording_loop ():
            # Record the start time
            start_time = pd.Timestamp.now()
            while (pd.Timestamp.now() - start_time).total_seconds() < 50:
                audio_data = sd.rec(self.L, samplerate = self.Fs, channels = 1, blocking = True)
                audio_data = audio_data.flatten()  # Flatten the audio data to a 1D array

                # Perform overlapping FFT
                start_idx = 0
                end_idx = L
                while end_idx <= len(audio_data):
                    window_data = audio_data[start_idx:end_idx]

                    # Perform FFT on the windowed data
                    Fn = Fs / 2
                    FTy = np.fft.fft(window_data, L) / L
                    Fv = np.linspace(0, 1, int(L / 2) + 1) * Fn
                    x = np.abs(FTy[:int(L / 2) + 1]) * 2

                    M = np.column_stack((Fv, x))

                    # Extract the FFT values at specific frequencies
                    target_frequencies = [200, 400, 600, 800]
                    FFT_values = np.zeros(len(target_frequencies))

                    for i, target_frequency in enumerate(target_frequencies):
                        index = np.argmin(np.abs(M[:, 0] - target_frequency))
                        FFT_values[i] = M[index, 1]

                    FFT200, FFT400, FFT600, FFT800 = FFT_values

                    # Display the FFT values
                    print(f"FFT200: {FFT200}")
                    print(f"FFT400: {FFT400}")
                    print(f"FFT600: {FFT600}")
                    print(f"FFT800: {FFT800}")

                    # Create a new row of results for the current iteration
                    current_time = pd.Timestamp.now()
                    new_row = [FFT200, FFT400, FFT600, FFT800, current_time]
                    results.append(new_row)

                    # Publish the results via ROS
                    self.fft_msg = Float64MultiArray(data=[ FFT200, FFT400, FFT600, FFT800, current_time.timestamp()])
                    self.fft_pub.publish(self.fft_msg)

                    # Move to the next window
                    start_idx += self.step_size
                    end_idx += self.step_size

            # Convert the list of results to a DataFrame
            results_df = pd.DataFrame(results, columns=['Force', 'FFT200', 'FFT400', 'FFT600', 'FFT800', 'Time'])

            # Write the results DataFrame to a CSV file
            # Using mode='a' to append if the file already exists; if you want to overwrite, use mode='w'
            results_df.to_csv('C.csv', mode='a', index=False)

    def _initialize_csv(self):
        """Initialize the CSV file with headers."""
        with open(self.csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Time", "X", "Y", "Z", \
                             "FFT200", "FFT400", "FFT600", "FFT800",\
                             "F_x", "F_y", "F_z",\
                             "Tor_x", "Tor_y", "Tor_z" ])

    def pose_callback(self, msg):
        """Callback for PoseStamped messages."""
        self.lock.acquire()
        try:
            self.latest_pose = msg
            self._write_to_csv()
        finally:
            self.lock.release()

    def wrench_callback(self, msg):
        """Callback for WrenchStamped messages."""
        self.lock.acquire()
        try:
            self.latest_wrench = msg
            self._write_to_csv()
        finally:
            self.lock.release()

    def _write_to_csv(self):
        """Write the latest data to the CSV file."""
        if self.latest_pose and self.latest_wrench:
            time_stamp = rospy.Time.now().to_sec()
            position = self.latest_pose.pose.position
            force = self.latest_wrench.wrench.force
            torque = self.latest_wrench.wrench.torque

            with open(self.csv_file, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    time_stamp,
                    position.x, position.y, position.z,
                    force.x, force.y, force.z,
                    torque.x, torque.y, torque.z
                ])

if __name__ == "__main__":
    try:
        collector = DataCollectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Data collection node terminated.")







