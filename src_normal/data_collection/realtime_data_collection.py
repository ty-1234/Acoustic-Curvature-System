#!/usr/bin/env python3
import rospy
import numpy as np
import pandas as pd
import sounddevice as sd
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, WrenchStamped

# ==========================
# Helper Methods
# ==========================

def get_forces_from_wrench(wrench_msg):
    """
    Extracts the end-effector forces along x, y, z axes from a WrenchStamped message.
    """
    fx = wrench_msg.wrench.force.x
    fy = wrench_msg.wrench.force.y
    fz = wrench_msg.wrench.force.z
    return fx, fy, fz

def get_position_from_pose(pose_msg):
    """
    Extracts the end-effector position along x, y, z axes from a PoseStamped message.
    """
    x = pose_msg.pose.position.x
    y = pose_msg.pose.position.y
    z = pose_msg.pose.position.z
    return x, y, z

# def get_torque_from_wrench(wrench_msg):
#     """
#     Extracts the end-effector torque along x, y, z axes from a WrenchStamped message.
#     """
#     tx = wrench_msg.wrench.torque.x
#     ty = wrench_msg.wrench.torque.y
#     tz = wrench_msg.wrench.torque.z
#     return tx, ty, tz

# ==========================
# Global Variables
# ==========================
latest_pose = None
latest_wrench = None

def pose_callback(msg):
    global latest_pose
    latest_pose = msg

def wrench_callback(msg):
    global latest_wrench
    latest_wrench = msg

# ==========================
# Main Code
# ==========================

if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('fft_publisher', anonymous=True)

    # Subscribers
    pose_sub = rospy.Subscriber('/robot_pose', PoseStamped, pose_callback)
    wrench_sub = rospy.Subscriber('/end_effector_wrench', WrenchStamped, wrench_callback)

    # Publisher for the FFT results
    fft_pub = rospy.Publisher('/fft_results', Float64MultiArray, queue_size=10)

    Fs = 22050  # Reduced sample rate
    L = 10000   # Window length
    step_size = 1000  # Increased step size for overlapping windows

    # Empty list to store the results
    results = []

    # Record the start time
    start_time = pd.Timestamp.now()

    # We will record for approximately 50 seconds
    rate = rospy.Rate(10)  # 10 Hz loop
    while (pd.Timestamp.now() - start_time).total_seconds() < 50 and not rospy.is_shutdown():
        # Ensure we have the latest data
        if latest_pose is None or latest_wrench is None:
            rospy.logwarn("Waiting for pose and wrench messages...")
            rate.sleep()
            continue

        # Record audio data
        audio_data = sd.rec(L, samplerate=Fs, channels=1, blocking=True)
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

            # Get forces, position, torque
            fx, fy, fz = get_forces_from_wrench(latest_wrench)
            tx, ty, tz = get_torque_from_wrench(latest_wrench)
            x_pos, y_pos, z_pos = get_position_from_pose(latest_pose)

            # Current time
            current_time = pd.Timestamp.now()

            # Create a new row of results for the current iteration
            new_row = [
                FFT200, FFT400, FFT600, FFT800,
                x_pos, y_pos, z_pos,
                fx, fy, fz,
                tx, ty, tz,
                current_time
            ]
            results.append(new_row)

            # Publish the results via ROS
            fft_msg = Float64MultiArray(data=[
                FFT200, FFT400, FFT600, FFT800,
                x_pos, y_pos, z_pos,
                fx, fy, fz,
                tx, ty, tz,
                current_time.timestamp()
            ])
            fft_pub.publish(fft_msg)

            # Move to the next window
            start_idx += step_size
            end_idx += step_size

        rate.sleep()

    # Convert the list of results to a DataFrame
    columns = [
        'FFT200', 'FFT400', 'FFT600', 'FFT800',
        'PosX', 'PosY', 'PosZ',
        'ForceX', 'ForceY', 'ForceZ',
        'TorqueX', 'TorqueY', 'TorqueZ',
        'Time'
    ]
    results_df = pd.DataFrame(results, columns=columns)

    # Write the results DataFrame to a CSV file
    results_df.to_csv('C.csv', mode='a', index=False)

    rospy.loginfo("Data collection complete. Results saved to C.csv")

    # TODO: Once the recording for a specific location at a specific depth, the code asks the user
    # to specificy which location and which depth has to be addressed for the data recordeded
    # E.g. location C at depth 3
