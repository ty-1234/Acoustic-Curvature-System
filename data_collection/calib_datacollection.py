#! /usr/bin/env python3

import os
import shlex
import subprocess
import sys
import time
from datetime import datetime

import json
import numpy as np
import cv2
import moveit_commander
import pathlib2 as pathlib
import rospy
import sensor_msgs.msg
import franka_msgs.msg
from controller_manager_msgs.srv import ListControllers, ListControllersRequest, SwitchController, SwitchControllerRequest
from cv_bridge import CvBridge
from franka_control.msg import ErrorRecoveryActionGoal
from geometry_msgs.msg import Pose 
import tf
import matplotlib.pyplot as plt
import tkinter as tk 
from tkinter import messagebox
import threading

object_list = ["cup"]
object_location_list = ["1", "2", "3", "4", "5"] 

robot_grasp_positions = ["left", "centre", "right"]

class FrankaDataCollection: 
	def __init__(self, data_dir, velocity_scaling_factor = 0.1, read_depth = False, read_pc = False, starting_sample = 399, record_topics = None,
	 current_object = 0, current_location = 0):
		print("-------------------------------")
		"""Franka data Collection.

		Args:
			data_dir (str): Directory where all outputs will be saved.
			velocity_scaling_factor (float, optional): Robot velocity scaling factor. Defaults to 0.15.
			read_depth (bool, optional): If True, save depth images. Defaults to False.
			read_pc (bool, optional): If True, save pointclouds. Defaults to False.
			starting_experiment (int, optional): Index of the starting experiment. Defaults to 0.
			starting_sample (int, optional): Index of the starting sample. Defaults to 0.
			record_topics (list[str], optional): List of topics names to record. Defaults to ["/tf", "/joint_states"].
		"""
		
		# start demonstration joint configuration
		self.start_joints = [-0.10620659970998486, -0.23509732580394072, 0.07914333387529641, -1.336153769626952, 0.021463943217311824, 1.1018796476280164, 0.7660253239952856]


		# Virtual square parameters 
		
		self.square_x_dist = 0.4
		self.z_dist = 0.8
		self.sq_length = 0.1

		self.pregrasp_height = 0.25
		self.grasp_height = 0.175
		
		self.grid_x_dist = 0.5 # distance centre of virtual square is in front of the base of the robot
		self.grid_length_x = 0.2
		self.grid_length_y = 0.3
		self.grid_size = 20
		self.points = self.generate_grid_of_points() 
		self.running = False


		# Settings.
		self.DATA_DIR = data_dir
		self.VELOCITY_SCALING_FACTOR = velocity_scaling_factor
		self.READ_DEPTH_IMAGE = read_depth
		self.READ_POINTCLOUD = read_pc
		self.RECORD_TOPICS = record_topics or ["/tf", "/joint_states"]
		self.current_sample = starting_sample
		self.current_object = current_object
		self.current_location = current_location

		# Other attributes.
		self.bridge = CvBridge()
		self.rosbag_proc = None
		self.current_controller = None
		self.show_color_img = False
		self.show_depth_img = False
		self.ee_state_full = []
		self.joint_full = []
		self.captured_image = False
		self.robot_position = None

		self.final_positions = []

		# Robot initialization.
		NODE_NAME = "auto_data_collection"
		rospy.init_node(NODE_NAME)
		moveit_commander.roscpp_initialize(sys.argv)
		self.robot = moveit_commander.RobotCommander()
		self.scene = moveit_commander.PlanningSceneInterface() 
		GROUP_NAME = "panda_arm"
		self.group = moveit_commander.MoveGroupCommander(GROUP_NAME)
		self.group.set_max_velocity_scaling_factor(self.VELOCITY_SCALING_FACTOR)
		self.group.set_end_effector_link("panda_link8")
		self.group.set_pose_reference_frame("panda_link0")

		# Setup subscriptions.
		self.img_sub_top = rospy.Subscriber("/hand_camera/color/image_raw", sensor_msgs.msg.Image, self.color_img_cb_top)
		self.depth_sub_top = rospy.Subscriber("/hand_camera/depth/image_rect_raw", sensor_msgs.msg.Image, self.depth_img_cb_top)
		# self.img_sub_top = rospy.Subscriber("/color/image_raw", sensor_msgs.msg.Image, self.color_img_cb_top)
		self.recoverError = rospy.Publisher("/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=10)
	

		# Names and IDs of the available robot controllers.
		self.CONTROLLER_NAME2ID = {
			"torque": "franka_zero_torque_controller",
			"position": "position_joint_trajectory_controller",
			"impedance": "panda_leader_cartesian_impedance_controller",
			"impedance_advanced": "cartesian_impedance_advanced_controller"
		}
		# IDs and names of the available robot controllers.
		self.CONTROLLERS_ID2NAME = {c_id: c_name for c_name, c_id in self.CONTROLLER_NAME2ID.items()}

		# Get the name of the currently running robot controller.
		running_controllers = self.get_controllers_id(filter_state="running")
		if running_controllers:
			self.current_controller = running_controllers[0]
		else:
			self.current_controller = None
			
		# Create output directories.
		pathlib.Path(self.experiment_path).mkdir(parents=True, exist_ok=True)
		
	def generate_grid_of_points(self):
		"""Generate a grid_sizexgrid_size grid of points"""
		return [(y, x) for y in np.linspace(-self.grid_length_y, self.grid_length_y, self.grid_size) for x in np.linspace(-self.grid_length_x, self.grid_length_x, self.grid_size)]
	
	def start_automation(self):
		"""Start the automated dataset collection process."""
		self.running = True
		self.full_autonomous_data_collection() 

	def stop_automation(self):
		"""Stop the automated dataset collection process."""
		self.running = False 
		rospy.loginfo("Automation process stopped.")

	def open_gripper(self):
		command = 'source ~/franka_ws/devel/setup.bash && rosrun franka_interactive_controllers libfranka_gripper_run 1'
		subprocess.Popen(['bash', '-c', command])
		rospy.loginfo("Executing open gripper command.")
	
	def close_gripper(self):
		command = 'source ~/franka_ws/devel/setup.bash && rosrun franka_interactive_controllers libfranka_gripper_run 0'
		subprocess.Popen(['bash', '-c', command])
		rospy.loginfo("Executing close gripper command.")

	def full_autonomous_data_collection(self):

		# make sure that the gripper is open
		self.open_gripper()

		# first grasp position
		# x,y = self.points[self.current_sample]
		y,x = self.points[self.current_sample]
		print(f'first grid coord is x: {x} y: {y}')
		print(f'first real space grid position is x: {x+self.grid_x_dist} y: {y}')
		# self.move_to_point(x+self.x_dist, y) 
		x_coord = x+self.grid_x_dist 
		y_coord = y
		self.move_to_point(x_coord,y_coord, self.pregrasp_height)

		# wait for object to be positioned
		self.show_continue_popup()
		# want tk window continue to be pressed

		while self.current_sample <= len(self.points):

			# position is logged
			self.log_position()

			# recording is active 
			# robot moves to home position 
			# recording is deactive 
			self.move_and_record_trajectory()

			# camera views logged 
			self.take_pictures_in_square()

			exit()

			# back to recorded position 
			self.go_to_logged_grid_position()

			# gripper down 
			self.move_to_point(x_coord,y_coord, self.grasp_height)

			rospy.sleep(2)

			# gripper closed 
			self.close_gripper()

			rospy.sleep(2)

			if self.current_sample < len(self.points):

				self.current_sample += 1
				
				# gripper up
				self.move_to_point(x_coord,y_coord, self.pregrasp_height)

				if self.current_sample % self.grid_size == 0:
					self.go_to_home_joint_state()

				# gripper new position 
				y,x = self.points[self.current_sample]
				x_coord = x+self.grid_x_dist 
				y_coord = y
				self.move_to_point(x_coord,y_coord, self.pregrasp_height)


				# gripper down 
				self.move_to_point(x_coord,y_coord, self.grasp_height)

				rospy.sleep(2)

				# gripper open
				self.open_gripper()

				rospy.sleep(2)

				# gripper up
				self.move_to_point(x_coord,y_coord, self.pregrasp_height)

			
			else:
				return
			
	def move_through_grid(self):

		# make sure that the gripper is open
		self.open_gripper()

		# wait for object to be positioned
		# self.show_continue_popup()
		# want tk window continue to be pressed

		for sample in range(len(self.points)):
			# gripper new position 
			y,x = self.points[sample]
			x_coord = x+self.grid_x_dist 
			y_coord = y
			self.move_to_point(x_coord,y_coord, self.pregrasp_height)



		

	def take_pictures_in_square(self):
		"""
		Automated collection of samples by moving the robot to different camera views
		"""
		self.go_to_home_joint_state()

		# Define the central virtual position 3 (centre of square) 
		home_pose = Pose() 
		home_pose.position.x = self.square_x_dist
		home_pose.position.y = 0.0 
		home_pose.position.z = self.z_dist
		# quaternion = tf.transformations.quaternion_from_euler(0, 3.14, 0)
		home_pose.orientation.x = 1.0
		home_pose.orientation.y = 0.0 
		home_pose.orientation.z = 0.0 
		home_pose.orientation.w = 0.0 

		# Compute IK for home pose 
		rospy.loginfo("Computing IK for home pose...")
		home_joint_goal = self.compute_ik(home_pose) # home_joint_goal is # plan
		if not home_joint_goal.joint_trajectory.points: 
			rospy.logerr("Failed to compute IK for the home pose.")
			return 

		# Move to the home position 
		rospy.loginfo("Moving to the home joint state...")
		# self.go_to_joint_state(home_joint_goal.joint_trajectory.points[-1].positions)
		
		# Define the four additional Cartesian poses 
		position_offsets = [
			{"x": 1.0, "y": 1.0}, # Position 1 
			{"x": 1.0, "y": -1.0}, # Position 2 
			{"x": 0.0, "y": 0.0}, # Position 3 (Middle)
			{"x": -1.0, "y": 1.0}, # Position 4
			{"x": -1.0, "y": -1.0}, # Position 5
		]

		for i, pos_off in enumerate(position_offsets):
			pose_goal = Pose() 
			pose_goal.position.x = self.square_x_dist + ( 0.5 * self.sq_length * pos_off["x"])
			pose_goal.position.y = (0.5 * self.sq_length * pos_off["y"])
			pose_goal.position.z = self.z_dist
			pose_goal.orientation = home_pose.orientation
			rospy.loginfo(f"Moving to Cartesian position {i+1}...")
			self.go_to_cartesian_pose(pose_goal)
			self.take_picture(i+1) 
			# self.go_to_home_joint_state()

		rospy.loginfo("Data collection completed. Returning to home position.")
		self.go_to_joint_state(home_joint_goal.joint_trajectory.points[-1].positions)




	def show_continue_popup(self):
		"""Display a blocking Tkinter popup window with a 'Continue' button."""	
		# Create a new top-level window 
		popup = tk.Toplevel()
		popup.title("Continue")

		# Create label and button in the window 
		label = tk.Label(popup, text="Please place the object and click 'Continue' to proceed.")
		label.pack(padx=20, pady=10)

		continue_button = tk.Button(popup, text="Continue", command=popup.destroy)
		continue_button.pack(pady=10) 
		rospy.logwarn("PLACE OBJECT AND PRESS CONTINUE BUTTON IN POPUP")

		# This will block the code execution until the popup is closed 
		popup.wait_window(popup) 
		rospy.loginfo("Continue button pressed. Resuming automation")

	def log_position(self):
		"""Save the current joint state to a private member variable"""
		self.grid_joint_state = self.group.get_current_joint_values() 
		rospy.loginfo(f"Position logged: Joint state saved to grid_joint_state")


	def move_and_record_trajectory(self):

		"""Move from the end position to the start position while recording a trajectory"""
		rospy.loginfo("Start recording")
		self.record_bag_file() 
		rospy.logwarn("RECORDING HAS STARTED. NOW MOVE BACK TO START POSITION")

		# Prepare lists to store trajectory data
		self.ee_state_full = []
		self.joint_full = []

		# Set the target joint state to the recorded start position 
		# This is the home position. Remember that the robot is moving from end to the start.
		self.group.set_joint_value_target(self.start_joints)

		# Plan a trajectory to the start position 
		plan = self.group.plan() 


		# Check if a valid plan was generated 

		if plan[1].joint_trajectory.points:
			rospy.loginfo("Trajectory plan generated. Executing and recording...")

			# Execute the planned trajectory while recording each step 
			for point in plan[1].joint_trajectory.points:
				# Set the target joint positions for the current trajectory point 
				self.group.set_joint_value_target(point.positions)
				self.group.go(wait=True) # Move to the next point in the trajectory 

				# Log current joint state and end-effector pose 
				ee_state = self.group.get_current_pose().pose 
				joint_state = self.group.get_current_joint_values() 
				self.ee_state_full.append([ee_state.position.x, ee_state.position.y, ee_state.position.z,
							  				ee_state.orientation.x, ee_state.orientation.y, ee_state.orientation.z, ee_state.orientation.w])

				self.joint_full.append(joint_state)

			rospy.loginfo("Trajectory execution complete. Stopping recording")

			self.go_to_home_joint_state()

			self.ee_state_full.append([ee_state.position.x, ee_state.position.y, ee_state.position.z,
							  				ee_state.orientation.x, ee_state.orientation.y, ee_state.orientation.z, ee_state.orientation.w])

			self.joint_full.append(joint_state)
			
			
		else:
			rospy.logwarn("Failed to generate a trajectory plan. Aborting movement")			

		# Step 3: Stop recording 
		self.stop_recording() 
		rospy.loginfo("Trajectory recording complete. Data saved")

		# Save the trajectory data to files
		save_path = os.path.join(self.experiment_path, f'demo_{self.current_sample}_traj')
		np.save(save_path + "_ee_state.npy", np.array(self.ee_state_full))
		np.save(save_path + "_joint_full.npy", np.array(self.joint_full))


	def go_to_home_joint_state(self):
		# Datacollection start position 
		joint_goal = self.start_joints
		self.group.go(joint_goal, wait=True)

	def go_to_logged_grid_position(self):

		self.group.go(self.grid_joint_state,wait=True)



	def move_to_point(self, x, y, z):
		"""Move the robot to the given point in Cartesian space."""
		rospy.loginfo(f"Moving to grid point: {x}, {y}") 
		pose_goal = Pose() 
		pose_goal.position.x = x 
		pose_goal.position.y = y 
		pose_goal.position.z = z
		pose_goal.orientation.x = 1.0 
		pose_goal.orientation.y = 0.0 
		pose_goal.orientation.z = 0.0 
		pose_goal.orientation.w = 0.0 
		self.go_to_cartesian_pose(pose_goal)

	

	def collect_data_at_point(self):
		"""Collect data at the current grid point."""
		# Record the joint state position and cartesian view 
		rospy.loginfo("Recording initial joint state and Cartesian view")
		joint_state = self.group.get_current_joint_values() 
		cartesian_pose = self.group.get_current_pose().pose 

		# Reverse trajectory recording 
		rospy.loginfo("Recording reverse joint state trajectory.")
		self.replay_demonstration() 

		# Take 5 pictures from different viewpoints 
		rospy.loginfo("Taking pictures from 5 viewpoints")
		for i in range(5):
			self.take_picture(i + 1) 

		# Return to the end position
		rospy.loginfo("Returning to the end position.")
		self.go_to_joint_state(joint_state) 

	def move_object_to_next_location(self):
		"""Simulate moving the object to a new location and reset point."""
		rospy.loginfo("Moving object to next location.")
		self.next_object_location() 
		self.current_point = 0

	def create_gui(self):
		"""Create a simple Tkinter GUI to start the automation process """
		root = tk.Tk() 
		root.title("Automated Dataset Collection")

		start_button = tk.Button(root, text="Start Automation", command=self.start_automation)
		start_button.pack(pady=20) 

		stop_button = tk.Button(root, text="Stop Automation", command=self.stop_automation)
		stop_button.pack(pady=20)

		exit_button = tk.Button(root, text="Exit", command=root.quit)
		exit_button.pack(pady=20) 

		root.mainloop() 

	def store_end_position(self):
		"""Stores the final end-effector Cartesian position (x, y) at the end of a demonstration."""
		final_pose = self.group.get_current_pose().pose # Get the current pose of the end effector
		x = final_pose.position.x 
		y = final_pose.position.y 
		rospy.loginfo(f"End effector position stored: x={x}, y={y}")
		self.final_positions.append((x, y))

	def plot_demonstrations(self):
		"""Plots the stored end goal positions in 2D space."""
		plt.close()
		if not self.final_positions:
			rospy.logwarn("No demonstrations to plot.")
			return 
		
		# Extract x and y positions 
		x_vals = [pos[0] for pos in self.final_positions]
		y_vals = [pos[1] for pos in self.final_positions]

		plt.figure(figsize=(6, 6))
		plt.scatter(y_vals, x_vals, c='red', label='End Points')
		plt.xlabel('Y Position (m)')
		plt.ylabel('X Position (m)')
		plt.title('Collected Demonstrations End Positions')
		plt.legend() 
		plt.grid(True) 
		plt.gca().invert_yaxis()
		plt.show(block=False)
		plt.pause(0.2)




	def color_img_cb_top(self, msg):
		"""Callback to get the RGB image from the ROS message."""

		# Get image from ROS message.
		color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
		# Convert from BGR to RGB.
		self.color_img_top = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

		self.show_color_img = True

	def depth_img_cb_top(self, msg):
		"""Callback to get the RGB image from the ROS message."""

		# Get image from ROS message.
		self.depth_img_top = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

		self.show_depth_img = True

		

	@property
	def experiment_path(self):
		"""Full path of the current experiment."""
		return os.path.join(self.DATA_DIR)#, "scene_" + str(object_list[self.current_object]))



	def take_picture(self, image_position):
		# Wait until we receive an image 
		rospy.sleep(1)
		if ((self.color_img_top is not None) and (self.depth_img_top is not None)):
			rospy.loginfo("Captured images at current position")
			self.capture_images(self.experiment_path, "demo_" + str(self.current_sample)+ "_image_" + str(image_position), image_position)
		else: 
			rospy.logwarn("Failed to capture image")


	def compute_ik(self, pose_target):
		self.group.set_pose_target(pose_target)
		plan = self.group.plan()[1]
		
		return plan 
	
	def go_to_joint_state(self, joint_goal):

		# This actually goes to different positions (joint_goal but I need to fix the offset.)
		self.group.go(joint_goal, wait=True)
		self.group.stop()

	def go_to_cartesian_pose(self, pose_goal):
		self.group.set_pose_target(pose_goal) 
		self.group.go(wait=True) 
		self.group.stop() 
		self.group.clear_pose_targets() 


		



	def get_controllers_id(self, filter_state=None):
		"""Get a list of the IDs of the avilable controllers.

		Args:
			filter_state (str, optional): If specified, only return controllers with matching state. Defaults to None.

		Returns:
			list[str]: List of the IDs of the controllers.
		"""

		rospy.wait_for_service("/controller_manager/list_controllers")
		try:
			request = ListControllersRequest()
			service = rospy.ServiceProxy("/controller_manager/list_controllers", ListControllers)
			response = service(request)
		except Exception as e:
			rospy.logerr("List controllers server is down. Unable to list running controllers.")
			rospy.logerr(e)
			return False

		if filter_state is None:
			return [c.name for c in response.controller]
		else:
			return [c.name for c in response.controller if c.state == filter_state]


	def enable_controller(self, controller_name):
		"""Enable a certain controller for the robot.

		Args:
			controller_name (str): The name of the controller. Valid options are "torque" or "position".

		Returns:
			bool: Success.
		"""


		if self.current_controller == controller_name:
			rospy.loginfo("Controller '{}' is already active.".format(self.current_controller))
			return True

		if controller_name not in self.CONTROLLER_NAME2ID.keys():
			rospy.logerr("Controller '{}' is not a valid controller! Not switching.".format(controller_name))
			return False

		running_controllers_id = self.get_controllers_id("running")
		# Limit to controllers specified in self.CONTROLLERS. This excludes the state controller, which should always be running.
		running_controllers_id = [c_id for c_id in running_controllers_id if c_id in self.CONTROLLER_NAME2ID.values()]

		rospy.wait_for_service("/controller_manager/switch_controller")
		try:
			request = SwitchControllerRequest()
			request.strictness = 1
			# Stop all running controllers.
			request.stop_controllers = running_controllers_id
			# Start the required controller.
			request.start_controllers = [self.CONTROLLER_NAME2ID[controller_name]]
			service = rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
			response = service(request)
		except Exception as e:
			rospy.logerr("Switch controller server is down. Unable to switch contoller to '{}'".format(controller_name))
			rospy.logerr(e)
			return False

		if not response.ok:
			rospy.logerr("Failed to switch to controller '{}'".format(controller_name))
			return False

		self.current_controller = controller_name

		
	def capture_images(self, output_dir, sample_id, image_position):
		"""Capture RGB, depth and/or pointcloud from the current position.
			sample_id (str): Sample id for the current capture.
		"""
		# if self.moved_away == True:
		# RGB.
		color_img_path_top = os.path.join(output_dir, "rgb_img_" + sample_id + ".png")

		cv2.imwrite(color_img_path_top, self.color_img_top)

		jet_depth_img_path_top = os.path.join(output_dir, "jet_depth_img_" + sample_id + ".png")
		normalized_depth_img = cv2.normalize(self.depth_img_top, None, 0, 255, cv2.NORM_MINMAX)

		depth_8bit = normalized_depth_img.astype(np.uint8)
		depth_8bit = cv2.normalize(self.depth_img_top, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		cv2.imwrite(jet_depth_img_path_top, cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET))

		depth_img_path_top = os.path.join(output_dir, "depth_img_" + sample_id + ".png")
		cv2.imwrite(depth_img_path_top, depth_8bit)



		rospy.loginfo("Recorder image '{}'".format(image_position))


	def record_bag_file(self):
		"""Record a ROS bag."""
		save_path = os.path.join(self.experiment_path,
								"demo_" + str(self.current_sample))
		# Ensure that the output directory exists. 
		pathlib.Path(self.experiment_path).mkdir(parents=True, exist_ok=True)

		def _record():
			command = ["rosbag", "record", "-O", save_path] + self.RECORD_TOPICS 
			self.rosbag_proc = subprocess.Popen(command) 

		# Run the recording function in a separate thread 
		recording_thread = threading.Thread(target=_record) 
		recording_thread.start() 
		rospy.loginfo("ROS bag recording started in a separate thread.")
	
	def stop_recording(self):
		"""Stop recording a ROS bag.
		
		Returns:
			bool: Success.
		"""

		if self.rosbag_proc is None:
			rospy.logwarn("No recording in progress: you can start one by pressing 's'.")
			return False

		# Stop recording.
		self.rosbag_proc.send_signal(subprocess.signal.SIGINT)
		self.rosbag_proc = None
		rospy.loginfo("Finished recording.")

		
		return True

	def next_object_location(self):
		"""Move to the next object location."""

		# Set indices.
		self.current_location += 1
		self.current_sample = 1
		
		if self.current_location >= len(object_location_list):
			self.current_object += 1
			print("COMPLETED FULL ROTATION. SELECTING NEXT OBJECT")
			
		if self.current_object >= len(object_list):
			print("REACHED END OF OBJECT LIST.\nEXITING NOW")
			exit()

		# Ensure output directories exist.
		pathlib.Path(self.experiment_path).mkdir(parents=True, exist_ok=True)

		rospy.loginfo("Starting experiment {}".format(object_location_list[self.current_location]))
	

	def recover_error(self):
		"""Quickly recover from a soft error (e.g. someone pushed the robot while in position control."""

		command = ErrorRecoveryActionGoal()
		self.recoverError.publish(command)



if __name__ == '__main__':
	# test_generate_grid_points()
	
	data_collection = FrankaDataCollection(data_dir="/home/franka/dataset_collection/src/dataset")
	
	# print(data_collection.group.get_current_joint_values() )
	# data_collection.go_to_home_joint_state()
	data_collection.create_gui()
	# data_collection.move_through_grid()
	# data_collection.take_pictures_in_square()

	

	

