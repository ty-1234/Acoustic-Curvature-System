#!/home/franka/miniconda3/envs/tariq/bin/python
 
"""
Module Name: Data Collection for Cablibration
Author: Giuliano Gemmani
Email: giulianogemmani@gmail.com
Date: YYYY-MM-DD
 
Description:
    A brief summary of the module’s purpose and functionality.
 
"""
 
from __future__ import print_function
 
 
# Libraries needed
import os
import sys
import time
import math
import rospy
from pyaudio import PyAudio
 
"""
print(time.now()) # default datatype is int
print(float(time.now() * 1000)) # prints time and milliseconds # time from epoch
 
"""
 
 
import threading
import wave
import random
import datetime
import message_filters
import moveit_msgs.msg
import moveit_commander
import subprocess
import numpy as np
import pandas as pd
 
import tf
from matplotlib.pyplot import imsave
from math import pi, dist, fabs, cos
 
#from cv_bridge import CvBridge
from std_msgs.msg import String, Bool, Int16MultiArray
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState, Image
from moveit_msgs.msg import CollisionObject, DisplayTrajectory, MotionPlanRequest, Constraints, PositionConstraint, JointConstraint
from controller_manager_msgs.srv import ListControllers, ListControllersRequest, SwitchController, SwitchControllerRequest
from geometry_msgs.msg import PoseStamped, Pose, Point
from actionlib_msgs.msg import GoalStatusArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from moveit_commander.conversions import pose_to_list
from pathlib import Path
from geometry_msgs.msg import WrenchStamped
import csv  # Add CSV module for writing data
 
 
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import Pose, PoseStamped
 
 
def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False
 
    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)
 
    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)
 
    return True
 
 
class FrankaDataCollection(object):
    
    def __init__(self):
        super(FrankaDataCollection, self).__init__()
 
        moveit_commander.roscpp_initialize(sys.argv)                    # Initialize moveit_commander
        rospy.init_node("calibration_node", anonymous=True)             # Initialize node     
        robot = moveit_commander.RobotCommander()                       # Initialize tf listener  
        scene = moveit_commander.PlanningSceneInterface()               # Inizitalize the planning scene
        group_name = "panda_arm"                                        # Initialize the move group for the panda_arm
        move_group = moveit_commander.MoveGroupCommander(group_name)
 
        
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)   # Defines the topic on which the robot publishes the trajectory
        self.planning_frame = move_group.get_planning_frame()                                                                                       # get the planning frame
        self.eef_link = move_group.get_end_effector_link()                                                                                          # get ee frame
        self.move_group = move_group
        self.section_pub = rospy.Publisher('/curving_section', String, queue_size=10)  # Added publisher for section labels
        # Set the planner publiser
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )
 
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)
 
        # ee-link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)
 
        # List of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())
 
        # Sometimes for debugging it is useful to print the entire state of the robot
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
 
        # Joint names in a dictionary
        self.joint_names = {
            "joint1": "panda_joint1",
            "joint2": "panda_joint2",
            "joint3": "panda_joint3",
            "joint4": "panda_joint4",
            "joint5": "panda_joint5",
            "joint6": "panda_joint6",
            "joint7": "panda_joint7"
        }
 
        # Parameters : path coverage + pushing
 
        self.AB_dist =                  0.03                                                    # in [m] --- distance to travel
        self.n_loc =                    10                                                      # n. of locations on which pushing action is performed
        self.steplength =               self.AB_dist / self.n_loc                               # in [m]
        
        self.n_pushing_actions =        20                                                      # number of pushing action per contact location
        self.cycles =                   30                                                      # time on which travel AB is performed
 
        self.offset_pushing =           0.001                                                   # in [m] : offset position in z direction (world frame) where pushing action has to start
        self.time_of_pushing =          15                                                      # in [s] : time spent applying a prescribed force
        self.time_of_pushing_margin =   1                                                       # in [s] : time waited before (and after) a single pushing action is performed
        self.rebuild_time =             5                                                       # in [s] : time spent in wating for the
        self.pushing_increment =        0.1                                                     # in [N] : increment for the next pishing action within the same location
 
        self.N =                        self.n_loc * self.cycles*self.n_pushing_actions        # N of all Data points
        self.N_counter =                0                                                      # counter for data points collected
         # Names and IDs of the available robot controllers.
        self.CONTROLLER_NAME2ID = {
            "torque": "franka_zero_torque_controller",
            "position": "position_joint_trajectory_controller",
            "impedance": "panda_leader_cartesian_impedance_controller",
            "impedance_advanced": "cartesian_impedance_advanced_controller"
        }
        # IDs and names of the available robot controllers.
        self.CONTROLLERS_ID2NAME = {c_id: c_name for c_name, c_id in self.CONTROLLER_NAME2ID.items()}
 
        
        # HOME : joints & ee_pose (position, orientation)
 
        self.joint_home = [0.06841830223171314, -0.41813771533517224, -0.12852136230674158, -1.9645842290600706, -0.03316074410412047, 1.6035182970937927, 0.8142089765552016]
        self.pose_home = ([0.4223942641978751, -0.04042834717301551, 0.6434531923890092], [-0.9988081975410994, 0.039553416742988796, -0.027085673649872546, 0.009169407373883585])
        self.joint_middle = [-1.0863472633110847, 1.0869070429590033, 2.109136418413529, -2.216250412393391, 0.3391740601878158, 2.8352848181560386, -0.3562797731873062]
       
        self.joint_A_position = [-0.8578415742434058, 1.0355864113422364, 2.009982678902781, -2.0569649532050414, -0.031394204897361146, 2.997959954518935, -0.14988057123745482]
        self.CONTROLLER_NAME2ID = {
            "torque":               "franka_zero_torque_control[-0.7809745708194721, 1.1661021832040674, 1.9708445979462788, -2.0634305065221947, -0.5160919516219034, 3.047398131163245, 0.38560656003985133, 0.035884205251932144, 0.035884205251932144]ler",
            "position":             "position_joint_trajectory_controller",
            "impedance":            "panda_leader_cartesian_impedance_controller",
            "impedance_advanced":   "cartesian_impedance_advanced_controller",
            "force_controller":     "my_cartesian_force_controller"
        }
        
        # IDs and names of the available robot controllers.
        self.CONTROLLERS_ID2NAME = {c_id: c_name for c_name, c_id in self.CONTROLLER_NAME2ID.items()}
 
        # Get the name of the currently running robot controller.
        running_controllers = self.get_controllers_id(filter_state="running")
        if running_controllers:
            self.current_controller = running_controllers[0]
        else:
            self.current_controller = None
    
        
    # Returns the position and orientation of the end-effector in a tuple
    def get_ee_pose_tuple(self):        
        
        self.current_pose = self.move_group.get_current_pose().pose  
        rospy.loginfo("End-Effector Position: x={}, y={}, z={}".format(self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z))
        rospy.loginfo("End-Effector Orientation: x={}, y={}, z={}, w={}".format(self.current_pose.orientation.x, self.current_pose.orientation.y, self.current_pose.orientation.z, self.current_pose.orientation.w))
        
        return [self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z], \
            [self.current_pose.orientation.x, self.current_pose.orientation.y, self.current_pose.orientation.z, self.current_pose.orientation.w]
 
    def display_current_pose(self):
        pose = self.move_group.get_current_pose().pose
        rospy.loginfo("End-Effector Position: x={}, y={}, z={}".format(pose.position.x, pose.position.y, pose.position.z))
        rospy.loginfo("End-Effector Orientation: x={}, y={}, z={}, w={}".format(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))
    def get_quaternion_from_euler(self,roll, pitch, yaw):
        """
        Convert an Euler angle to a quaternion.
        
        Input
            :param roll: The roll (rotation around x-axis) angle in radians.
            :param pitch: The pitch (rotation around y-axis) angle in radians.
            :param yaw: The yaw (rotation around z-axis) angle in radians.
        
        Output
            :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
        """
        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        
        return [qx, qy, qz, qw]
 
    # Go to home position in joint space
    def go_home(self):
        joint_goal = self.joint_home
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
    def go_middle(self):
         joint_goal = self.joint_middle
         self.move_group.go(joint_goal, wait=True)
        
         self.move_group.stop()
         rospy.sleep(2.0)
    def go_to_A(self):
         joint_goal = self.joint_A_position
         self.move_group.go(joint_goal, wait=True)
        
         self.move_group.stop()
         rospy.sleep(2.0)
    def open_gripper(self):
        command = 'source ~/franka_ws/devel/setup.bash && rosrun franka_interactive_controllers libfranka_gripper_run 1'
        subprocess.Popen(['bash', '-c', command])
        rospy.loginfo("Executing open gripper command.")
        rospy.sleep(2.0)
 
    def close_gripper(self):
        command = 'source ~/franka_ws/devel/setup.bash && rosrun franka_interactive_controllers libfranka_gripper_run 0'
        subprocess.Popen(['bash', '-c', command])
        rospy.loginfo("Executing close gripper command.")
        #self._adjust_end_effector_orientation(11)
       # self._adjust_specific_joint_orientation("joint7",22.1)
        rospy.sleep(2.0)
    def _adjust_end_effector_orientation(self, angle_deg):
        # Get current pose of the end-effector
        current_pose = self.move_group.get_current_pose().pose
        
        # Convert the desired bend angle from degrees to radians
        angle_rad = math.radians(angle_deg)
        
        # Create a quaternion from the Euler angle (assuming rotation around the z-axis)
        quaternion = self.get_quaternion_from_euler(0, 0, angle_rad)
        
        # Modify the orientation of the end-effector with the quaternion
        current_pose.orientation.x = quaternion[0]
        current_pose.orientation.y = quaternion[1]
        current_pose.orientation.z = quaternion[2]
        current_pose.orientation.w = quaternion[3]
        
        # Create a new pose goal with the adjusted orientation
        pose_goal = Pose()
        pose_goal.position = current_pose.position
        pose_goal.orientation = current_pose.orientation
        
        # Move to the new pose with adjusted wrist orientation
        self.move_group.set_pose_target(pose_goal)
        success = self.move_group.go(wait=True)
        
        if not success:
            rospy.loginfo("Failed to adjust wrist orientation.")
        
        # Clear the pose targets after moving
        self.move_group.clear_pose_targets()
    def _adjust_specific_joint_orientation(self, joint_name, angle_deg):
        """
        Adjust the orientation of a specific joint (e.g., joint5) instead of the whole arm.
 
        Parameters:
        - joint_name (str): The name of the joint to adjust (e.g., "joint5").
        - angle_deg (float): The desired angle for the joint in degrees.
        """
        # Get current joint positions (angles)
        current_joint_angles = self.move_group.get_current_joint_values()
 
        # Identify the index of the joint to adjust based on the joint name
        joint_index = list(self.joint_names.keys()).index(joint_name)
 
        # Modify the angle of the specific joint (convert to radians)
        angle_rad = math.radians(angle_deg)
        current_joint_angles[joint_index] = angle_rad
 
        # Set the modified joint angles as the target for the move group
        self.move_group.set_joint_value_target(current_joint_angles)
 
        # Move to the new joint configuration
        success = self.move_group.go(wait=True)
 
        if not success:
            rospy.logerr(f"Failed to adjust {joint_name} orientation.")
            return False  
 
        return True  
 
 
    # Create an ee-pose object in R^3 specifying a  tuple of position and orientation previoulsy recorded
    def create_pose(self, tuple_pos_ee):
        pose = Pose()
        pose.header.frame_id = '/panda_link0'
        pose.pose.position.x = tuple_pos_ee[0][0]
        pose.pose.position.y = tuple_pos_ee[0][1]
        pose.pose.position.z = tuple_pos_ee[0][2]
 
        pose.pose.orientation.x = tuple_pos_ee[1][0]
        pose.pose.orientation.y = tuple_pos_ee[1][1]
        pose.pose.orientation.z = tuple_pos_ee[1][2]
        pose.pose.orientation.w = tuple_pos_ee[1][3]
 
        rospy.loginfo("End-Effector Position: x={}, y={}, z={}".format(
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z ))
        rospy.loginfo("End-Effector Orientation: x={}, y={}, z={}, w={}".format(
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w ))
 
        return pose
    
    #Save the current joint state to a private member variable and publishes the result
    def log_position(self):      
        self.current_joint_states = self.move_group.get_current_joint_values()
        rospy.loginfo(f"Position logged: Joint state saved to current_joint_states")
 
    # This function is called whenever a new PoseStamped message is received   
    def pose_callback(self, msg):
        self.current_pose = msg
        rospy.loginfo("Received Pose: %s", self.current_pose)
    
    # Cretates a list of ee-pose used for the AB travel
    def point_generation(self):       
        initial_z = self.pose_A_tuple[0][2]  # Start from the initial z position
        for _ in range(self.n_loc):
            pose = Pose()
 
            # Fixed position for x and y
            pose.position.x =   self.pose_A_tuple[0][0]
            pose.position.y =   initial_z #self.pose_A_tuple[0][1]  # Fixed y position
            pose.position.z =   self.pose_A_tuple[0][1]             # Set the current z position
 
            # Fixed orientation
            pose.orientation.x = self.pose_A_tuple[1][0]
            pose.orientation.y = self.pose_A_tuple[1][1]
            pose.orientation.z = self.pose_A_tuple[1][2]
            pose.orientation.w = self.pose_A_tuple[1][3]
 
            # Add the pose to the list
            self.pose_list.append(pose)
 
            # Increment z for the next pose
            initial_z += self.steplength  # Update z for the next pose
 
        return self.pose_list
    
    # Create a list of 20 values from 0.1 to 2.0
    def force_generation(self):       
        forces = [round(i * 0.1, 1) for i in range(1, 21)]
        return forces
         
    # Move to a Cartesian Pose Goal
    def go_to_cartesian_pose(self, pose_goal):              
        self.move_group.set_pose_target(pose_goal)          
        success = self.move_group.go(wait=True)
        if not success:
            rospy.loginfo("Failed to reach the goal pose")
        self.move_group.clear_pose_targets()
 
    """
    Specify ee-pose in order to move             
    def move_to_point(self, x, y, z):
        #Move the robot to the given point in Cartesian space
        rospy.loginfo(f"Moving to grid point: {x}, {y}, {z}")
        pose_goal = Pose()
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z
          
        # Orientation is maintained the same
        pose_goal.orientation.x = self.pose_A_tuple[1][0]
        pose_goal.orientation.y = self.pose_A_tuple[1][1]
        pose_goal.orientation.z = self.pose_A_tuple[1][2]
        pose_goal.orientation.w = self.pose_A_tuple[1][3]
        self.go_to_cartesian_pose(pose_goal)
    """
    
    # Go to A (reference) ---> joint space
 
 
        # Go to a pose goal ---> Cartesian space
    def go_to_next_loc (self, pose_goal):            
            self.go_to_cartesian_pose(pose_goal)
            self.display_current_pose()
            self.move_group.clear_pose_targets()
 
        # Go to the upper position (current x,y, position and orientation), but upper in z direction ----> World frame
    def go_up(self):
            
            pose_goal = Pose ()                                     # create Pose Object
            pose_goal = self.move_group.get_current_pose().pose     # get current pose
            pose_goal.pose.position.z =  self.pose_A_tuple[0][2]    # adjust pose on z direction as for A
            
            self.move_group.set_pose_target(pose_goal)              # Move upwards
            success =               self.move_group.go(wait=True)   # Check if desired pose is achieved                    
            if not success:                                                          
                rospy.loginfo("Failed to reach the goal pose")
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            self.display_current_pose()                                             
 
        # Defines a service-client request to retrieve the name of the controller
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
        Args:            controller_name (str): The name of the controller. Valid options are "torque" or "position".
        Returns:         bool: Success.
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
            # Start the required controller.l
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
 
 
        
    # Node publisher for setting force
    def set_force_publisher(self, force_z):
            
            rospy.init_node('force_publisher_node', anonymous=True)
            pub = rospy.Publisher('/my_cartesian_force_controller/target_wrench', WrenchStamped, queue_size=10)
            rate = self.rate  # Publish at 10 Hz
 
            while not rospy.is_shutdown():
                # Create and publish a message
                msg = WrenchStamped()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "base_link"
                msg.wrench.force.x = 0.0
                msg.wrench.force.y = 0.0
                msg.wrench.force.z = force_z
                msg.wrench.torque.x = 0.0
                msg.wrench.torque.y = 0.0
                msg.wrench.torque.z = 0.0
 
                rospy.loginfo("Publishing target wrench: %s", msg)
                pub.publish(msg)
                rate.sleep()
 
        # Method for the subscribing node
    def set_force_subscriber(self, duration):
            rospy.init_node('force_subscriber_node', anonymous=True)                                                # Creation of the subscriber node
 
            # Callback function
            def callback(data):
                rospy.loginfo("Received message: %s", data)
            
            sub = rospy.Subscriber('/my_cartesian_force_controller/target_wrench', WrenchStamped, callback)         # Subscriber
            start_time = rospy.Time.now()                                                                           # Track the start time
            rospy.loginfo("Subscriber started. Listening for %d seconds...", duration)
 
            while rospy.Time.now() - start_time < rospy.Duration(duration):                                         # Run for the specified duration
                rospy.sleep(0.1)                                                                                    # Sleep briefly to allow messages to be processed
 
            rospy.loginfo("Time limit reached. Stopping subscriber.")
            sub.unregister()                                                                                        # Stop the subscription
    
    def recover_force(self):
            rospy.init_node('error_recovery_publisher')
            pub = rospy.Publisher('/franka_control/error_recovery/goal', ErrorRecoveryActionGoal, queue_size=10)
            recovery_goal = ErrorRecoveryActionGoal()
            pub.publish(recovery_goal)
            rospy.loginfo("Published recovery goal.")
 
            
            
        # Plan travel for going from A to B using list of points
    
    # def travel_AB(self):
        
    #     for i in range(self.n_loc):
            
    #         self.enable_controller("panda_leader_cartesian_impedance_controller")       # Check controller used is for MoveIt
    #         self.go_to_next_loc(pose_goal = self.pose_list[i])                          # Go to next loc point
            
    #         """
    #         for j in range(self.n_pushing_actions):
                
    #             self.enable_controller("my_force_controller")                                               # Switch controller to cartesian force controller            
    #             self.set_force_publisher(force_z = self.force_values[j])                                    # Publisher ---> Set pushing force : Force_j for pushing_action = j                                                   
    #             self.set_force_subscriber(duration = self.time_of_pushing + self.time_of_pushing_margin)    # Subscriber ---> Pushing action at current_loc = i for Force_j acting for 15 [s]
    #             self.recover_force()                                                                        # Policy for ERROR RECOVERY
 
    #             self.set_force_publisher(force_z = 0.0)                                                     # Release force applied Force_j = 0
    #             self.set_force_subscriber(duration = self.time_of_pushing + self.time_of_pushing_margin)    # Subscriber ---> Null force applied at ee acting for 2 [s]
    #             self.recover_force()
 
    #             self.enable_controller("panda_leader_cartesian_impedance_controller")                       # Switch controller to cartesian motion
    #             self.go_up()                                                                                # Go to upper position to start the next pushing action
    #         """
            
                                                                              
    def move_Downwards(self):
        # Ask the user for the known curvature value
        curvature_value = float(input("Enter known curvature value (e.g., 0.01818): "))
        curvature_str = str(curvature_value).replace(".", "_")
        output_dir = os.path.join(os.path.dirname(__file__), "../csv_data/raw")
        output_filename = f"raw_robot_{curvature_str}.csv"
        output_path = os.path.join(output_dir, output_filename)
 
        # Prepare a list to store data rows
        data_rows = []
        self.open_gripper()
        section_labels = ["0cm", "1cm", "2cm", "3cm", "4cm", "5cm"]
        for i, label in enumerate(section_labels):
            self.section_pub.publish(label)  # Publish the section label
            rospy.loginfo(f"📤 Published section label: {label}")
 
            # Get current pose of the end-effector
            current_pose = self.move_group.get_current_pose().pose
 
            # Close the gripper and log the action
            self.close_gripper()
            real_time_close = datetime.datetime.now().isoformat()
            rospy.sleep(5)  # Sleep for 2 seconds while the gripper is closed
            data_rows.append([label, current_pose.position.x, current_pose.position.y, current_pose.position.z, real_time_close, "close"])
 
            # Open the gripper and log the action
            self.open_gripper()
            real_time_open = datetime.datetime.now().isoformat()
            data_rows.append([label, current_pose.position.x, current_pose.position.y, current_pose.position.z, real_time_open, "open"])
 
            # Proceed with robot movement (backwards in x-axis and y-axis)
            new_x = current_pose.position.x
            new_y = current_pose.position.y
            new_z = current_pose.position.z - 0.01
 
            # Create a new pose goal with the modified position
            pose_goal = Pose()
            pose_goal.position.x = new_x
            pose_goal.position.y = new_y
            pose_goal.position.z = new_z
 
            # Keep the orientation the same
            pose_goal.orientation.x = current_pose.orientation.x
            pose_goal.orientation.y = current_pose.orientation.y
            pose_goal.orientation.z = current_pose.orientation.z
            pose_goal.orientation.w = current_pose.orientation.w
 
            # Move the robot to the new position
            self.move_group.set_pose_target(pose_goal)
            success = self.move_group.go(wait=True)
 
           # rospy.sleep(2.0)
            if success:
                rospy.loginfo("Gripper successfully closed.")
            else:
                rospy.loginfo("Failed to move the robot horizontally backwards")
 
            # Clear pose targets after moving
            self.move_group.clear_pose_targets()
            rospy.sleep(2.0)
 
        # Save all rows to the CSV file
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Section', 'PosX', 'PosY', 'PosZ', 'Timestamp', 'Clamp'])  # Write header
            writer.writerows(data_rows)  # Write data rows
 
        rospy.loginfo(f"✅ Data saved to {output_path}")
 
def main():
    robot = FrankaDataCollection()
    #robot.open_gripper()
    robot.go_to_A()
    robot.move_Downwards()
    robot.open_gripper()
 
 
if __name__ == "__main__":
    main()