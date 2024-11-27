#!/usr/bin/env python3

"""
Module Name: Data Collection for Cablibration
Author: Giuliano Gemmani
Email: giulianogemmani@gmail.com
Date: YYYY-MM-DD

Description:
    A brief summary of the module’s purpose and functionality.

"""

from __future__ import print_function
from six.moves import input

# Libraries needed
import os
import sys
import cv2
import time
import math
import rospy
import random
import datetime
import message_filters
import moveit_msgs.msg
import moveit_commander
import matplotlib.pyplot
import pathlib2 as pathlib

import numpy as np
import pandas as pd

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


from sensor_msgs.msg import JointState
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import Pose, PoseStamped
from franka_msgs.msg import ErrorRecoveryActionGoal


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
        

        # HOME : joints & ee_pose (position, orientation)

        self.joint_home = [0.06841830223171314, -0.41813771533517224, -0.12852136230674158, -1.9645842290600706, -0.03316074410412047, 1.6035182970937927, 0.8142089765552016]
        self.pose_home = ([0.4223942641978751, -0.04042834717301551, 0.6434531923890092], [-0.9988081975410994, 0.039553416742988796, -0.027085673649872546, 0.009169407373883585])

        self.joint_A = [-0.07225411493736401, 0.7042132443777533, -0.762403663134082, -1.9573604556635804, 0.689735511051284, 2.3939630041746365, -0.4795062159217066]
        
        self.pose_A_tuple = ([0.42286654805779356, -0.43881502835294306, 0.19661691760359146],  \
                             [0.9998739918297471, 0.012466862743398636, -0.0010366787916414417, 0.009772568386409755])
        
        # Add offset to the z coordinate upwards
        self.pose_A_tuple[0][2] += self.offset_pushing 
        
        self.current_pose =                 None
        self.current_joint_states =         None
        self.pose_list =                    []
        self.force_values =                 []
        
        # Creation of the list of locations and forces
        self.pose_list =                    self.point_generation()
        self.force_values =                 self.force_generation()
        
        self.current_force_msg =            None
        self.current_controller =           None
        self.received_force_messages =      []
        self.rate =                         rospy.Rate(10)  # 10 Hz

        # Names and IDs of the available robot controllers.
        self.CONTROLLER_NAME2ID = {
			"torque":               "franka_zero_torque_controller",
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

    # Go to home position in joint space
    def go_home(self):
        joint_goal = self.joint_home
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
    
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
        initial_y = self.pose_A_tuple[0][1]
        for _ in range(self.n_loc):
            pose = Pose()

            # Fixed position for x and z
            pose.position.x = self.pose_A_tuple[0][0]
            pose.position.y = initial_y                         # Set the current y
            pose.position.z = self.pose_A_tuple[0][2]

            # Fixed orientation 
            pose.orientation.x = self.pose_A_tuple[1][0]
            pose.orientation.y = self.pose_A_tuple[1][1]
            pose.orientation.z = self.pose_A_tuple[1][2]
            pose.orientation.w = self.pose_A_tuple[1][3]

            # Add the pose to the list
            self.pose_list.append(pose)

            # Increment y for the next pose
            initial_y += self.steplength

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
    def go_to_A(self):
            joint_goal = self.joint_A
            self.move_group.go(joint_goal, wait=True)
            self.move_group.stop()

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
    
    def travel_AB(self):
        
        for i in range(self.n_loc):
            
            self.enable_controller("panda_leader_cartesian_impedance_controller")       # Check controller used is for MoveIt
            self.go_to_next_loc(pose_goal = self.pose_list[i])                          # Go to next loc point
            
            """
            for j in range(self.n_pushing_actions):
                
                self.enable_controller("my_force_controller")                                               # Switch controller to cartesian force controller            
                self.set_force_publisher(force_z = self.force_values[j])                                    # Publisher ---> Set pushing force : Force_j for pushing_action = j                                                   
                self.set_force_subscriber(duration = self.time_of_pushing + self.time_of_pushing_margin)    # Subscriber ---> Pushing action at current_loc = i for Force_j acting for 15 [s]
                self.recover_force()                                                                        # Policy for ERROR RECOVERY

                self.set_force_publisher(force_z = 0.0)                                                     # Release force applied Force_j = 0
                self.set_force_subscriber(duration = self.time_of_pushing + self.time_of_pushing_margin)    # Subscriber ---> Null force applied at ee acting for 2 [s]
                self.recover_force()

                self.enable_controller("panda_leader_cartesian_impedance_controller")                       # Switch controller to cartesian motion
                self.go_up()                                                                                # Go to upper position to start the next pushing action
            """
            
                                                                                

if __name__ == '__main__':
    robot = FrankaDataCollection()
    #robot = FrankaDataCollection(data_dir="/home/franka/dataset_collection/src/dataset")
    try:
        robot.go_home() # ---> joint space
        robot.travel_AB() # ----> cartesian space
        robot.go_home() # ---> joint space

        moveit_commander.roscpp_shutdown()
    except rospy.ROSInterruptException:
        pass

    