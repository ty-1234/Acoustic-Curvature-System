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
import copy
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
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
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
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        move_group = self.move_group
        
        


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

        # PARAMETERS
        # Contact Location parameters

        self.AD_dist                    =  0.03                                                    # in [m] --- distance to travel
        self.n_loc                      =  4                                                       # n. of contact locations on which pushing action is performed
        self.steplength                 =  self.AD_dist / self.n_loc                               # in [m]

        # Depth location parameters   
        self.n_loc_depth                =  5                                                       # number of depth location
        self.A_depth_path               =  0.0025                                                  # in [m] Local path lenght in depth
        self.stepdepth                  =  self.A_depth_path/self.n_loc_depth                      # in [m] : how much deep robot's moving (SPECIFY up/dpwn)
        self.time_to_nextloc            =  5                                                       # in [s]: time of sleep after reaching next location
        self.locations                  = ['A', 'B', 'C', 'D']                                     # List of characters identifying locations

        # Time pushing parameters
        self.offset_pushing             =  0.001                                                   # in [m] : offset position in z direction (world frame) where pushing action has to start
        self.time_of_pushing            =  15                                                      # in [s] : time spent applying a prescribed force
        self.time_of_pushing_margin     =  1                                                       # in [s] : time waited before (and after) a single pushing action is performed
        self.rebuild_time               =  5                                                       # in [s] : time spent in wating for the 

        self.cycles                     =  30                                                      # time on which travel AB is performed
        self.N_expected                 =  self.n_loc * self.cycles*self.n_loc_depth               # N of all Data points
        self.N_counter                  =  0                                                       # Counter for data points collected
       
        # Scling factors for velocity and acceleration

        self.velocity_factor            =  0.03
        self.accel_factor               =  0.03


        self.move_group.set_max_velocity_scaling_factor(self.velocity_factor)      
        self.move_group.set_max_acceleration_scaling_factor(self.accel_factor)

        # Pilz industrial motion planner
        #self.move_group.set_planner_id("PTP")  # Point-To-Point motion planning (replace with LIN or CIRC if needed)
        #self.move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")


        # HOME : joints & ee_pose (position, orientation)
        self.joint_home = [0.06841830223171314, -0.41813771533517224, -0.12852136230674158, -1.9645842290600706, -0.03316074410412047, 1.6035182970937927, 0.8142089765552016]
        self.pose_home = ((0.4223942641978751, -0.04042834717301551, 0.6434531923890092), (-0.9988081975410994, 0.039553416742988796, -0.027085673649872546, 0.009169407373883585))
      
        #self.tuple_A = ([0.423388229572494, -0.43235706624340936, 0.19891670799361205], \
        #                [0.9998784426635428, 0.0036294142785418, 0.006485625517795244, 0.013706345624055689])

        self.tuple_A =([0.42421536353883255, -0.43269371706923415, 0.19480828311366946],\
                       [0.9994694985353553, -0.026368557482317014, 0.016344030238721, 0.009914300244743338])      

        #self.tuple_A[0][2] += self.offset_pushing              # Add offset to the z coordinate upwards

        self.pose_A  = self.tuple_to_pose(self.tuple_A)
        print("Pose A:\n")
        print(self.pose_A)
        #print("/////////////////////////////////////////////")                           
        self.current_pose                   = None
        self.current_joint_states           = None
        self.list_of_loc                    = []
        self.joint_states_AB                = []

        # Creation of list of of contact location poses : A,B,C,D respectively
        self.list_of_loc                    = self.create_listof_loc(self.pose_A, self.steplength, self.n_loc)
        #print("List of poses for location:\n")
        #self.print_listofPose(self.list_of_loc, self.n_loc)

        self.loc_A_depths                    = []
        self.loc_B_depths                    = []
        self.loc_C_depths                    = []
        self.loc_D_depths                    = []

        #print("List of poses for depths:\n")
        #print("Print Pose 0:\n")
        self.loc_A_depths                    = self.create_listof_depth(self.list_of_loc[0], self.stepdepth, self.n_loc_depth)
        #self.print_listofPose(self.loc_A_depths, self.n_loc_depth)

        self.loc_B_depths                    = self.create_listof_depth(self.list_of_loc[1], self.stepdepth, self.n_loc_depth)
        #self.print_listofPose(self.loc_B_depths, self.n_loc_depth)

        self.loc_C_depths                    = self.create_listof_depth(self.list_of_loc[2], self.stepdepth, self.n_loc_depth)
        #self.print_listofPose(self.loc_C_depths, self.n_loc_depth)

        self.loc_D_depths                    = self.create_listof_depth(self.list_of_loc[3], self.stepdepth, self.n_loc_depth)
        #self.print_listofPose(self.loc_D_depths, self.n_loc_depth)

        #print(type(self.loc_C_depth))

        self.current_controller             = None
        self.rate                           = rospy.Rate(10)  # 10 Hz

        #self.list_of_loc                    = self.Pose_list_creation(self.pose_list_loc, self.n_loc)
        #print(self.list_of_loc)

        """
        self.list_A_depth                   = self.Pose_list_creation(self.loc_A_depth, self.n_loc_depth)      
        self.list_B_depth                   = self.Pose_list_creation(self.loc_B_depth, self.n_loc_depth)
        self.list_C_depth                   = self.Pose_list_creation(self.loc_C_depth, self.n_loc_depth)
        self.list_D_depth                   = self.Pose_list_creation(self.loc_D_depth, self.n_loc_depth)
        """
        
        #print(type(self.list_C_depth))

        # Names and IDs of the available robot controllers.
        self.CONTROLLER_NAME2ID = {
			"torque":               "franka_zero_torque_controller",
			"position":             "position_joint_trajectory_controller",
			"impedance":            "panda_leader_cartesian_impedance_controller",
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

    def display_joint_values(self):
        joint_goal = self.move_group.get_current_joint_values()
        print("Current Joint Values:", joint_goal)

    # Go to home position in joint space
    def go_home(self):
        rospy.loginfo("Going to HOME position")
        joint_goal = self.joint_home
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
        rospy.loginfo("HOME position achieved!")
    
    # Go to joint staes goal
    def go_joint(self, joint_goal):
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

    # Prints a list of Pose objects
    def print_listofPose(self, list, n_elem):
        for i in range(n_elem):
            print(f"Pose {i}")
            print(list[i])
            print("\n--------------------------------\n")
    
    # Given ee as a tuple, it returns as a Pose object
    def tuple_to_pose(self, position_orientation):
        """
        Converts a tuple with position and orientation to a Pose object.
        
        Args:
            position_orientation (tuple): A tuple of the form 
                ([x, y, z], [x, y, z, w])
                where the first list is the position and the second list 
                is the orientation quaternion.
        
        Returns:
            Pose: A Pose object containing the given position and orientation.
        """
        position, orientation = position_orientation
        pose = Pose()
        pose.position = Point(*position)  # Unpack [x, y, z] into Point
        pose.orientation = Quaternion(*orientation)  # Unpack [x, y, z, w] into Quaternion
        return pose

    #Save the current joint state to a private member variable and publishes the result
    def log_position(self):      
        self.current_joint_states = self.move_group.get_current_joint_values() 
        rospy.loginfo(f"Position logged: Joint state saved to current_joint_states")

    # This function is called whenever a new PoseStamped message is received   
    def pose_callback(self, msg):
        self.current_pose = msg
        rospy.loginfo("Received Pose: %s", self.current_pose)
    
    # Create a list of pose for all contact locations based on a reference
    def create_listof_loc(self, ref, increment, n_loc):        
        list_of_pos = []
        for i in range(n_loc):
            pose = copy.deepcopy(ref)
            pose.position.y +=  i*increment
            list_of_pos.append(pose)
        #print(list_of_pos)
        return list_of_pos

    # Create a list of depth for the selected contact location
    def create_listof_depth(self, ref, increment, n_loc_depth):
        list_of_depths      = []                      # list of tuple-pose 
        increment           = -increment              # change sign since robot has to move DOWN
        for i in range(n_loc_depth):
            pose = copy.deepcopy(ref)
            pose.position.z += i*increment
            list_of_depths.append(pose)

        return list_of_depths
       
    # Move to a Cartesian Pose Goal
    def go_to_cartesian_pose(self, pose_goal):         	    
        self.move_group.set_pose_target(pose_goal)          
        success = self.move_group.go(wait=True)
        self.move_group.clear_pose_targets() 
        return success
    
    # Go to A (reference) ---> joint space
    def go_to_A(self):
        #joint_goal = self.joint_A
        #self.move_group.go(joint_goal, wait=True)
        #self.move_group.stop()
        self.go_to_cartesian_pose(self.pose_A)

    # Go to a pose goal ---> Cartesian space
    def go_to_next_loc (self, pose_goal):            
            success = self.go_to_cartesian_pose(pose_goal)
            print("Next position achieved !")
            self.display_current_pose()
            self.move_group.clear_pose_targets()
            return success

        # Go to the upper position (current x,y, position and orientation), but upper in z direction ----> World frame

    def go_up(self):

        tuple                     =  self.get_ee_pose_tuple()
        tuple [0][2]              =  self.pose_A_tuple[0][2]    # adjust pose on z direction as for A
        pose_goal                 =  self.create_pose(tuple)  
        
        self.move_group.set_pose_target(pose_goal)              # Move upwards

        success                   =  self.move_group.go(wait=True)   # Check if desired pose is achieved                    
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
        # rospy.init_node('force_publisher_node', anonymous=True)
        pub = rospy.Publisher('/my_cartesian_force_controller/target_wrench', WrenchStamped, queue_size=10)
        rate = self.rate  # Publish at 10 Hz

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
        # rospy.init_node('force_subscriber_node', anonymous=True)                                                # Creation of the subscriber node

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
        pub = rospy.Publisher('/franka_control/error_recovery/goal', ErrorRecoveryActionGoal, queue_size=10)
        recovery_goal = ErrorRecoveryActionGoal()
        pub.publish(recovery_goal)
        rospy.loginfo("Published recovery goal.")
          
    # Plan travel for going from A to B using list of points
    def travel_AB(self):

        loc_list                = self.list_of_loc
        n_loc                   = self.n_loc
        n_depth                 = self.n_loc_depth
        depth_list              = [self.loc_A_depths, self.loc_B_depths, self.loc_C_depths, self.loc_D_depths]                  # Define the list of depths for each location
        time_of_pushing         = self.time_of_pushing
        rebuild_time            = self.rebuild_time
        time_to_next_loc        = self.time_to_nextloc
        locations               = self.locations

        # Set a linear motion
        #self.move_group.set_planner_id("LIN")

        for i, loc in enumerate(locations, start = 0):
            

            #------------GOING TO LOCATION {i} -----------------------------    
            start_time_loc = rospy.Time.now()
            success = self.go_to_next_loc(pose_goal = loc_list[i])                                                              # Go to next location
            if success:
                rospy.loginfo(f"LOCATION {loc} achieved: {loc_list[i]}.\nWaiting for {time_to_next_loc} seconds")
                rospy.sleep(time_to_next_loc)                                                                                   # Waiting for 5 [s]       
            else:
                rospy.logwarn(f"Failed to achieve LOCATION {loc}:\n{loc_list[i]}")       
            end_time_loc = rospy.Time.now() - start_time_loc
            print(f"Time spent for reaching LOCATION {loc} + sleep: {end_time_loc.to_sec()} [s]")

            #-----------GOING to ALL DEPTH of LOCATION {i} + GOING UP ------------------
            for j in range(n_depth):
                start_time_depth = rospy.Time.now()
                success = self.go_to_next_loc(pose_goal = depth_list[i][j])                                                     # Go to next location
                if success:
                    rospy.loginfo(f"DEPTH {j} achieved realted to LOCATION {loc}:\n{depth_list[i][j]}.\nWaiting for {time_of_pushing} [s]")
                    rospy.sleep(time_of_pushing)                                                                                # Wait for 15 [s]                                                              
                else:
                    rospy.logwarn(f"Failed to achieve DEPTH {j}:\n{depth_list[i][j]}")
                end_time_depth = rospy.Time.now() - start_time_depth
                print(f"Time spent for reaching DEPTH {j} in LOCATION {loc} (going down): {end_time_depth.to_sec()} [s]")

                # ------GOING  UP ---------------#
                start_time_goup = rospy.Time.now()
                success = self.go_to_next_loc(pose_goal = loc_list[i])                                                                              
                if success:
                    rospy.loginfo(f"LOCATION {loc} achieved for GOING UP: {loc_list[i]}.\nWaiting for {rebuild_time} [s]")                 
                    rospy.sleep(rebuild_time)                                                                                   # Wait for 5 [s]
                else:
                    rospy.logwarn(f"Failed to achieve LOCATION {loc} on GOING UP: \n{loc_list[i]}")
                end_time_goup = rospy.Time.now() - start_time_goup
                print(f"Time spent for GOING UP + rebuild time in LOCATION {loc}: {end_time_goup.to_sec()} [s]")

            joint_goal = self.move_group.get_current_joint_values()                                                             # Record joint states  
            self.joint_states_AB.append(joint_goal)

            

if __name__ == '__main__':
    robot = FrankaDataCollection()
    #robot = FrankaDataCollection(data_dir="/home/franka/dataset_collection/src/dataset")
    try:
        #robot.display_current_pose()
        #robot.display_joint_values()
        start_time = time.time()  
        robot.go_home() # ---> joint space
        time_gohome = time.time() - start_time
        print(f"Time for GOING HOME: {time_gohome} [s]")


        #robot.go_to_A()
        start_time_travel_AB = time.time()  
        robot.travel_AB() # ----> cartesian space
        time_travel_AB = time.time() - start_time_travel_AB
        print(f"Time for GOING HOME: {time_travel_AB} [s]")

        robot.go_home() # ---> joint space
        print(f"Overall time: {time.time() - start_time} [s]")

        moveit_commander.roscpp_shutdown()
    except rospy.ROSInterruptException:
        pass

    