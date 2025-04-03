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
from geometry_msgs.msg import PoseStamped, Pose, Point
from actionlib_msgs.msg import GoalStatusArray
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from moveit_commander.conversions import pose_to_list
from pathlib import Path

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

        # Initialize moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)

        # Initialize node
        rospy.init_node("calibration_node", anonymous=True)

        # Initialize tf listener
        robot = moveit_commander.RobotCommander()
        # Inizitalize the planning scene
        scene = moveit_commander.PlanningSceneInterface()
        
        # Initialize the move group for the panda_arm
        group_name = "panda_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)


        #self.pose_publisher = rospy.Publisher('/generated_pose', PoseStamped, queue_size=10)

        # Subscribe to the PoseStamped topic (adjust topic name as needed)
        #self.pose_subscriber = rospy.Subscriber('/current_pose', PoseStamped, self.pose_callback)

        # Defines the topic on which the robot publishes the trajectory
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size=20)
        self.planning_frame = move_group.get_planning_frame() #get the planning frame
        self.eef_link = move_group.get_end_effector_link() # get ee frame 


        # self.move_group.set_end_effector_link("panda_link8") # sets the ee-link to panda_link8
        #self.group_names = self.robot.get_group_names()
        # Set the planner publiser
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

 
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")


        # Misc variables
        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        # Joint names
        self.joint_names = {
            "joint1": "panda_joint1",
            "joint2": "panda_joint2",
            "joint3": "panda_joint3",
            "joint4": "panda_joint4",
            "joint5": "panda_joint5",
            "joint6": "panda_joint6",
            "joint7": "panda_joint7"
        }


        # PATH cooverage parameters
        self.AB_dist = 0.03 # in [m] ---
        self.n_samples =100
        self.steplength = self.AB_dist / self.n_samples # in [m]
        self.cycles = 30

        self.n_datapoints = self.n_samples * self.cycles

        self.offset_pushing = 0.001 # in [m]
        self.time_of_pushing = 2 # in [s]
        self.n_pushingactions = 0
        self.n_datapoints_collected = 0
        

        # HOME : joints & ee_pose (position, orientation)

        self.joint_home = [0.06841830223171314, -0.41813771533517224, -0.12852136230674158, -1.9645842290600706, -0.03316074410412047, 1.6035182970937927, 0.8142089765552016]
        self.pose_home = ([0.4223942641978751, -0.04042834717301551, 0.6434531923890092], [-0.9988081975410994, 0.039553416742988796, -0.027085673649872546, 0.009169407373883585])

        self.joint_A = [-0.07225411493736401, 0.7042132443777533, -0.762403663134082, -1.9573604556635804, 0.689735511051284, 2.3939630041746365, -0.4795062159217066]
        self.pose_A_tuple = ([0.42286654805779356, -0.43881502835294306, 0.19661691760359146], [0.9998739918297471, 0.012466862743398636, -0.0010366787916414417, 0.009772568386409755])
        self.pose_A_tuple[0][2] += self.offset_pushing # add offset to the z coordinate upwards
        

        self.current_pose = None
        self.pose_list = []
        self.pose_list = self.point_generation()
        
        
        # Saving directories and parameters
        # Define the path to a directory

        #self.DATA_DIR = data_dir
        #directory_path = Path("/home/your_username/your_directory")

    
    # Get the current ee-pose
    # Returns the position and orientation of the end-effector in a tuple
    def get_ee_pose_tuple(self):
        
        self.current_pose = self.move_group.get_current_pose().pose
        
        rospy.loginfo("End-Effector Position: x={}, y={}, z={}".format(self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z))
        rospy.loginfo("End-Effector Orientation: x={}, y={}, z={}, w={}".format(self.current_pose.orientation.x, self.current_pose.orientation.y, self.current_pose.orientation.z, self.current_pose.orientation.w))
        
        return [self.current_pose.position.x, self.current_pose.position.y, self.current_pose.position.z], [self.current_pose.orientation.x, self.current_pose.orientation.y, self.current_pose.orientation.z, self.current_pose.orientation.w]

    def display_current_pose(self):
        pose = self.move_group.get_current_pose().pose
        rospy.loginfo("End-Effector Position: x={}, y={}, z={}".format(pose.position.x, pose.position.y, pose.position.z))
        rospy.loginfo("End-Effector Orientation: x={}, y={}, z={}, w={}".format(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))

    # Go to home position in joint space
    def go_home(self):
        joint_goal = self.joint_home
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()
    
    # Create an  STAMPED ee-pose in R^3 from a tuple of position and orientation
    # Stamped pose contains: header (frame_id, stamp, seq) and pose (position, orientation)
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
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
        ))
        rospy.loginfo("End-Effector Orientation: x={}, y={}, z={}, w={}".format(
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w
        ))

        return pose
    
    def log_position(self):
        #Save the current joint state to a private member variable
        self.grid_joint_state = self.group.get_current_joint_values() 
        rospy.loginfo(f"Position logged: Joint state saved to grid_joint_state")
    
    def pose_callback(self, msg):
        # This function is called whenever a new PoseStamped message is received
        self.current_pose = msg
        rospy.loginfo("Received Pose: %s", self.current_pose)
    
    def point_generation(self):
        
        initial_y = self.pose_A_tuple[0][1]

        for _ in range(self.n_samples):
            pose = Pose()

            # Fixed position for x and z
            pose.position.x = self.pose_A_tuple[0][0]
            pose.position.y = initial_y  # Set the current y
            pose.position.z = self.pose_A_tuple[0][2]

            # Fixed orientation (identity quaternion)
            pose.orientation.x = self.pose_A_tuple[1][0]
            pose.orientation.y = self.pose_A_tuple[1][1]
            pose.orientation.z = self.pose_A_tuple[1][2]
            pose.orientation.w = self.pose_A_tuple[1][3]

            # Add the pose to the list
            self.pose_list.append(pose)

            # Increment y for the next pose
            initial_y += self.steplength

        return self.pose_list

    # Go to A (reference) in joint space
    def go_to_A(self):
        joint_goal = self.joint_A
        self.move_group.go(joint_goal, wait=True)
        self.move_group.stop()

    # Plan travel for going from A to B using list of points
    def travel_AB(self):

        for i in range(self.n_samples):

            pose_goal = self.pose_list[i]
            self.move_group.set_pose_target(pose_goal)

            success = self.move_group.go(wait=True)
            if not success:
                rospy.loginfo("Failed to reach the goal pose")
            
            self.move_group.stop()                                                  # Calling `stop()` ensures that there is no residual movement
            self.display_current_pose()
            self.move_group.clear_pose_targets()
            
            #self.current_pose = self.get_current_pose().pose


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

    