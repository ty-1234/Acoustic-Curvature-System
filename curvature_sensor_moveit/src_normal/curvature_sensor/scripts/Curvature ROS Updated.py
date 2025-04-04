
#!/usr/bin/env python3

import os
import sys
import rospy
import csv
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from moveit_commander import RobotCommander, MoveGroupCommander, PlanningSceneInterface, roscpp_initialize

class FrankaDataCollection:
    def __init__(self):
        roscpp_initialize(sys.argv)
        rospy.init_node("curvature_logger_node", anonymous=True)

        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group_name = "panda_arm"
        self.move_group = MoveGroupCommander(self.group_name)

        self.section_pub = rospy.Publisher('/curving_section', String, queue_size=10)
        self.log_file = os.path.join("csv_data", "ros", "robot_log.csv")
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Position_cm", "PosX", "PosY", "PosZ", "OriX", "OriY", "OriZ", "OriW", "Timestamp"])

    def get_current_pose(self):
        pose = self.move_group.get_current_pose().pose
        return pose

    def log_pose(self, label):
        pose = self.get_current_pose()
        timestamp = datetime.now().isoformat()

        row = [
            label,
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
            timestamp
        ]

        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        rospy.loginfo(f"Logged section {label} at {timestamp}")

    def move_and_log(self):
        section_labels = ["0cm", "1cm", "2cm", "3cm", "4cm", "5cm", "6cm"]
        for i, label in enumerate(section_labels):
            self.section_pub.publish(label)
            rospy.sleep(1.0)  # small delay to ensure broadcast
            self.log_pose(label)
            rospy.sleep(10.0)  # simulate time for robot press

def main():
    robot_logger = FrankaDataCollection()
    robot_logger.move_and_log()

if __name__ == "__main__":
    main()
