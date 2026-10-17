#!/usr/bin/env bash
export ROS_MASTER_URI=http://10.42.0.2:11311
export ROS_IP=10.42.0.1
unset ROS_HOSTNAME
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/bridge_ws/install/setup.bash
ros2 run ros1_bridge dynamic_bridge --bridge-all-topics
