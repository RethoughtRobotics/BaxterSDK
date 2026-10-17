#!/usr/bin/env bash

# # Source ROS 1 workspace
# source /opt/ros/noetic/setup.bash  
# source ~/rethink_ws/devel/setup.bash

# Source ROS 2 workspace
source /opt/ros/humble/setup.bash  

export ROS_MASTER_URI=http://10.42.0.2:11311
export ROS_IP=10.42.0.1
unset ROS_HOSTNAME

export DISPLAY=:1
export QT_X11_NO_MITSHM=1
export LIBGL_ALWAYS_SOFTWARE=1       