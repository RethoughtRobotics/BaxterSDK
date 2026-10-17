#!/usr/bin/env bash

# Build Rethink workspace
source /opt/ros/noetic/setup.bash
cd ~/rethink_ws

# Build the headers first, then the rest of the workspace
catkin_make --pkg baxter_core_msgs baxter_maintenance_msgs intera_core_msgs  

# Dont think this is needed to make the bridge workwork 
catkin_make

# Source Rethink workspace 
source ~/rethink_ws/devel/setup.bash

# Source ROS 2 workspace with ported Baxter messages, then build the bridge
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/bridge_ws
colcon build --symlink-install --packages-select ros1_bridge --cmake-force-configure
