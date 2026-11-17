"""Launch a ZED camera as the third-eye, publishing on the same topics as the RealSense third-eye.

Requires the zed-ros2-wrapper built in this workspace (clone into src/):
  cd baxter_sdk && git clone -b kilted https://github.com/RethoughtRobotics/zed-ros2-wrapper.git src/zed-ros2-wrapper
  cd baxter_sdk && git clone -b kilted https://github.com/RethoughtRobotics/zed-ros2-interfaces.git src/zed-ros2-interfaces
  colcon build --packages-up-to zed_wrapper
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('baxter_interface')
    zed_pkg = get_package_share_directory('zed_wrapper')

    camera_model = LaunchConfiguration('camera_model')
    serial_number = LaunchConfiguration('serial_number')

    override_config = os.path.join(pkg, 'config', 'third_eye_zed.yaml')
    common_config = os.path.join(zed_pkg, 'config', 'common.yaml')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_model', default_value='zed2i', description='ZED model: zed, zedm, zed2, zed2i, zedx, zedxm'
        ),
        DeclareLaunchArgument(
            'serial_number', default_value='0', description='ZED serial number (0 = first available)'
        ),
        Node(
            package='zed_wrapper',
            namespace='cameras/third_eye',
            executable='zed_wrapper',
            name='zed_node',
            output='screen',
            parameters=[
                common_config,
                override_config,
                {
                    'general.camera_model': camera_model,
                    'general.camera_name': 'third_eye',
                    'general.serial_number': serial_number,
                },
            ],
            remappings=[
                ('zed_node/rgb/image_rect_color', 'color/image_raw'),
                ('zed_node/rgb/camera_info', 'color/camera_info'),
            ],
        ),
    ])
