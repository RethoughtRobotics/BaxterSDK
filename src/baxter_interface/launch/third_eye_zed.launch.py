"""Launch a ZED 2i as the third-eye using the composable node interface.

Requires the zed-ros2-wrapper built in this workspace (clone into src/):
  cd baxter_sdk && git clone -b kilted https://github.com/RethoughtRobotics/zed-ros2-wrapper.git src/zed-ros2-wrapper
  cd baxter_sdk && git clone -b kilted https://github.com/RethoughtRobotics/zed-ros2-interfaces.git src/zed-ros2-interfaces
  colcon build --packages-up-to zed_wrapper zed_ros2
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    pkg = get_package_share_directory('baxter_interface')
    zed_pkg = get_package_share_directory('zed_wrapper')

    override_config = os.path.join(pkg, 'config', 'third_eye_zed.yaml')
    common_config = os.path.join(zed_pkg, 'config', 'common_stereo.yaml')
    camera_config = os.path.join(zed_pkg, 'config', 'zed2i.yaml')

    container = ComposableNodeContainer(
        name='zed_container',
        namespace='cameras/third_eye',
        package='rclcpp_components',
        executable='component_container_isolated',
        output='screen',
    )

    zed_node = ComposableNode(
        package='zed_components',
        namespace='cameras/third_eye',
        plugin='stereolabs::ZedCamera',
        name='zed_node',
        parameters=[
            common_config,
            camera_config,
            override_config,
        ],
        remappings=[
            ('zed_node/left/image_rect_color', 'color/image_raw'),
            ('zed_node/left/camera_info', 'color/camera_info'),
        ],
        extra_arguments=[{'use_intra_process_comms': True}],
    )

    load_zed = LoadComposableNodes(
        target_container='/cameras/third_eye/zed_container',
        composable_node_descriptions=[zed_node],
    )

    return LaunchDescription([container, load_zed])
