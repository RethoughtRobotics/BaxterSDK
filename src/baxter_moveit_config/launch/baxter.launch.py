"""
Baxter real-robot bringup.

Assumes the baxter_bridge Docker container is already running:
  docker run --rm --network=host baxter_bridge:latest .

Usage:
  ros2 launch baxter_moveit_config baxter.launch.py
  ros2 launch baxter_moveit_config baxter.launch.py use_rviz:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder('baxter', package_name='baxter_moveit_config').to_moveit_configs()
    launch_package_path = moveit_config.package_path

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument('use_rviz', default_value='true'))

    # world -> base static transform
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_package_path / 'launch/static_virtual_joint_tfs.launch.py'))
        )
    )

    # Connect the bridge's reference/* TF tree to the main tree.
    # Baxter publishes a "reference" (commanded) state as a separate TF tree
    # rooted at reference/base. It's at the same physical location as base,
    # so a zero-offset static transform is correct. Without this, move_group's
    # planning_scene_monitor floods the log with transform warnings.
    ld.add_action(
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'base', 'reference/base'],
        )
    )

    # Robot state publisher fed by real joint states.
    # ignore_timestamp avoids TF_OLD_DATA conflicts with the bridge's own TF.
    ld.add_action(
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[
                moveit_config.robot_description,
                {'ignore_timestamp': True},
            ],
            remappings=[('/joint_states', '/robot/joint_states')],
        )
    )

    ld.add_action(
        Node(
            name='move_group',
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            remappings=[('/joint_states', '/robot/joint_states')],
            parameters=[
                moveit_config.to_dict(),
                {
                    'publish_robot_description_semantic': True,
                    'allow_trajectory_execution': True,
                    'publish_planning_scene': True,
                    'publish_geometry_updates': True,
                    'publish_state_updates': True,
                    'publish_transforms_updates': True,
                    'monitor_dynamics': False,
                    'moveit_manage_controllers': False,
                    'trajectory_execution.allowed_execution_duration_scaling': 1.5,
                    'trajectory_execution.allowed_goal_duration_margin': 1.0,
                    'trajectory_execution.allowed_start_tolerance': 0.1,
                },
            ],
        )
    )

    # RViz with MoveIt plugin
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_package_path / 'launch/moveit_rviz.launch.py')),
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        )
    )

    return ld
