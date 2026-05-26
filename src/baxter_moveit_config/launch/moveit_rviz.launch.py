from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder('baxter', package_name='baxter_moveit_config').to_moveit_configs()

    rviz_config = str(moveit_config.package_path / 'config' / 'moveit.rviz')

    return LaunchDescription([
        Node(
            package='rviz2',
            executable='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[
                moveit_config.robot_description,
                moveit_config.robot_description_semantic,
                moveit_config.robot_description_kinematics,
                moveit_config.planning_pipelines,
                moveit_config.joint_limits,
            ],
            # MoveGroupInterface inside RViz subscribes to 'joint_states' directly
            # (used by Cartesian path and current state monitor).
            # Remap to match the Zenoh bridge's actual topic.
            remappings=[('/joint_states', '/robot/joint_states')],
        )
    ])
