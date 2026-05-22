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

    # Static transform: world -> base
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_package_path / 'launch/static_virtual_joint_tfs.launch.py'))
        )
    )

    # Robot state publisher — ignore timestamps to avoid TF conflict with bridge
    ld.add_action(
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[
                moveit_config.robot_description,
                {'ignore_timestamp': True},
            ],
            remappings=[('/joint_states', '/robot/joint_states')],
        )
    )

    # move_group — disable controller management since we have no ros2_control stack
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_package_path / 'launch/move_group.launch.py'))
        )
    )

    # RViz
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_package_path / 'launch/moveit_rviz.launch.py')),
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        )
    )

    return ld
