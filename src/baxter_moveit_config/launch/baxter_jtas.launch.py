"""
Baxter real-robot bringup with Python JTAS (no ros2_control).

Equivalent to the ROS1 baxter_moveit_config launch:
  - robot_state_publisher  (URDF, joint_states from bridge)
  - world->base static transform
  - Python joint_trajectory_action_server  (left + right arms)
  - move_group  (MoveItSimpleControllerManager -> Python JTAS)
  - RViz with MoveIt plugin
  - warehouse DB  (optional, off by default)

Prereqs:
  docker run --rm --network=host baxter_bridge:latest

Usage:
  ros2 launch baxter_moveit_config baxter_jtas.launch.py
  ros2 launch baxter_moveit_config baxter_jtas.launch.py use_rviz:=false
  ros2 launch baxter_moveit_config baxter_jtas.launch.py use_warehouse:=true
"""

from ament_index_python.packages import get_package_share_directory
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
    baxter_interface_share = get_package_share_directory('baxter_interface')

    simple_controllers_yaml = str(launch_package_path / 'config' / 'simple_controllers.yaml')
    jtas_params_yaml = str(baxter_interface_share + '/config/param.yaml')

    ld = LaunchDescription()

    ld.add_action(DeclareLaunchArgument('use_rviz', default_value='true'))
    ld.add_action(DeclareLaunchArgument('use_warehouse', default_value='false'))

    # ---------- TF ----------

    # world -> base (from SRDF virtual joint)
    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_package_path / 'launch/static_virtual_joint_tfs.launch.py'))
        )
    )

    # ---------- Robot state publisher ----------

    # Reads joint states from the Zenoh bridge, not from ros2_control
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

    # ---------- Python JTAS ----------

    # Hosts FollowJointTrajectory action servers at:
    #   robot/limb/left/follow_joint_trajectory
    #   robot/limb/right/follow_joint_trajectory
    ld.add_action(
        Node(
            package='baxter_interface',
            executable='joint_trajectory_action_server',
            name='baxter_joint_trajectory_action_server',
            output='screen',
            parameters=[jtas_params_yaml],
        )
    )

    # ---------- MoveIt move_group ----------

    ld.add_action(
        Node(
            name='move_group',
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            remappings=[('/joint_states', '/robot/joint_states')],
            arguments=[
                '--ros-args',
                '--log-level',
                'moveit.ros.planning_scene_monitor:=ERROR',
            ],
            parameters=[
                moveit_config.to_dict(),
                simple_controllers_yaml,
                {
                    # Use Python JTAS, not ros2_control
                    'moveit_controller_manager': 'moveit_simple_controller_manager/MoveItSimpleControllerManager',
                    'moveit_manage_controllers': False,
                    'allow_trajectory_execution': True,
                    'publish_robot_description_semantic': True,
                    'publish_planning_scene': True,
                    'publish_geometry_updates': True,
                    'publish_state_updates': True,
                    'publish_transforms_updates': True,
                    'monitor_dynamics': False,
                    'trajectory_execution.allowed_execution_duration_scaling': 2.0,
                    'trajectory_execution.allowed_goal_duration_margin': 2.0,
                    'trajectory_execution.allowed_start_tolerance': 0.1,
                },
            ],
        )
    )

    # ---------- RViz ----------

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_package_path / 'launch/moveit_rviz.launch.py')),
            condition=IfCondition(LaunchConfiguration('use_rviz')),
        )
    )

    # ---------- Warehouse DB (optional) ----------

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(launch_package_path / 'launch/warehouse_db.launch.py')),
            condition=IfCondition(LaunchConfiguration('use_warehouse')),
        )
    )

    return ld
