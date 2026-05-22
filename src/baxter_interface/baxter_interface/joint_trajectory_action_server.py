import rclpy
from joint_trajectory_action.joint_trajectory_action import JointTrajectoryActionServer
from rclpy.executors import MultiThreadedExecutor

from baxter_interface import Limb


def main():
    rclpy.init()
    node = rclpy.create_node('baxter_joint_trajectory_action_server')
    log = node.get_logger()

    log.info('Waiting for robot joint states...')
    left_limb = Limb('left', node)
    right_limb = Limb('right', node)
    log.info('Limbs ready')

    JointTrajectoryActionServer(
        'left_arm_controller/follow_joint_trajectory',
        [left_limb],
        node,
    )
    JointTrajectoryActionServer(
        'right_arm_controller/follow_joint_trajectory',
        [right_limb],
        node,
    )
    JointTrajectoryActionServer(
        'both_arms_controller/follow_joint_trajectory',
        [left_limb, right_limb],
        node,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    log.info('Baxter joint trajectory action server running')
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
