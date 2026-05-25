import time

import rclpy
from joint_trajectory_action.joint_trajectory_action import JointTrajectoryActionServer
from rclpy.executors import MultiThreadedExecutor


def main():
    rclpy.init()
    node = rclpy.create_node('baxter_joint_trajectory_action_server')
    log = node.get_logger()

    log.info('Waiting for robot joint states...')
    log.info('Limbs ready')

    JointTrajectoryActionServer('left', mode='position', node=node)
    JointTrajectoryActionServer('right', mode='position', node=node)

    # Warm up: let publishers discover the bridge
    log.info('Warming up (5s)...')
    t0 = time.time()
    while time.time() - t0 < 5.0:
        rclpy.spin_once(node, timeout_sec=0.05)

    # SingleThreadedExecutor — everything on one node.
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
