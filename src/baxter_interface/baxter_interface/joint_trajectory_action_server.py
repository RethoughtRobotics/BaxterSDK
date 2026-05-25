import argparse
import sys
import time

import rclpy
from joint_trajectory_action.joint_trajectory_action import JointTrajectoryActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.utilities import remove_ros_args


def main():
    rclpy.init()

    arg_fmt = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=arg_fmt)
    parser.add_argument(
        '-l', '--limb', dest='limb', default='both',
        choices=['both', 'left', 'right'],
        help='joint trajectory action server limb',
    )
    parser.add_argument(
        '-r', '--rate', dest='rate', default=100.0,
        type=float, help='trajectory control rate (Hz)',
    )
    parser.add_argument(
        '-m', '--mode', default='position_w_id',
        choices=['position_w_id', 'position', 'velocity'],
        help='control mode for trajectory execution',
    )
    parser.add_argument(
        '-i', '--interpolation', default='bezier',
        choices=['bezier', 'bezier_with_velocity', 'minjerk'],
        help='interpolation method for trajectory generation',
    )
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])

    node = rclpy.create_node('baxter_joint_trajectory_action_server')
    log = node.get_logger()

    log.info('Waiting for robot joint states...')
    log.info('Limbs ready')

    limbs = ['left', 'right'] if args.limb == 'both' else [args.limb]
    jtas = []
    for limb in limbs:
        jtas.append(JointTrajectoryActionServer(limb, mode=args.mode, node=node,
                                                rate=args.rate, interpolation=args.interpolation))

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    log.info('Baxter joint trajectory action server running (mode: %s)' % args.mode)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for j in jtas:
            j.clean_shutdown()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
