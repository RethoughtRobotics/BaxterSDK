#!/usr/bin/env python3

"""Drive the teleop with the key presses recorded in a bag, at their original timing.

Same operator input through the current code, so a recording of the replay compares directly
with the original bag (analyze_teleop_bag.sh <original> <replay>). The teleop is built with the
original session's arm, frame and speeds; start the robot (or sim) from the same pose.

  python -m baxter_interface.baxter_teleop.key_replay <bag>
"""

import argparse
import time

import rclpy

from .bag_report import load
from .cartesian_delta_teleop import CartesianDeltaTeleop


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag')
    args, ros_args = parser.parse_known_args()
    recorded = load(args.bag)
    keys = recorded['key']
    session = recorded['config'].get('args', {})
    rclpy.init(args=['key_replay', *ros_args])
    teleop = CartesianDeltaTeleop(
        arm=recorded['arm'],
        mode=session.get('mode', 'velocity'),
        frame=session.get('frame', 'base'),
        linear_speed=session.get('linear_speed'),
        angular_speed=session.get('angular_speed'),
        key_hold_initial=session.get('key_hold_initial', 0.55),
    )
    teleop.publish_config({'key_replay_of': args.bag, 'args': session})
    start = time.monotonic() - keys[0][0] + 1.0  # first key 1 s after the teleop is ready
    pending = iter(keys + [(keys[-1][0] + 3.0, '\x1b')])  # then settle 3 s and quit (ESC)
    upcoming = next(pending)

    def recorded_key(timeout):  # stands in for getch
        nonlocal upcoming
        if time.monotonic() - start < upcoming[0]:
            time.sleep(timeout)
            return None
        key, upcoming = upcoming[1], next(pending, (float('inf'), None))
        return key

    teleop.key_source = recorded_key
    print(f'Replaying {len(keys)} keys from {args.bag} over {keys[-1][0] - keys[0][0]:.0f} s')
    try:
        teleop.spin()
    finally:
        teleop.node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
