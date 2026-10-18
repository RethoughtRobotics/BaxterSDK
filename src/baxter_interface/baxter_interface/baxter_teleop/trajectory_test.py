#!/usr/bin/env python3

"""Drive the teleop through a scripted TCP trajectory, for comparing controllers on equal input.

The trajectory is a target like VRTarget and KeyTarget: spin_target servos the TCP onto its goal
(servo_twist, with the path velocity as feedforward), so this exercises the same control code as
VR and the keyboard. It starts from the current TCP (run tuck_arms -u first) and holds the start
orientation. Phases (offsets in the robot base frame, 0.1 m/s, 1 s pauses):

  square x3     20 cm square in x-y, three times: does the configuration return with the hand?
  down and up   20 cm down and back
  elbow limit   35 cm forward and 30 cm out, 9 cm past where the e1 (elbow) limit stops the arm, and back
  circle x2     10 cm radius circle in x-y, twice

  python -m baxter_interface.baxter_teleop.trajectory_test --ik dls|sns [--arm left]
"""

import argparse
import os
import time

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation

from .cartesian_delta_teleop import CartesianDeltaTeleop
from .pose_servo import servo_twist

DT = 0.01
SPEED = 0.1  # m/s
PAUSE = 1.0  # s
SQUARE = [(0.2, 0, 0), (0.2, 0.2, 0), (0, 0.2, 0), (0, 0, 0)]


def protocol():
    """Dense path offsets (n, 3) at DT, and phases [(name, t_start, t_end)]."""
    points, phases = [np.zeros(3)], []

    def line_to(target):
        start, target = points[-1], np.asarray(target, float)
        n = max(1, int(np.ceil(np.linalg.norm(target - start) / (SPEED * DT))))
        points.extend(start + (target - start) * k / n for k in range(1, n + 1))

    def pause():
        points.extend([points[-1]] * int(PAUSE / DT))

    def phase(name, build):
        start = len(points) * DT
        build()
        phases.append((name, start, len(points) * DT))
        pause()

    pause()
    phase('square x3', lambda: [line_to(c) for _ in range(3) for c in SQUARE])
    phase('down and up', lambda: [line_to(c) for c in ((0, 0, -0.2), (0, 0, 0))])
    # Diagonal, away from the body: straight forward the upper elbow meets the torso before e1 does.
    phase('elbow limit', lambda: [line_to(c) for c in ((0.35, 0.3, 0), (0, 0, 0))])

    def circle():
        n = int(2 * 2 * np.pi * 0.1 / (SPEED * DT))
        angle = np.pi + np.linspace(0, 4 * np.pi, n + 1)[1:]
        points.extend(np.c_[0.1 + 0.1 * np.cos(angle), 0.1 * np.sin(angle), np.zeros(n)])

    phase('circle x2', circle)
    return np.asarray(points), phases


class TrajectoryTarget:
    """Goal = start TCP + path offset at the elapsed time, orientation held; done after the end."""

    def __init__(self, offsets, gain=5.0, settle=2.0):
        self.offsets, self.gain, self.settle = offsets, gain, settle
        self.start = self.origin = self.goal = None
        self.done = False

    def reset(self):
        pass

    def twist(self, tcp_position, tcp_rotation):
        now = time.monotonic()
        tcp_rotation = Rotation.from_matrix(tcp_rotation)
        if self.start is None:
            self.start, self.origin = now, (np.asarray(tcp_position, float).copy(), tcp_rotation)
        i = int((now - self.start) / DT)
        if i >= len(self.offsets) + self.settle / DT:
            self.done, self.goal = True, None
            return None
        i = min(i, len(self.offsets) - 1)
        nxt = min(i + 1, len(self.offsets) - 1)
        self.goal = (self.origin[0] + self.offsets[i], self.origin[1])
        feedforward = np.r_[(self.offsets[nxt] - self.offsets[i]) / DT, np.zeros(3)]
        return servo_twist(self.goal, tcp_position, tcp_rotation, self.gain, 4 * SPEED, 1.0, feedforward=feedforward)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ik', choices=['dls', 'sns'], default='dls')
    parser.add_argument('--arm', choices=['left', 'right'], default='left')
    args, ros_args = parser.parse_known_args()
    sns_limits = os.path.join(get_package_share_directory('baxter_interface'), 'config', 'sns_joint_limits.yaml')
    rclpy.init(args=['trajectory_test', *ros_args])
    teleop = CartesianDeltaTeleop(arm=args.arm, mode='velocity', ik=args.ik, sns_limits=sns_limits, frame='base')
    offsets, phases = protocol()
    teleop.publish_config({
        'trajectory_test': {'ik': args.ik, 'speed': SPEED, 'phases': phases},
        'args': {'arm': args.arm, 'input': 'trajectory', 'ik': args.ik},
    })
    target = TrajectoryTarget(offsets)
    teleop.key_source = lambda timeout: '\x1b' if target.done else time.sleep(timeout)
    print(f'{args.ik}: {len(offsets) * DT:.0f} s trajectory, phases ' + ', '.join(p[0] for p in phases))
    try:
        teleop.spin_target(target)
    finally:
        teleop.node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
