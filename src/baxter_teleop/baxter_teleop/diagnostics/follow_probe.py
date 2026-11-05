"""Try the follow sliders on a held pose, before any policy: how soft does the arm get, and does it stay
calm. Follow is explained in inputs/steer.py.

    python -m baxter_teleop.diagnostics.follow_probe --arm left
    ros2 run rqt_reconfigure rqt_reconfigure     (follow_probe: steer_follow_arm, steer_follow_wrist)

Plays the pose the arm is in through the stream a plan plays through, so a squeeze keeps position control.
Nothing else may command the arm: stop the teleop first. Squeeze the cuff, push, and raise a slider a little
at a time. Letting go takes the share off, so let go at the first buzz. Stops on Ctrl-C, on Baxter's contact
safety, or when the arm leaves its setpoint; the arm then returns to its normal position hold. Saves the
samples: time, squeezed, shares of s0 and w0, joints (7), hand torque (7).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

import baxter_interface

from ..inputs.cuff import Cuff
from ..inputs.steer import CuffSteer
from ..outputs.raw_position import JointCommandPublisher
from ..outputs.replay_stream import ReplayStream
from ..settings import joint_names

RATE = 100.0  # Hz, of the samples


class Pose:
    """A trajectory that stays at q."""

    def __init__(self, q):
        self.q = q

    def __call__(self, t):
        return self.q, np.zeros(7), np.zeros(7)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', default='left', choices=('left', 'right'))
    p.add_argument('--out', type=Path, default=Path('follow_probe.npz'))
    args = p.parse_args()
    rclpy.init()
    node = Node('follow_probe')
    limb = baxter_interface.Limb(args.arm, node)
    names = joint_names(args.arm)
    clock = lambda: node.get_clock().now().nanoseconds * 1e-9  # noqa: E731
    stream = ReplayStream(JointCommandPublisher(node, args.arm, names, limb.joint_names()), clock)
    steer = stream.offset = CuffSteer(node, args.arm, Cuff(node, args.arm), stream)
    while steer.q_measured is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    q0 = steer.q_measured.copy()
    samples, start, next_sample = [], time.monotonic(), 0.0
    stream.play(Pose(q0), clock())
    stream.start()
    print('Holding this pose. Squeeze, push, raise the sliders slowly. Ctrl-C to stop.', flush=True)
    try:
        while not steer.tripped and not steer.contact:
            rclpy.spin_once(node, timeout_sec=0.002)
            steer.live(steer.q_measured)
            now = time.monotonic() - start
            if now >= next_sample:
                next_sample = now + 1.0 / RATE
                held = steer.held
                samples.append([now, held, *steer.follow.share[[0, 4]], *steer.q_measured, *steer.hand_torque()])
    except KeyboardInterrupt:
        pass
    finally:
        print('Stopped: contact safety' if steer.contact else 'Stopped', flush=True)
        stream.close()
        limb.exit_control_mode()
    np.savez(args.out, samples=np.array(samples), q0=q0)
    print(f'Samples: {args.out.resolve()}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
