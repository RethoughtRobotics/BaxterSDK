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
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

import baxter_interface

from ..inputs.cuff import Cuff
from ..inputs.steer import CuffSteer
from ..outputs.raw_position import JointCommandPublisher
from ..outputs.replay_stream import ReplayStream
from ..settings import joint_names

RATE = 100.0  # Hz, of the samples
SPIN_WAIT = 0.1  # s, of the executor's own thread


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
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # Ctrl-C must still reach the arm's hold
    node = Node('follow_probe')
    limb = baxter_interface.Limb(args.arm, node)
    names = joint_names(args.arm)
    clock = lambda: node.get_clock().now().nanoseconds * 1e-9  # noqa: E731
    stream = ReplayStream(JointCommandPublisher(node, args.arm, names, limb.joint_names()), clock)
    steer = stream.offset = CuffSteer(node, args.arm, Cuff(node, args.arm), stream)
    # Its own thread: following acts on the newest joint angles and the sliders need the parameter services,
    # and one callback per wait (rclpy.spin_once) falls behind Baxter's state topics.
    executor = rclpy.get_global_executor()
    executor.add_node(node)
    running = threading.Event()
    running.set()

    def spin():
        while running.is_set():
            executor.spin_once(timeout_sec=SPIN_WAIT)

    spinner = threading.Thread(target=spin, daemon=True)
    spinner.start()
    while steer.q_measured is None:
        time.sleep(SPIN_WAIT)
    q0 = steer.q_measured.copy()
    samples, start = [], time.monotonic()
    stream.play(Pose(q0), clock())
    stream.start()
    print('Holding this pose. Squeeze, push, raise the sliders slowly. Ctrl-C to stop.', flush=True)
    try:
        while not steer.tripped and not steer.contact:
            steer.live(steer.q_measured)
            now = time.monotonic() - start
            samples.append([now, steer.held, *steer.follow.share[[0, 4]], *steer.q_measured, *steer.hand_torque()])
            time.sleep(1.0 / RATE)
    except KeyboardInterrupt:
        pass
    finally:
        print('Stopped: contact safety' if steer.contact else 'Stopped', flush=True)
        stream.close()
        limb.exit_control_mode()
        running.clear()
        spinner.join(timeout=1.0)
    np.savez(args.out, samples=np.array(samples), q0=q0)
    print(f'Samples: {args.out.resolve()}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
