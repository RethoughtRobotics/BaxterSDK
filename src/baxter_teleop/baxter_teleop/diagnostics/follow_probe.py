"""Try the follow sliders on a held pose, before any policy: how soft does the arm get, and does it stay
calm. Follow is explained in inputs/steer.py.

    python -m baxter_teleop.diagnostics.follow_probe --arm left
    ros2 run rqt_reconfigure rqt_reconfigure     (follow_probe: steer_follow_arm, steer_follow_wrist)
    python -m baxter_teleop.diagnostics.follow_probe --hold   (the plain hold: no follow, no admittance)
    python -m baxter_teleop.diagnostics.follow_probe --chase 3   (the setpoint chases the arm while held)

With --chase G the held setpoint moves toward the measured joints at G per second times the deflection
(a 3 mrad deadzone keeps a free arm still): the arm goes where the hand pushes it, at a speed set by the
push, and stays where it is left. Zero-g stays suppressed. G starts the live parameters chase_gain_arm
(s0 to e1) and chase_gain_wrist (w0 to w2, WRIST_RATIO times G):
    ros2 param set /follow_probe chase_gain_wrist 40.0

With --hold the setpoint never moves: what the push does to the arm is the position hold's own give, and
the samples' joints minus q0 is that deflection. The same is published live as /follow_probe/deflection
(JointState: position the deflection in mrad, effort the hand torque), for a live plot (rqt_plot).

Plays the pose the arm is in through the stream a plan plays through, so a squeeze keeps position control.
Nothing else may command the arm: stop the teleop first. Squeeze the cuff, push, and raise a slider a little
at a time. Letting go takes the share off, so let go at the first buzz. Stops on Ctrl-C, on Baxter's contact
safety, or when the arm leaves its setpoint; the arm then returns to its normal position hold. Saves the
samples: time, squeezed, shares of s0 and w0, joints (7), hand torque (7), the setpoint (7).
"""

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState

import baxter_interface

from ..inputs.cuff import Cuff
from ..inputs.steer import CuffSteer, deadzone
from ..outputs.raw_position import JointCommandPublisher
from ..outputs.replay_stream import ReplayStream
from ..settings import STEER, joint_names

RATE = 100.0  # Hz, of the samples
CHASE_DEADZONE = np.array(STEER['chase_deadzone'], float)
CHASE = {'chase_gain_arm': slice(0, 4), 'chase_gain_wrist': slice(4, 7)}  # 1/s, live parameters
WRIST_RATIO = float(STEER['chase_gain_wrist']) / float(STEER['chase_gain_arm'])
SPIN_WAIT = 0.1  # s, of the executor's own thread


class Pose:
    """A trajectory that stays at q, moving at qd when the chase moves it: the velocity goes out with the
    setpoint, so the joint controllers and the inverse dynamics feedforward move with the arm instead of
    braking it (a setpoint at rest brakes every motion of the arm)."""

    def __init__(self, q):
        self.q, self.qd = q, np.zeros(7)

    def __call__(self, t):
        return self.q, self.qd, np.zeros(7)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', default='left', choices=('left', 'right'))
    p.add_argument('--out', type=Path, default=Path('follow_probe.npz'))
    p.add_argument('--hold', action='store_true', help='a plain position hold: follow share 0, admittance off')
    p.add_argument('--chase', type=float, default=None, metavar='GAIN', help='1/s: the held setpoint chases the arm')
    args = p.parse_args()
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # Ctrl-C must still reach the arm's hold
    node = Node('follow_probe')
    limb = baxter_interface.Limb(args.arm, node)
    names = joint_names(args.arm)
    clock = lambda: node.get_clock().now().nanoseconds * 1e-9  # noqa: E731
    stream = ReplayStream(JointCommandPublisher(node, args.arm, names, limb.joint_names()), clock)
    steer = stream.offset = CuffSteer(node, args.arm, Cuff(node, args.arm), stream)
    if args.hold or args.chase is not None:
        steer.follow.share[:] = 0.0
        steer.admittance.damping[:] = np.inf
    if args.chase is not None:
        node.declare_parameter('chase_gain_arm', args.chase)
        node.declare_parameter('chase_gain_wrist', WRIST_RATIO * args.chase)
    deflection = node.create_publisher(JointState, '/follow_probe/deflection', 10)
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
    pose = Pose(q0.copy())
    stream.play(pose, clock())
    stream.start()
    print('Holding this pose. Squeeze, push, raise the sliders slowly. Ctrl-C to stop.', flush=True)
    try:
        last = time.monotonic()
        while not steer.tripped and not steer.contact:
            steer.live(steer.q_measured)
            now = time.monotonic() - start
            dt, last = time.monotonic() - last, time.monotonic()
            if args.chase is not None:
                gain = np.zeros(7)
                for name, joints in CHASE.items():
                    gain[joints] = node.get_parameter(name).value
                rate = gain * deadzone(steer.q_measured - pose.q, CHASE_DEADZONE) if steer.held else np.zeros(7)
                pose.q, pose.qd = pose.q + rate * dt, rate
            samples.append([
                now,
                steer.held,
                *steer.follow.share[[0, 4]],
                *steer.q_measured,
                *steer.hand_torque(),
                *pose.q,
            ])
            sample = JointState(
                name=names, position=(1e3 * (steer.q_measured - pose.q)).tolist(), effort=steer.hand_torque().tolist()
            )
            sample.header.stamp = node.get_clock().now().to_msg()
            deflection.publish(sample)
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
