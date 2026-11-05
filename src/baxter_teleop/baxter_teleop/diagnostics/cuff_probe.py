"""Check that a held cuff can steer, before any policy: does the arm stay stiff with the cuff's zero-g
suppressed, does the squeeze still read, and how far does a push stand above the torque noise.

    python -m baxter_teleop.diagnostics.cuff_probe --arm left

Holds the arm where it is with the stream a plan plays through, and suppresses the cuff's zero-g. The
only command is the pose the arm is already in. Nothing else may command the arm: stop the teleop first.
Squeeze the cuff and push in each direction when asked. On exit or Ctrl-C the arm returns to its normal
position hold. Prints the verdict and saves the samples.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import rclpy
from baxter_core_msgs.msg import CollisionDetectionState, SEAJointState
from rclpy.node import Node
from std_msgs.msg import Empty

import baxter_interface

from ..inputs.cuff import Cuff
from ..inputs.steer import SUPPRESS_PERIOD, TARE_TIME, Admittance
from ..outputs.raw_position import JointCommandPublisher
from ..outputs.setpoint import Setpoint
from ..settings import JOINTS, STEER, joint_names

PHASES = (
    ('Hands off', 4.0),
    ('Squeeze the cuff, no push', 4.0),
    ('Hold and push toward the robot', 4.0),
    ('Hold and push away', 4.0),
    ('Hold and push down', 4.0),
    ('Hold and push up', 4.0),
    ('Let go', 3.0),
)
STIFF = 0.05  # rad: a squeezed arm that moved further than this is in zero-g
RATE = 200.0  # Hz


def sign_agrees(force, moved):
    """Per joint pushed past its deadband: the arm gives way along the torque Admittance would follow,
    -(effort - tare). True, or None with no such push. Steering the wrong way would pull into the hand."""
    pushed = np.abs(force).max(0, initial=0.0) > 2 * np.array(STEER['deadband']) if len(force) else np.zeros(7, bool)
    if not pushed.any():
        return None
    return bool(all(np.sum(-force[:, j] * moved[:, j]) > 0 for j in np.nonzero(pushed)[0]))


def verdict(samples):
    """Findings from `samples` (time, phase, squeezed, torque - tare (7), displacement (7), contact)."""
    phase, squeezed = samples[:, 1].astype(int), samples[:, 2] > 0
    force = samples[:, 3:10]
    moved = samples[:, 10:17]
    still = force[phase == 0]
    noise = np.abs(still).max(0) if len(still) else np.zeros(7)
    pushes = np.abs(force[phase >= 2]).max(0) if (phase >= 2).any() else np.zeros(7)
    return {
        'cuff_read': bool(squeezed.any()),
        'stiff': bool(np.abs(moved[squeezed]).max(initial=0.0) < STIFF),
        'moved_rad': float(np.abs(moved).max(initial=0.0)),
        'sign_agrees': sign_agrees(force[phase >= 2], moved[phase >= 2]),
        'noise_nm': noise,
        'push_nm': pushes,
        'contact_events': int((np.diff(samples[:, 17]) > 0).sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', default='left', choices=('left', 'right'))
    p.add_argument('--out', type=Path, default=Path('cuff_probe.npz'))
    args = p.parse_args()
    rclpy.init()
    node = Node('cuff_probe')
    limb = baxter_interface.Limb(args.arm, node)
    names = joint_names(args.arm)
    send = JointCommandPublisher(node, args.arm, names, limb.joint_names())
    suppress = node.create_publisher(Empty, f'/robot/limb/{args.arm}/suppress_cuff_interaction', 1)
    cuff = Cuff(node, args.arm)
    state = {'effort': None, 'q': None, 'contact': 0}

    def on_effort(msg):
        i = [msg.name.index(name) for name in names]
        state['effort'] = np.array(msg.actual_effort)[i] - np.array(msg.gravity_model_effort)[i]
        state['q'] = np.array(msg.actual_position)[i]

    def on_contact(msg):
        state['contact'] = int(msg.collision_state)

    node.create_subscription(SEAJointState, f'/robot/limb/{args.arm}/gravity_compensation_torques', on_effort, 10)
    node.create_subscription(
        CollisionDetectionState, f'/robot/limb/{args.arm}/collision_detection_state', on_contact, 10
    )
    while state['q'] is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    q0 = state['q'].copy()
    hold = Setpoint(q0, np.zeros(7), np.zeros(7))
    filtered = Admittance(**{k: STEER[k] for k in ('damping', 'deadband', 'decay', 'cutoff', 'max_rate', 'max_offset')})
    samples, phase, start = [], 0, time.monotonic()
    next_suppress = 0.0
    print(PHASES[0][0], flush=True)
    try:
        ends = np.cumsum([s for _, s in PHASES])
        while (now := time.monotonic() - start) < ends[-1]:
            rclpy.spin_once(node, timeout_sec=0.0)
            if now >= next_suppress:
                suppress.publish(Empty())
                next_suppress = now + SUPPRESS_PERIOD
            send(hold)
            filtered.measure(state['effort'], now, cuff.squeezed)
            while now >= ends[phase]:
                phase += 1
                print(PHASES[phase][0], flush=True)
            if np.max(np.abs(state['q'] - q0)) > 2 * STIFF and cuff.squeezed:
                print('The arm is free: stopping, the suppression does not hold.', flush=True)
                break
            samples.append([
                now,
                phase,
                cuff.squeezed,
                *(filtered.effort - filtered.tare),
                *(state['q'] - q0),
                state['contact'],
            ])
            time.sleep(1.0 / RATE)
    finally:
        limb.exit_control_mode()
    samples = np.array(samples)
    np.savez(args.out, samples=samples, joints=JOINTS, tare_time=TARE_TIME)
    found = verdict(samples)
    print(f'Cuff still reads: {found["cuff_read"]}')
    print(f'Arm stiff while squeezed: {found["stiff"]} (moved {found["moved_rad"]:.3f} rad)')
    print(f'Contact safety events: {found["contact_events"]}')
    print('          ' + ''.join(f'{j:>7}' for j in JOINTS))
    print('noise N m ' + ''.join(f'{x:7.2f}' for x in found['noise_nm']))
    print('push  N m ' + ''.join(f'{x:7.2f}' for x in found['push_nm']))
    print(f'Samples: {args.out.resolve()}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
