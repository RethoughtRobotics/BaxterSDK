"""Fit the joint error model on some bags and test it on others.

  python -m baxter_teleop.diagnostics.joint_error_fit --train BAG [BAG ...] --test BAG [BAG ...]
                                                      [--arm left] [--taps 30] [--out model.npz]

Data: any bag of raw position streaming with the inverse dynamics feedforward (raw_id teleop, VR or
policy, or `replay_demo.sh --mode pose`) that recorded /robot/limb/{arm}/joint_command and
/robot/joint_states. The commanded angle is the named JointCommand (RAW_POSITION_MODE only), the
measured angle Baxter's joint state, both by receive time on this computer; the commanded velocity
is the commanded angle differentiated. Streaming stretches (commands no more than `gap` apart) are
resampled at dt and kept as separate segments, so nothing is fitted across a pause.

The table: per joint, the steady lag the model found (sum of its impulse response, ms), how much of
the training error it explains, and on the test bags the rms error before and after subtracting the
prediction. Explained below zero means the prediction is worse than none: the model does not hold.
"""

import argparse
from pathlib import Path

import numpy as np
import rosbag2_py
from baxter_core_msgs.msg import JointCommand
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState

from ..joint_error_model import JOINTS, JointErrorModel

MIN_SEGMENT = 1.0  # s: shorter streaming stretches are dropped


def raw_segments(bag, arm, dt=0.01, gap=0.05):
    """[(v, e), ...] on a uniform grid of period dt, one per streaming stretch of raw position commands:
    v the commanded velocity, e the commanded minus the measured angle, each (N, 7) in JOINTS order."""
    names = [f'{arm}_{j}' for j in JOINTS]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[f'/robot/limb/{arm}/joint_command', '/robot/joint_states']))
    commands, states = [], []
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic == '/robot/joint_states':
            m = deserialize_message(data, JointState)
            if names[0] in m.name:
                states.append([t * 1e-9, *(m.position[m.name.index(n)] for n in names)])
        else:
            m = deserialize_message(data, JointCommand)
            if m.mode == JointCommand.RAW_POSITION_MODE:
                commands.append([t * 1e-9, *(m.command[list(m.names).index(n)] for n in names)])
    if not commands or not states:
        return []
    commands, states = np.array(commands), np.array(states)

    segments = []
    breaks = np.flatnonzero(np.diff(commands[:, 0]) > gap) + 1
    for run in np.split(commands, breaks):
        start, end = max(run[0, 0], states[0, 0]), min(run[-1, 0], states[-1, 0])
        if end - start < MIN_SEGMENT:
            continue
        grid = np.arange(start, end, dt)
        q_cmd = np.stack([np.interp(grid, run[:, 0], run[:, 1 + j]) for j in range(len(JOINTS))], axis=1)
        q = np.stack([np.interp(grid, states[:, 0], states[:, 1 + j]) for j in range(len(JOINTS))], axis=1)
        segments.append((np.gradient(q_cmd, dt, axis=0), q_cmd - q))
    return segments


def load(bags, arm, dt):
    segments = []
    for bag in bags:
        found = raw_segments(bag, arm, dt)
        seconds = sum(len(v) for v, _ in found) * dt
        print(f'  {Path(bag).name}: {len(found)} segments, {seconds:.0f} s of raw streaming')
        segments += found
    return segments


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--train', nargs='+', required=True, help='bags to fit on')
    parser.add_argument('--test', nargs='+', required=True, help='bags to test on (not used for fitting)')
    parser.add_argument('--arm', default='left')
    parser.add_argument('--taps', type=int, default=30, help='impulse response length in samples (default: 30)')
    parser.add_argument('--dt', type=float, default=0.01, help='grid period, s (default: 0.01)')
    parser.add_argument('--ridge', type=float, default=1e-3, help='ridge penalty (default: 1e-3)')
    parser.add_argument('--out', help='save the fitted model here (.npz)')
    args = parser.parse_args()

    print('train:')
    train = load(args.train, args.arm, args.dt)
    print('test:')
    test = load(args.test, args.arm, args.dt)
    if not train or not test:
        raise SystemExit('no raw position streaming found')

    model = JointErrorModel(args.taps, args.dt, args.ridge).fit(train)
    fitted, held_out = model.explained(train), model.explained(test)
    print(f'\n{args.taps} taps ({args.taps * args.dt * 1e3:.0f} ms), {args.arm} arm')
    print(f'{"joint":6}{"lag ms":>8}{"train explained":>17}{"test rms deg":>14}{"after deg":>11}{"test explained":>16}')
    for j, name in enumerate(JOINTS):
        rms, left, explained = held_out[j]
        print(
            f'{name:6}{model.lag()[j] * 1e3:8.0f}{fitted[j][2] * 100:16.0f}%'
            f'{np.degrees(rms):14.2f}{np.degrees(left):11.2f}{explained * 100:15.0f}%'
        )
    if args.out:
        model.save(args.out)
        print(f'saved {args.out}')


if __name__ == '__main__':
    main()
