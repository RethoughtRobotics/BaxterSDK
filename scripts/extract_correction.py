#!/usr/bin/env python3
"""Extract the correction offset from a zero-g takeover bag where it wasn't published.

The correction is: actual TCP - policy target TCP, during cuff squeeze intervals.
Writes a CSV with columns: t, active, dx, dy, dz (metres), and plots it.

Usage:
    python3 scripts/extract_correction.py <bag_path> [-o corrections.png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'install/lib/python3.12/site-packages'))

from baxter_core_msgs.msg import DigitalIOState, EndpointState
from geometry_msgs.msg import PoseStamped
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions


def read_bag(bag_path):
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(bag_path), storage_id='mcap'), ConverterOptions('', ''))

    target_t, target_xyz = [], []
    actual_t, actual_xyz = [], []
    cuff_t, cuff_val = [], []

    while reader.has_next():
        topic, data, ts = reader.read_next()
        t = ts * 1e-9

        if topic == '/policy/left/target':
            msg = deserialize_message(data, PoseStamped)
            p = msg.pose.position
            target_t.append(t)
            target_xyz.append([p.x, p.y, p.z])

        elif topic == '/robot/limb/left/endpoint_state':
            msg = deserialize_message(data, EndpointState)
            p = msg.pose.position
            actual_t.append(t)
            actual_xyz.append([p.x, p.y, p.z])

        elif topic == '/robot/digital_io/left_lower_cuff/state':
            msg = deserialize_message(data, DigitalIOState)
            cuff_t.append(t)
            cuff_val.append(msg.state)

    return (
        np.array(target_t),
        np.array(target_xyz),
        np.array(actual_t),
        np.array(actual_xyz),
        np.array(cuff_t),
        np.array(cuff_val),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('bag', type=Path)
    p.add_argument('-o', '--output', type=Path, default=None)
    args = p.parse_args()

    target_t, target_xyz, actual_t, actual_xyz, cuff_t, cuff_val = read_bag(args.bag)

    t0 = min(target_t[0], actual_t[0])
    target_t -= t0
    actual_t -= t0
    cuff_t -= t0

    # Interpolate target onto actual timestamps
    target_interp = np.column_stack([np.interp(actual_t, target_t, target_xyz[:, i]) for i in range(3)])
    delta = actual_xyz - target_interp  # actual - planned

    # Interpolate cuff state onto actual timestamps (held = squeezed)
    cuff_interp = np.interp(actual_t, cuff_t, cuff_val) > 0.5

    # Find squeeze intervals
    transitions = np.diff(cuff_interp.astype(int))
    starts = np.where(transitions == 1)[0]
    stops = np.where(transitions == -1)[0]

    print(f'Duration: {actual_t[-1]:.1f} s')
    print(f'Squeeze intervals: {len(starts)}')
    for i, s in enumerate(starts):
        end = stops[i] if i < len(stops) else len(actual_t) - 1
        t_start, t_end = actual_t[s], actual_t[end]
        peak = np.max(np.abs(delta[s : end + 1]), axis=0) * 1e3
        print(
            f'  {i + 1}: {t_start:.1f}-{t_end:.1f} s, peak offset (mm): x={peak[0]:.0f} y={peak[1]:.0f} z={peak[2]:.0f}'
        )

    # Stats during squeezes
    squeezed = cuff_interp
    if squeezed.any():
        peak_all = np.max(np.abs(delta[squeezed]), axis=0) * 1e3
        print(f'\nPeak correction during squeezes (mm): x={peak_all[0]:.0f} y={peak_all[1]:.0f} z={peak_all[2]:.0f}')

    # Save CSV
    csv_path = args.bag / 'correction_offset.csv'
    header = 't,active,dx,dy,dz'
    data = np.column_stack([actual_t, cuff_interp.astype(float), delta])
    np.savetxt(csv_path, data, delimiter=',', header=header, comments='', fmt='%.6f')
    print(f'\nCSV: {csv_path}')

    # Plot
    out = args.output or args.bag / 'correction_offset.png'
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 3]})

    ax1.plot(actual_t, cuff_interp.astype(float), 'b-')
    ax1.set_ylabel('cuff squeezed')
    ax1.set_ylim(-0.1, 1.1)

    ax2.plot(actual_t, delta[:, 0], label='x')
    ax2.plot(actual_t, delta[:, 1], label='y')
    ax2.plot(actual_t, delta[:, 2], label='z')
    ax2.set_ylabel('actual - plan (m)')
    ax2.set_xlabel('Time (s)')
    ax2.legend()

    fig.suptitle(args.bag.name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Plot: {out}')


if __name__ == '__main__':
    main()
