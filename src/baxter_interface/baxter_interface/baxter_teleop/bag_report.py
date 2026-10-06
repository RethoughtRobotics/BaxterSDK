#!/usr/bin/env python3

"""Report on teleop bags from record_teleop.sh: how well the hand followed the operator's intent.

Intent is the held key's direction (keyboard, timed as KeyTarget does) or the VR servo goal's
motion. Per motion segment and per bag it reports the direction error of the achieved TCP motion,
off-axis drift, orientation drift, joint limits, torque saturation, Jacobian conditioning and how
far the configuration wandered from the untuck pose; per bag also how well the robot tracked the
joint velocity commands. Several bags are compared side by side.

  analyze_teleop_bag.sh <bag> [<bag> ...]   (also writes <bag>/joints.npz for contact replay)
"""

import argparse
import json
import os

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation

import baxter_interface

from .cartesian_delta_teleop import CartesianDeltaTeleop, frax, jax, jnp
from .keymap import twist_from_key
from .urdf_tools import extract_arm_chain_urdf, resolve_baxter_urdf

TORQUE_LIMITS = np.array([50.0, 100.0, 50.0, 50.0, 15.0, 15.0, 15.0])  # Baxter, N m
SUFFIXES = CartesianDeltaTeleop.JOINT_SUFFIXES


def load(bag):
    """Arrays from a bag, times in seconds from its first message."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    config = next(
        (json.loads(m.data) for topic, _, m in _messages(bag, types, '/config') if topic.startswith('/teleop/')), {}
    )
    sim = next((json.loads(m.data) for topic, _, m in _messages(bag, types, '/config') if topic == '/sim/config'), {})
    arm = config.get('args', {}).get('arm', 'left')
    joints = [f'{a}_{s}' for a in (arm, 'right' if arm == 'left' else 'left') for s in SUFFIXES]
    rows = {k: [] for k in ('key', 'goal', 'cmd', 'js', 'ep')}
    for topic, t, m in _messages(bag, types):
        if topic == f'/teleop/{arm}/key':
            rows['key'].append((t, m.data))
        elif topic == f'/teleop/{arm}/goal':
            p, o = m.pose.position, m.pose.orientation
            rows['goal'].append([t, p.x, p.y, p.z, o.x, o.y, o.z, o.w])
        elif topic == f'/robot/limb/{arm}/joint_command' and m.mode == 2:  # velocity mode
            index = {n: i for i, n in enumerate(m.names)}
            rows['cmd'].append([t] + [m.command[index[n]] for n in joints[:7]])
        elif topic == '/robot/joint_states' and joints[0] in m.name:
            index = {n: i for i, n in enumerate(m.name)}
            rows['js'].append(
                [t]
                + [m.position[index[n]] for n in joints]
                + [m.velocity[index[n]] for n in joints[:7]]
                + [m.effort[index[n]] for n in joints[:7]]
            )
        elif topic == f'/robot/limb/{arm}/endpoint_state':
            p, o, v = m.pose.position, m.pose.orientation, m.twist.linear
            rows['ep'].append([t, p.x, p.y, p.z, o.x, o.y, o.z, o.w, v.x, v.y, v.z])
    data = {k: np.array([r for r in v if not isinstance(r, tuple)]) for k, v in rows.items() if k != 'key'}
    t0 = data['js'][0, 0]
    for k in data:
        if len(data[k]):
            data[k][:, 0] -= t0
    data['key'] = [(t - t0, c) for t, c in rows['key']]
    data.update(
        arm=arm,
        joints=joints,
        config=config,
        sim=sim,
        robot='sim' if sim or '_sim_' in os.path.basename(bag.rstrip('/')) else 'real',
    )
    return data


def save_joints(bag, data):
    """<bag>/joints.npz for BaxFlow-T's contact replay (baxflow.contacts)."""
    np.savez(
        os.path.join(bag, 'joints.npz'),
        t=data['js'][:, 0],
        q=data['js'][:, 1:15],
        names=data['joints'],
        empty=bool(data['sim'].get('empty', False)),
    )


def _messages(bag, types, suffix=None):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    if suffix:
        reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in types if t.endswith(suffix)]))
    while reader.has_next():
        topic, raw, t = reader.read_next()
        yield topic, t * 1e-9, deserialize_message(raw, get_message(types[topic]))


def intent(data):
    """Unit intended TCP direction (base frame) at each endpoint sample, or zeros when idle."""
    t = data['ep'][:, 0]
    out = np.zeros((len(t), 3))
    if data['key']:
        args = data['config'].get('args', {})
        hold_initial, hold_repeat = args.get('key_hold_initial', 0.55), 0.10
        held = []  # (start, end, direction)
        for time, key in data['key']:
            v = twist_from_key(key, 1.0, 1.0)[:3]
            if not np.any(v):
                continue
            if held and held[-1][2] is not None and np.array_equal(held[-1][2], v) and time <= held[-1][1]:
                held[-1][1] = time + hold_repeat
            else:
                held.append([time, time + hold_initial, v])
        for start, end, v in held:
            out[(t >= start) & (t < end)] = v
    elif len(data['goal']) > 1:
        g = data['goal']
        vel = np.gradient(g[:, 1:4], g[:, 0], axis=0)
        index = np.searchsorted(g[:, 0], t).clip(0, len(g) - 1)
        moving = (np.linalg.norm(vel[index], axis=1) > 0.02) & (np.abs(g[index, 0] - t) < 0.05)
        out[moving] = vel[index][moving] / np.linalg.norm(vel[index][moving], axis=1, keepdims=True)
    return out


def analyze(data):
    ep, js, cmd = data['ep'], data['js'], data['cmd']
    q, effort = js[:, 1:8], js[:, 22:29]
    want = intent(data)
    active = np.linalg.norm(want, axis=1) > 0
    moving = np.linalg.norm(ep[:, 8:11], axis=1) > 0.02
    cos = np.sum(want * ep[:, 8:11], axis=1) / (np.linalg.norm(ep[:, 8:11], axis=1) + 1e-12)
    direction_error = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    use = active & moving
    # Off-axis motion: the TCP displacement perpendicular to the intended direction while active.
    step = np.diff(ep[:, 1:4], axis=0)
    along = np.sum(step * want[1:], axis=1)
    perpendicular = np.linalg.norm(step - along[:, None] * want[1:], axis=1)
    off_axis = perpendicular[active[1:]].sum() / max(np.abs(along[active[1:]]).sum(), 1e-9)
    # Orientation change while only translation keys (or VR translation) were active.
    rotation = Rotation.from_quat(ep[:, 4:8])
    turn = (rotation[1:] * rotation[:-1].inv()).magnitude()
    orientation_drift = np.degrees(turn[active[1:]].sum())
    low, high = (np.asarray(x) for x in CartesianDeltaTeleop.JOINT_POSITION_LIMITS)
    at_limit = (q - low < 0.03) | (high - q < 0.03)
    saturated = np.abs(effort) > 0.98 * TORQUE_LIMITS
    home = np.asarray(baxter_interface.settings.UNTUCK_POSITIONS[data['arm']])
    lag, tracking = np.nan, np.nan
    if len(cmd):
        best = []
        for delay in (0.0, 0.05, 0.1, 0.15, 0.2):
            index = np.searchsorted(js[:, 0], cmd[:, 0] + delay).clip(0, len(js) - 1)
            best.append((
                np.median(
                    np.linalg.norm(js[index, 15:22] - cmd[:, 1:8], axis=1)
                    / (np.linalg.norm(cmd[:, 1:8], axis=1) + 1e-3)
                ),
                delay,
            ))
        tracking, lag = min(best)
    names = data['joints'][:7]
    robot = frax.core.manipulator.Manipulator(
        extract_arm_chain_urdf(resolve_baxter_urdf(), data['arm'], names), joint_ordering=names
    )
    jacobian = jax.jit(robot.ee_jacobian)
    cond = [np.linalg.cond(np.asarray(jacobian(jnp.asarray(row, dtype=jnp.float32)))) for row in q[::20]]
    return {
        'duration s': js[-1, 0],
        'direction error deg (mean)': np.mean(direction_error[use]) if use.any() else np.nan,
        'direction error deg (p95)': np.percentile(direction_error[use], 95) if use.any() else np.nan,
        'off-axis mm per m moved': 1e3 * off_axis,
        'orientation drift deg': orientation_drift,
        'config from untuck deg (max)': np.degrees(np.abs(q - home).max()),
        'e0 range deg': np.degrees(np.ptp(q[:, 2])),
        'time at a joint limit %': 100 * at_limit.any(1).mean(),
        'joints at limits': ','.join(np.array(SUFFIXES)[at_limit.any(0)]) or '-',
        'time torque saturated %': 100 * saturated.any(1).mean(),
        # The other arm only holds still; moving means this arm pushed it.
        'other arm pushed deg (max)': np.degrees(np.abs(js[:, 8:15] - js[0, 8:15]).max()),
        'jacobian cond (median/max)': f'{np.median(cond):.0f}/{np.max(cond):.0f}',
        'velocity tracking error % (median)': 100 * tracking,
        'best-fit robot lag s': lag,
    }


def segments(data):
    """Per motion segment (contiguous intended motion) rows for the detailed table."""
    ep = data['ep']
    want = intent(data)
    active = np.linalg.norm(want, axis=1) > 0
    edges = np.flatnonzero(np.diff(np.r_[0, active.astype(int), 0]))
    rotation = Rotation.from_quat(ep[:, 4:8])
    rows = []
    for a, b in zip(edges[::2], edges[1::2]):
        if ep[b - 1, 0] - ep[a, 0] < 0.2:
            continue
        d = want[a]
        move = ep[b - 1, 1:4] - ep[a, 1:4]
        moving = np.linalg.norm(ep[a:b, 8:11], axis=1) > 0.02
        cos = (ep[a:b, 8:11] @ d) / (np.linalg.norm(ep[a:b, 8:11], axis=1) + 1e-12)
        err = np.degrees(np.arccos(np.clip(cos[moving], -1, 1)))
        rows.append((
            ep[a, 0],
            ep[b - 1, 0] - ep[a, 0],
            np.round(d, 2),
            1e3 * (move @ d),
            1e3 * np.linalg.norm(move - (move @ d) * d),
            np.mean(err) if len(err) else np.nan,
            np.degrees((rotation[b - 1] * rotation[a].inv()).magnitude()),
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bags', nargs='+')
    args = parser.parse_args()
    results = {}
    for bag in args.bags:
        data = load(bag)
        name = os.path.basename(bag.rstrip('/'))
        print(
            f'\n=== {name} ({data["robot"]}, {data["arm"]} arm, '
            f'{data["config"].get("args", {}).get("input", "?")} input, git {data["config"].get("git", "?")})'
        )
        print(
            f'{"t (s)":>6} {"dur":>5} {"intent dir":>16} {"along mm":>9} {"off-axis mm":>11} {"dir err deg":>11} {"turn deg":>8}'
        )
        for row in segments(data):
            print(
                f'{row[0]:6.1f} {row[1]:5.1f} {str(row[2]):>16} {row[3]:9.0f} {row[4]:11.0f} {row[5]:11.0f} {row[6]:8.1f}'
            )
        results[name] = analyze(data)
        save_joints(bag, data)
    width = max(len(n) for n in results)
    print('\n' + f'{"metric":36s}' + ''.join(f' {n:>{width}s}' for n in results))
    for metric in next(iter(results.values())):
        cells = []
        for r in results.values():
            v = r[metric]
            cells.append(f' {v:>{width}.1f}' if isinstance(v, (float, np.floating)) else f' {str(v):>{width}s}')
        print(f'{metric:36s}' + ''.join(cells))


if __name__ == '__main__':
    main()
