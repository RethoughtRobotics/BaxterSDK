#!/usr/bin/env python3

"""Replay a recorded VR teleop bag through two translation mappings and loop the result in RViz.

Both runs start from the untuck pose (tuck_arms -u) and drive a simulated arm with the teleop's
own pieces: frax model, DiffIKSolver, joint limits and acceleration ramp. Velocity
commands are tracked perfectly, so any difference between the runs comes from the mapping alone.

  current   config/vr_axes.yaml: translation in the controller axes, then the TCP axes, at each clutch
  proposed  config/vr_axes_base.yaml: translation in the anchor frame (set by pressing A), mapped
            once onto the robot base

"Intent" is the hand motion mapped through the controller -> base map the operator felt on the
first stroke of the real run (snapped to the nearest signed permutation); it is printed so it can
be compared with linear_axes in config/vr_axes_base.yaml.
"""

import argparse
import itertools
import os
import types

import numpy as np
import rclpy
import rosbag2_py
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

import baxter_interface

from . import vr_target
from .cartesian_delta_teleop import CartesianDeltaTeleop, frax, jax, jnp
from .diff_ik import DiffIKSolver
from .urdf_tools import extract_arm_chain_urdf, resolve_baxter_urdf

DT = 0.01
RUNS = {'current': (0.0, 1.0, 0.3, 1.0), 'proposed': (0.1, 0.8, 0.3, 1.0)}  # name: marker RGBA
SPACING = 1.0  # m each robot sits either side of the world origin


def untuck_pose():
    """{arm: 7 joints}, the shared home pose (order s0 s1 e0 e1 w0 w1 w2)."""
    return baxter_interface.settings.UNTUCK_POSITIONS


def urdf_fk(urdf, link, q):
    """Pose of `link` in base from the full URDF, with joint values q {name: angle}."""
    import xml.etree.ElementTree as ET

    by_child = {j.find('child').get('link'): j for j in ET.parse(urdf).getroot().findall('joint')}
    chain = []
    while link in by_child:
        chain.append(by_child[link])
        link = chain[-1].find('parent').get('link')
    T = np.eye(4)
    for j in reversed(chain):
        o = j.find('origin')
        A = np.eye(4)
        A[:3, :3] = Rotation.from_euler('xyz', np.array((o.get('rpy') or '0 0 0').split(), float)).as_matrix()
        A[:3, 3] = np.array((o.get('xyz') or '0 0 0').split(), float)
        T = T @ A
        if j.get('type') == 'revolute':
            B = np.eye(4)
            axis = np.array(j.find('axis').get('xyz').split(), float)
            B[:3, :3] = Rotation.from_rotvec(axis * q.get(j.get('name'), 0.0)).as_matrix()
            T = T @ B
    return T


def read_bag(bag, arm):
    """VR target poses [t, x, y, z, qx, qy, qz, qw] and the real TCP orientation at each stamp."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag, storage_id='mcap'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
    topics = [f'/vive/{arm}/target', f'/robot/limb/{arm}/endpoint_state']
    reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
    from baxter_core_msgs.msg import EndpointState

    target, endpoint = [], []
    while reader.has_next():
        topic, raw, t = reader.read_next()
        msg = deserialize_message(raw, PoseStamped if topic == topics[0] else EndpointState)
        p, o = msg.pose.position, msg.pose.orientation
        (target if topic == topics[0] else endpoint).append([t * 1e-9, p.x, p.y, p.z, o.x, o.y, o.z, o.w])
    return np.array(target), np.array(endpoint)


def snap_to_signed_permutation(M):
    """Nearest matrix with one +-1 per row and column, i.e. an axes map like config/vr_axes.yaml."""
    best = max(itertools.permutations(range(3)), key=lambda p: sum(abs(M[i, p[i]]) for i in range(3)))
    P = np.zeros((3, 3))
    for i, j in enumerate(best):
        P[i, j] = np.sign(M[i, j])
    return P


def simulate(bag, arm, gain, linear_speed, angular_speed):
    target_msgs, endpoint = read_bag(bag, arm)
    config_dir = os.path.join(get_package_share_directory('baxter_interface'), 'config')
    cfg, cfg_base = (os.path.join(config_dir, f) for f in ('vr_axes.yaml', 'vr_axes_base.yaml'))
    clock = [0.0]
    vr_target.time = types.SimpleNamespace(monotonic=lambda: clock[0])  # VRTarget's clock -> bag time
    node = types.SimpleNamespace(create_subscription=lambda *args, **kwargs: None)

    # The controller -> base map the operator felt on the first stroke of the real run.
    probe = vr_target.VRTarget(node, '', cfg, gain, linear_speed, angular_speed)
    moving = np.r_[False, np.any(np.diff(target_msgs[:, 1:], axis=0) != 0, axis=1)]
    first = int(np.argmax(moving))
    rc0 = Rotation.from_quat(target_msgs[first - 1, 4:8])
    rt0 = Rotation.from_quat(endpoint[np.searchsorted(endpoint[:, 0], target_msgs[first, 0]), 4:8])
    felt = rt0.as_matrix() @ probe.linear_axes @ rc0.inv().as_matrix()
    base_axes = snap_to_signed_permutation(felt)
    print('Controller -> base map felt on the first real stroke:\n', felt.round(2))
    targets = {
        'current': vr_target.VRTarget(node, '', cfg, gain, linear_speed, angular_speed),
        'proposed': vr_target.VRTarget(node, '', cfg_base, gain, linear_speed, angular_speed),
    }
    same = np.array_equal(base_axes, targets['proposed'].linear_axes)
    print(
        'Snapped:\n',
        base_axes.astype(int),
        '\n',
        'matches' if same else 'DIFFERS from',
        'vr_axes_base.yaml linear_axes',
    )

    urdf = resolve_baxter_urdf()
    joints = [f'{arm}_{s}' for s in CartesianDeltaTeleop.JOINT_SUFFIXES]
    robot = frax.core.manipulator.Manipulator(extract_arm_chain_urdf(urdf, arm, joints), joint_ordering=joints)
    ee_transform, ee_jacobian = jax.jit(robot.ee_transform), jax.jit(robot.ee_jacobian)
    q0 = np.asarray(untuck_pose()[arm], dtype=np.float32)
    T_ee = np.asarray(ee_transform(jnp.asarray(q0)))
    T_tcp = urdf_fk(urdf, f'{arm}_gripper', dict(zip(joints, q0.tolist())))
    tcp_offset = T_ee[:3, :3].T @ (T_tcp[:3, 3] - T_ee[:3, 3])  # TCP = URDF {arm}_gripper frame

    ik = DiffIKSolver(dt=DT, damping=0.05, max_joint_step=0.04, max_joint_velocity=6.0)
    vel_limits = np.asarray(CartesianDeltaTeleop.JOINT_VELOCITY_LIMITS, dtype=np.float32)
    acc_limits = np.asarray(CartesianDeltaTeleop.JOINT_ACCELERATION_LIMITS, dtype=np.float32)
    q_low, q_high = (np.asarray(lim) for lim in CartesianDeltaTeleop.JOINT_POSITION_LIMITS)

    def tcp(q):
        T = np.asarray(ee_transform(jnp.asarray(q)), dtype=np.float64)
        return T[:3, 3] + T[:3, :3] @ tcp_offset, T[:3, :3]

    times = np.arange(target_msgs[0, 0], target_msgs[-1, 0] + 1.0, DT)
    state = {name: {'q': q0.copy(), 'qdot': np.zeros(7, np.float32)} for name in targets}
    out = {name: {'q': [], 'tcp': [], 'goal': []} for name in targets}
    intent, clutched = [], []
    p_intent = tcp(q0)[0]
    msg_i = 0
    for t in times:
        clock[0] = t
        while msg_i < len(target_msgs) and target_msgs[msg_i, 0] <= t:
            m = PoseStamped()
            p, o = m.pose.position, m.pose.orientation
            p.x, p.y, p.z, o.x, o.y, o.z, o.w = target_msgs[msg_i, 1:]
            for target in targets.values():
                target._on_target(m)
            # Intent: clutched hand displacement through the first-stroke map, accumulated.
            if msg_i and targets['current']._moving:
                p_intent = p_intent + base_axes @ (target_msgs[msg_i, 1:4] - target_msgs[msg_i - 1, 1:4])
            msg_i += 1
        intent.append(p_intent)
        clutched.append(bool(targets['current']._moving))

        for name, target in targets.items():
            s = state[name]
            p, R = tcp(s['q'])
            twist = target.twist(p, R)
            if twist is None:  # clutch released: the arm holds and the ramp restarts from rest
                s['qdot'][:] = 0.0
            else:
                v, w = twist[:3], twist[3:]
                hand_twist = np.concatenate([v - np.cross(w, R @ tcp_offset), w]).astype(np.float32)
                J = ee_jacobian(jnp.asarray(s['q']))
                qdot = ik.joint_velocity(
                    s['q'],
                    J,
                    hand_twist,
                    joint_velocity_limits=vel_limits,
                    joint_position_limits=(q_low, q_high),
                    joint_acceleration_limits=acc_limits,
                )
                change = qdot - s['qdot']
                ratio = float(np.max(np.abs(change) / (acc_limits * DT)))
                s['qdot'] = s['qdot'] + change / max(ratio, 1.0)
                s['q'] = np.clip(s['q'] + s['qdot'] * DT, q_low, q_high).astype(np.float32)
            out[name]['goal'].append(np.full(3, np.nan) if target.goal is None else target.goal[0])
            out[name]['q'].append(s['q'].copy())
            out[name]['tcp'].append(tcp(s['q'])[0])

    result = {'t': times - times[0], 'intent': np.array(intent), 'clutched': np.array(clutched)}
    for name in targets:
        result[name] = {k: np.array(v) for k, v in out[name].items()}
    report(result)
    return result


def direction_error(r, signal, s, e, w=10):
    """Speed-weighted mean angle (deg) between `signal` motion and intended motion over 0.1 s windows."""
    di, ds = r['intent'][s + w : e] - r['intent'][s : e - w], signal[s + w : e] - signal[s : e - w]
    speed = np.linalg.norm(di, axis=1)
    keep = (speed > 0.002) & np.all(np.isfinite(ds), axis=1)
    if not keep.any():
        return float('nan')
    cos = np.sum(di * ds, axis=1)[keep] / (speed[keep] * np.linalg.norm(ds[keep], axis=1) + 1e-12)
    return float(np.average(np.degrees(np.arccos(np.clip(cos, -1, 1))), weights=speed[keep]))


def report(r):
    """Per stroke, how far the servo goal and the achieved TCP head away from the intended motion."""
    clutched = r['clutched']
    edges = np.flatnonzero(np.diff(clutched.astype(int))) + 1
    strokes = [(s, next((e for e in edges if e > s), len(clutched))) for s in edges if clutched[s]]
    strokes = [(s, e) for s, e in strokes if e - s > 30]
    print('\nDirection error vs intent, degrees (goal = where the mapping sends the TCP; tcp = where it went)')
    print(
        f'{"stroke":>6} {"t (s)":>6} {"hand path (mm)":>14}  '
        + '  '.join(f'{n + " goal":>14} {n + " tcp":>13}' for n in RUNS)
    )
    for k, (s, e) in enumerate(strokes):
        length = 1e3 * np.sum(np.linalg.norm(np.diff(r['intent'][s:e], axis=0), axis=1))
        cols = [
            f'{direction_error(r, r[n]["goal"], s, e):14.0f} {direction_error(r, r[n]["tcp"], s, e):13.0f}'
            for n in RUNS
        ]
        print(f'{k:6d} {r["t"][s]:6.1f} {length:14.0f}  ' + '  '.join(cols))


class Playback:
    """Publishes both simulated runs side by side, looping, with intent and TCP trails."""

    def __init__(self, node, result, arm, rate):
        self.node, self.r, self.arm, self.rate = node, result, arm, rate
        self.untuck = untuck_pose()
        self.pubs = {n: node.create_publisher(JointState, f'/{n}/joint_states', 10) for n in RUNS}
        self.markers = node.create_publisher(MarkerArray, '/vr_replay/markers', 10)
        self.static = StaticTransformBroadcaster(node)
        tfs = []
        for i, name in enumerate(RUNS):
            tf = TransformStamped()
            tf.header.frame_id, tf.child_frame_id = 'world', f'{name}/base'
            tf.transform.translation.y = SPACING * (1 - 2 * i)
            tf.transform.rotation.w = 1.0
            tfs.append(tf)
        self.static.sendTransform(tfs)
        self.i = 0
        node.create_timer(DT * 3 / rate, self.tick)  # every 3rd sample

    def tick(self):
        r, i = self.r, self.i
        stamp = self.node.get_clock().now().to_msg()
        for name in RUNS:
            js = JointState()
            js.header.stamp = stamp
            other = 'right' if self.arm == 'left' else 'left'
            js.name = ['head_pan'] + [
                f'{a}_{s}' for a in (self.arm, other) for s in CartesianDeltaTeleop.JOINT_SUFFIXES
            ]
            js.position = [0.0] + r[name]['q'][i].tolist() + list(self.untuck[other])
            self.pubs[name].publish(js)
        self.markers.publish(self.trails(i, stamp))
        self.i = (i + 3) % len(r['t'])

    def trails(self, i, stamp):
        arr = MarkerArray()
        start = max(0, i - int(8.0 / DT))  # last 8 s
        for k, name in enumerate(RUNS):
            for j, (pts, rgba, width, ns) in enumerate([
                (self.r['intent'][start : i + 1], (1.0, 1.0, 1.0, 0.8), 0.006, 'intent (first-stroke map)'),
                (self.r[name]['tcp'][start : i + 1], RUNS[name], 0.012, f'{name} TCP'),
            ]):
                m = Marker(type=Marker.LINE_STRIP, action=Marker.ADD, ns=ns, id=k * 10 + j)
                m.header.frame_id, m.header.stamp = f'{name}/base', stamp
                m.scale.x = width
                m.color.r, m.color.g, m.color.b, m.color.a = rgba
                m.pose.orientation.w = 1.0
                m.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts[::3]]
                arr.markers.append(m)
            goal = self.r[name]['goal'][i]
            sphere = Marker(type=Marker.SPHERE, ns='servo goal', id=k * 10 + 4)
            sphere.header.frame_id, sphere.header.stamp = f'{name}/base', stamp
            sphere.action = Marker.ADD if np.all(np.isfinite(goal)) else Marker.DELETE
            if sphere.action == Marker.ADD:
                sphere.pose.position = Point(x=float(goal[0]), y=float(goal[1]), z=float(goal[2]))
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.05
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 1.0, 0.85, 0.0, 0.9
            arr.markers.append(sphere)
            label = Marker(type=Marker.TEXT_VIEW_FACING, action=Marker.ADD, ns='label', id=k * 10 + 5)
            label.header.frame_id, label.header.stamp = f'{name}/base', stamp
            label.pose.position.z, label.pose.orientation.w, label.scale.z = 1.2, 1.0, 0.12
            label.color.r, label.color.g, label.color.b, label.color.a = RUNS[name]
            clutch = 'CLUTCHED' if self.r['clutched'][i] else 'released'
            label.text = f'{name.upper()}  t={self.r["t"][i]:.1f}s  {clutch}'
            arr.markers.append(label)
        return arr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bag', help='rosbag2 directory recorded during VR teleop')
    parser.add_argument('--arm', choices=['left', 'right'], default='left')
    parser.add_argument('--vr-gain', type=float, default=3.0, help='same as the teleop --vr-gain')
    default_linear, default_angular = CartesianDeltaTeleop.VR_SPEEDS
    parser.add_argument('--linear-speed', type=float, default=default_linear, help='same as the teleop (m/s cap)')
    parser.add_argument('--angular-speed', type=float, default=default_angular, help='same as the teleop (rad/s cap)')
    parser.add_argument('--rate', type=float, default=1.0, help='playback speed (1 = real time)')
    parser.add_argument('--no-playback', action='store_true', help='only simulate and print the report')
    args, ros_args = parser.parse_known_args()

    result = simulate(args.bag, args.arm, args.vr_gain, args.linear_speed, args.angular_speed)
    if args.no_playback:
        return
    rclpy.init(args=['vr_replay', *ros_args])
    node = rclpy.create_node('vr_replay')
    Playback(node, result, args.arm, args.rate)
    print(f'\nLooping {result["t"][-1]:.0f} s at {args.rate}x. Ctrl+C to stop.')
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
