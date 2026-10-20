"""TrajectoryTarget: the servo chases the pose due now with the path's own velocity as feedforward."""

import types

import numpy as np
import pytest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation

from baxter_teleop.inputs.trajectory import TrajectoryTarget

VELOCITY = np.array([0.1, -0.05, 0.02])  # m/s along the test line
START = np.array([0.6, 0.2, 0.1])


def make(**kwargs):
    clock = [0.0]
    node = types.SimpleNamespace(
        create_subscription=lambda *a, **k: None,
        get_clock=lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=int(clock[0] * 1e9))),
        get_logger=lambda: types.SimpleNamespace(warn=lambda *a: None),
    )
    return TrajectoryTarget(node, '', gain=2.0, max_linear=5.0, max_angular=5.0, **kwargs), clock


def path(times, positions, rotvecs=None):
    msg = Path()
    for k, t in enumerate(times):
        pose = PoseStamped()
        pose.header.stamp.sec, pose.header.stamp.nanosec = int(t), int(round((t % 1) * 1e9))
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, positions[k])
        q = Rotation.from_rotvec(np.zeros(3) if rotvecs is None else rotvecs[k]).as_quat()
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
        msg.poses.append(pose)
    return msg


def line(t0=10.0, steps=32, dt=0.04, noise=0.0, seed=0):
    times = t0 + dt * np.arange(steps)
    positions = START + np.outer(times - t0, VELOCITY)
    positions += np.random.default_rng(seed).normal(0.0, noise, positions.shape)
    return times, positions


def test_on_the_path_the_twist_is_the_path_velocity():
    target, clock = make()
    times, positions = line()
    target._on_path(path(times, positions))
    clock[0] = 10.5
    on_path = START + 0.5 * VELOCITY
    twist = target.twist(on_path, np.eye(3))
    np.testing.assert_allclose(twist[:3], VELOCITY, atol=1e-5)
    np.testing.assert_allclose(twist[3:], 0.0, atol=1e-6)


def test_without_feedforward_the_servo_alone_lags():
    target, clock = make(feedforward=False)
    times, positions = line()
    target._on_path(path(times, positions))
    clock[0] = 10.5
    np.testing.assert_allclose(target.twist(START + 0.5 * VELOCITY, np.eye(3)), 0.0, atol=1e-6)


def test_error_is_pulled_back_at_the_gain():
    target, clock = make()
    times, positions = line()
    target._on_path(path(times, positions))
    clock[0] = 10.5
    behind = START + 0.5 * VELOCITY - np.array([0.01, 0.0, 0.0])
    twist = target.twist(behind, np.eye(3))
    np.testing.assert_allclose(twist[:3], VELOCITY + 2.0 * np.array([0.01, 0.0, 0.0]), atol=1e-5)


def test_holds_the_last_pose_then_goes_stale():
    target, clock = make(stale_after=0.2)
    times, positions = line()
    target._on_path(path(times, positions))
    clock[0] = times[-1] + 0.1
    target.twist(positions[-1], np.eye(3))
    np.testing.assert_allclose(target.goal[0], positions[-1])
    clock[0] = times[-1] + 0.3
    assert target.twist(positions[-1], np.eye(3)) is None


def test_continue_extrapolates_the_last_segment():
    target, clock = make(end='continue')
    times, positions = line()
    target._on_path(path(times, positions))
    clock[0] = times[-1] + 0.1
    target.twist(positions[-1], np.eye(3))
    np.testing.assert_allclose(target.goal[0], positions[-1] + 0.1 * VELOCITY, atol=1e-9)


def test_rotation_feedforward_is_in_the_base_frame():
    target, clock = make(interpolation='cubic')
    times = 10.0 + 0.04 * np.arange(32)
    rate = np.array([0.0, 0.0, 0.5])  # rad/s about base z
    tilt = Rotation.from_rotvec([np.pi, 0.0, 0.0])  # TCP pointing down, as on the arm
    rotvecs = [(Rotation.from_rotvec(rate * (t - 10.0)) * tilt).as_rotvec() for t in times]
    target._on_path(path(times, np.tile(START, (32, 1)), rotvecs))
    clock[0] = 10.5
    on_path = Rotation.from_rotvec(rate * 0.5) * tilt
    twist = target.twist(START, on_path.as_matrix())
    np.testing.assert_allclose(twist[3:], rate, atol=1e-3)


def test_smooth_fit_does_not_differentiate_jitter():
    """1 mm noise on 40 ms steps differentiates to ~35 mm/s of velocity noise; the fit removes most of it."""
    errors = {}
    for interpolation in ('linear', 'smooth'):
        target, clock = make(interpolation=interpolation)
        times, positions = line(noise=0.001)
        target._on_path(path(times, positions))
        e = []
        for now in np.linspace(10.2, 11.0, 41):
            clock[0] = now
            target.twist(START, np.eye(3))
            _, _, v, _ = target._sample(now)
            e.append(np.linalg.norm(v - VELOCITY))
        errors[interpolation] = np.sqrt(np.mean(np.square(e)))
    assert errors['smooth'] < 0.5 * errors['linear']


def test_paths_with_non_increasing_stamps_are_ignored():
    target, clock = make()
    times, positions = line()
    target._on_path(path(times[::-1], positions))
    clock[0] = 10.5
    assert target.twist(START, np.eye(3)) is None


def test_rejects_unknown_options():
    with pytest.raises(ValueError):
        make(interpolation='quintic')
    with pytest.raises(ValueError):
        make(end='loop')
