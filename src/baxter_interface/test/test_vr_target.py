"""VRTarget motion scaling: the goal moves motion_scale x the hand's translation, rotation 1:1."""

import types

import numpy as np
import pytest
import yaml
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

from baxter_interface.baxter_teleop import vr_target

BASE = {
    'linear_frame': 'base',
    'linear_axes': ['+x', '+y', '+z'],
    'angular_axes': ['+x', '+y', '+z'],
    'linear_scale': [1.0, 1.0, 1.0],
    'angular_scale': [1.0, 1.0, 1.0],
    'linear_deadband': [0.0, 0.0, 0.0],
    'angular_deadband': [0.0, 0.0, 0.0],
    'smoothing': 0.0,
}


def make(tmp_path, monkeypatch, **cfg):
    clock = [0.0]
    monkeypatch.setattr(vr_target, 'time', types.SimpleNamespace(monotonic=lambda: clock[0]))
    path = tmp_path / 'vr.yaml'
    path.write_text(yaml.safe_dump({**BASE, **cfg}))
    node = types.SimpleNamespace(create_subscription=lambda *a, **k: None)
    return vr_target.VRTarget(node, '', str(path), 1.0, 2.0, 5.0), clock


def send(target, clock, position, rotvec=(0, 0, 0)):
    clock[0] += 0.004
    msg = PoseStamped()
    p, o = msg.pose.position, msg.pose.orientation
    p.x, p.y, p.z = position
    o.x, o.y, o.z, o.w = Rotation.from_rotvec(rotvec).as_quat()
    target._on_target(msg)


def stroke(target, clock, hand_move, hand_turn=(0, 0, 0), tcp=np.array([0.6, -0.2, 0.1])):
    """Clutch at the origin, move the hand by hand_move / hand_turn, return the goal."""
    send(target, clock, (0, 0, 0))
    send(target, clock, (1e-6, 0, 0))  # first change: clutched
    target.twist(tcp, np.eye(3))  # anchors the stroke at the TCP
    for k in range(1, 11):
        send(target, clock, np.asarray(hand_move) * k / 10, np.asarray(hand_turn) * k / 10)
    target.twist(tcp, np.eye(3))
    return target.goal, tcp


@pytest.mark.parametrize('scale', [1.0, 0.5, 0.25])
def test_translation_is_scaled(tmp_path, monkeypatch, scale):
    target, clock = make(tmp_path, monkeypatch, motion_scale=scale)
    (goal_p, _), tcp = stroke(target, clock, (0.10, -0.04, 0.02))
    np.testing.assert_allclose(goal_p - tcp, scale * np.array([0.10, -0.04, 0.02]), atol=1e-6)


def test_rotation_is_not_scaled(tmp_path, monkeypatch):
    target, clock = make(tmp_path, monkeypatch, motion_scale=0.5)
    (_, goal_r), _ = stroke(target, clock, (0, 0, 0), hand_turn=(0, 0, 0.3))
    np.testing.assert_allclose(goal_r.as_rotvec(), [0, 0, 0.3], atol=1e-6)


def test_default_is_one_to_one(tmp_path, monkeypatch):
    target, clock = make(tmp_path, monkeypatch)
    assert target.motion_scale == 1.0


def test_non_positive_scale_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        make(tmp_path, monkeypatch, motion_scale=0.0)
