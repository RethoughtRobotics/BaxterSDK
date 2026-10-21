"""VRTarget motion scaling: the goal moves motion_scale x the hand's translation, rotation 1:1."""

import types

import numpy as np
import pytest
import yaml
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation

from baxter_teleop.inputs import vr

BASE = {
    'linear_frame': 'base',
    'linear_axes': ['+x', '+y', '+z'],
    'angular_axes': ['+x', '+y', '+z'],
    'linear_scale': [1.0, 1.0, 1.0],
    'angular_scale': [1.0, 1.0, 1.0],
    'linear_deadband': [0.0, 0.0, 0.0],
    'angular_deadband': [0.0, 0.0, 0.0],
    'smoothing': 0.0,
    'swap_hands': True,  # axes as written (operator facing Baxter)
}


def make(tmp_path, monkeypatch, **cfg):
    clock = [0.0]
    monkeypatch.setattr(vr, 'time', types.SimpleNamespace(monotonic=lambda: clock[0]))
    path = tmp_path / 'vr.yaml'
    path.write_text(yaml.safe_dump({**BASE, **cfg}))
    node = types.SimpleNamespace(create_subscription=lambda *a, **k: None)
    return vr.VRTarget(node, '', str(path), 1.0, 2.0, 5.0), clock


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


@pytest.mark.parametrize('swap,sign', [(True, [1, 1, 1]), (False, [-1, -1, 1])])
def test_without_swap_hands_x_and_y_are_negated_for_translation_and_rotation(tmp_path, monkeypatch, swap, sign):
    target, clock = make(tmp_path, monkeypatch, swap_hands=swap)
    (goal_p, _), tcp = stroke(target, clock, (0.10, -0.04, 0.02))
    np.testing.assert_allclose(goal_p - tcp, np.multiply(sign, [0.10, -0.04, 0.02]), atol=1e-6)
    target, clock = make(tmp_path, monkeypatch, swap_hands=swap)
    (_, goal_r), _ = stroke(target, clock, (0, 0, 0), hand_turn=(0.1, 0.2, 0.3))
    np.testing.assert_allclose(goal_r.as_rotvec(), np.multiply(sign, [0.1, 0.2, 0.3]), atol=1e-6)
    assert np.isclose(np.linalg.det(target.angular_axes), 1.0)  # still a proper rotation


def test_tool_frame_maps_are_not_turned(tmp_path, monkeypatch):
    target, _ = make(tmp_path, monkeypatch, linear_frame='tool', swap_hands=False)
    np.testing.assert_allclose(target.linear_axes, np.eye(3))


def test_swap_hands_argument_overrides_the_config(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(vr, 'time', types.SimpleNamespace(monotonic=lambda: clock[0]))
    path = tmp_path / 'vr.yaml'
    path.write_text(yaml.safe_dump({**BASE, 'swap_hands': True}))
    node = types.SimpleNamespace(create_subscription=lambda *a, **k: None)
    target = vr.VRTarget(node, '', str(path), 1.0, 2.0, 5.0, swap_hands=False)
    assert not target.swap_hands
    np.testing.assert_allclose(target.linear_axes, np.diag([-1.0, -1.0, 1.0]))


@pytest.mark.parametrize('axes', [['x', '+y', '+z'], ['+x', '+y'], ['+x', '+y', '+w'], ['+x', '+x', '+z']])
def test_malformed_axes_are_rejected_with_a_clear_message(tmp_path, monkeypatch, axes):
    with pytest.raises(ValueError, match='axes'):
        make(tmp_path, monkeypatch, angular_axes=axes)


def test_a_new_stream_after_a_gap_anchors_at_the_current_pose(tmp_path, monkeypatch):
    target, clock = make(tmp_path, monkeypatch)
    stroke(target, clock, (0.10, 0, 0))
    clock[0] += 10.0  # stream stops: stale
    assert target.twist(np.array([0.6, -0.2, 0.1]), np.eye(3)) is None
    tcp = np.array([0.5, 0.2, 0.0])  # the arm was moved meanwhile
    send(target, clock, (0.6, 0.2, 0.0))  # a new stream, far from the old one
    target.twist(tcp, np.eye(3))  # anchors at the current TCP
    send(target, clock, (0.65, 0.2, 0.0))
    target.twist(tcp, np.eye(3))
    np.testing.assert_allclose(target.goal[0], tcp + [0.05, 0, 0], atol=1e-6)


def takeover(tmp_path, monkeypatch):
    """A Takeover of two base-frame targets on one clock: the policy's and the operator's."""
    policy, clock = make(tmp_path, monkeypatch)
    node = types.SimpleNamespace(create_subscription=lambda *a, **k: None)
    hand = vr.VRTarget(node, '', str(tmp_path / 'vr.yaml'), 1.0, 2.0, 5.0)
    return vr.Takeover(policy, hand), clock


def test_takeover_follows_the_policy_until_the_operator_clutches(tmp_path, monkeypatch):
    t, clock = takeover(tmp_path, monkeypatch)
    tcp = np.array([0.6, -0.2, 0.1])
    send(t.policy, clock, (0, 0, 0))
    send(t.policy, clock, (1e-6, 0, 0))
    t.twist(tcp, np.eye(3))  # the policy's stream anchors at the TCP
    send(t.policy, clock, (0.02, 0, 0))
    send(t.hand, clock, (0.3, 0.3, 0.3))  # the operator's target published, not clutched
    t.twist(tcp, np.eye(3))
    assert not t.active
    np.testing.assert_allclose(t.goal[0], tcp + [0.02, 0, 0], atol=1e-5)


def test_takeover_hands_over_both_ways_from_where_the_arm_is(tmp_path, monkeypatch):
    t, clock = takeover(tmp_path, monkeypatch)
    send(t.policy, clock, (0, 0, 0))
    send(t.policy, clock, (1e-6, 0, 0))
    t.twist(np.array([0.6, -0.2, 0.1]), np.eye(3))

    # The operator clutches: their stroke starts at the TCP where the policy left the arm
    tcp = np.array([0.62, -0.2, 0.1])
    send(t.hand, clock, (0.3, 0.3, 0.3))
    send(t.hand, clock, (0.3 + 1e-6, 0.3, 0.3))
    assert t.active
    assert t.twist(tcp, np.eye(3)) is None  # anchors on the TCP: no jump
    send(t.policy, clock, (0.5, 0, 0))  # the policy's stream goes on, overridden
    send(t.hand, clock, (0.3, 0.35, 0.3))
    t.twist(tcp, np.eye(3))
    np.testing.assert_allclose(t.goal[0], tcp + [0, 0.05, 0], atol=1e-5)

    # The operator lifts: the policy continues from where they left the arm, not from its old anchor
    tcp = np.array([0.62, -0.15, 0.1])
    clock[0] += 0.1
    send(t.hand, clock, (0.3, 0.35, 0.3))  # frozen: released
    send(t.policy, clock, (0.51, 0, 0))
    assert not t.active
    assert t.twist(tcp, np.eye(3)) is None
    send(t.policy, clock, (0.52, 0, 0))
    t.twist(tcp, np.eye(3))
    np.testing.assert_allclose(t.goal[0], tcp + [0.01, 0, 0], atol=1e-5)


def test_a_stale_operator_stream_hands_back_to_the_policy(tmp_path, monkeypatch):
    t, clock = takeover(tmp_path, monkeypatch)
    send(t.hand, clock, (0, 0, 0))
    send(t.hand, clock, (1e-6, 0, 0))
    assert t.active
    clock[0] += 1.0  # the controller stops publishing (lost tracking) while clutched
    assert not t.active
