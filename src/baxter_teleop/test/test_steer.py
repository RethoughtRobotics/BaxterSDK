"""Cuff steering: the admittance, its leak against a re-anchoring policy, and its way into the stream."""

import types

import numpy as np
import pytest

from baxter_teleop.diagnostics.cuff_probe import verdict
from baxter_teleop.inputs.steer import Admittance, CuffSteer, deadzone
from baxter_teleop.outputs.replay_stream import ReplayStream
from baxter_teleop.settings import LIMITS, STEER
from baxter_teleop.teleop import ESC, Teleop

DT = 0.005
KEYS = ('damping', 'deadband', 'decay', 'cutoff', 'max_rate', 'max_offset')
# The merge of multi_task_dit: ages of the chunks in the merged plan, and their weights.
AGES = np.array([0.27, 0.43, 0.59, 0.75, 0.91, 1.07, 1.23, 1.39])
WEIGHTS = np.array([0.128, 0.21, 0.172, 0.141, 0.115, 0.094, 0.077, 0.063])


def admittance(**over):
    return Admittance(**{**{k: STEER[k] for k in KEYS}, **over})


def settled(a, effort, held, now=0.0, seconds=2.0):
    for k in range(int(seconds * 100)):
        a.measure(effort, now + k * 0.01, held)
    return now + seconds


def test_deadzone_is_continuous_and_keeps_the_sign():
    assert np.allclose(deadzone(np.array([-3.0, -0.5, 0.5, 3.0]), 1.0), [-2.0, 0.0, 0.0, 2.0])


def test_a_push_moves_the_joint_with_it():
    """A hand pushing a joint positive makes the spring carry a negative torque: actual - gravity < 0."""
    a = admittance()
    now = settled(a, np.zeros(7), held=False)
    settled(a, np.r_[-5.0, np.zeros(6)], held=True, now=now)
    offset, rate = a.step(DT, held=True)
    assert offset[0] > 0 and rate[0] > 0 and not offset[1:].any()
    assert rate[0] == pytest.approx((5.0 - STEER['deadband'][0]) / STEER['damping'][0], rel=0.02)


def test_the_model_offset_at_this_pose_is_tared():
    a = admittance()
    bias = np.array([1.0, -14.0, -10.0, -9.0, 0.4, 0.5, 0.1])
    now = settled(a, bias, held=False)
    settled(a, bias, held=True, now=now)
    assert not a.step(DT, held=True)[0].any()


def test_noise_inside_the_deadband_does_not_move_the_arm():
    a = admittance()
    rng = np.random.default_rng(0)
    deadband = np.array(STEER['deadband'])
    now = settled(a, np.zeros(7), held=False)
    for k in range(200):
        a.measure(deadband * 0.9 * rng.uniform(-1, 1, 7), now + k * 0.01, True)
        assert not a.step(0.01, held=True)[0].any()


def test_a_released_cuff_only_lets_the_offset_decay():
    a = admittance()
    a.offset[:] = 0.1
    settled(a, np.full(7, 50.0), held=False)
    for _ in range(int(5 * STEER['decay'] / DT)):
        offset, _ = a.step(DT, held=False)
    assert np.abs(offset).max() < 0.1 * np.exp(-4)


def test_the_offset_and_its_rate_are_bounded():
    a = admittance(max_rate=0.2, max_offset=0.15)
    now = settled(a, np.zeros(7), held=False)
    settled(a, np.full(7, -500.0), held=True, now=now)
    for _ in range(4000):
        offset, rate = a.step(DT, held=True)
        assert np.abs(rate).max() <= 0.2 + 1e-9
    assert np.abs(offset).max() <= 0.15


def steered(decay, force=10.0, mobility=0.005, push=2.0, seconds=6.0, dt=0.01):
    """1-D arm on plan + offset, the plan anchored at the arm as it was (a relative policy holding still)."""
    x, offset = np.zeros(int(seconds / dt)), 0.0
    lags = (AGES / dt).astype(int)
    for k in range(1, len(x)):
        plan = WEIGHTS @ x[np.maximum(k - lags, 0)]
        offset += dt * (mobility * force * (k * dt < push) - offset / decay)
        x[k] = x[k - 1] + dt / 0.03 * (plan + offset - x[k - 1])
    return x


def test_an_offset_that_stays_runs_away_from_a_relative_policy():
    x = steered(decay=1e9)
    assert (x[-1] - x[-50]) / 0.49 > 0.1  # 4 s after the hand let go, still over 100 mm/s


def test_the_leak_stops_the_arm_when_the_hand_does_at_the_hands_speed():
    x = steered(decay=STEER['decay'])
    assert (x[190] - x[140]) / 0.5 == pytest.approx(0.05, rel=0.1)  # 10 N at 0.005 (m/s)/N
    assert abs(x[-1] - x[-50]) / 0.49 < 1e-3


class Plan:
    def __call__(self, t):
        return np.full(7, 0.5), np.zeros(7), np.zeros(7)


class Offset:
    def __init__(self, value):
        self.value, self.resets = np.asarray(value, float), 0

    def __call__(self, dt):
        return self.value, np.full(7, 0.1)

    def reset(self):
        self.resets += 1


def stream(offset):
    sent = []
    s = ReplayStream(sent.append, lambda: 0.0)
    s.offset = offset
    s.play(Plan(), 0.0)
    return s, sent


def test_the_stream_adds_the_offset_to_each_setpoint():
    s, sent = stream(Offset(np.full(7, 0.001)))
    s.tick()
    assert np.allclose(sent[-1].position, 0.501) and np.allclose(sent[-1].velocity, 0.1)


def test_the_offset_stays_inside_the_joint_limits():
    s, sent = stream(Offset(np.full(7, -5.0)))
    s.tick()
    assert np.all(sent[-1].position >= LIMITS.position_low)


def test_stopping_the_stream_drops_the_offset():
    offset = Offset(np.zeros(7))
    s, _ = stream(offset)
    s.stop()
    assert offset.resets == 1


class Node:
    def __init__(self):
        self.published, self.timer, self.effort, self.errors = [], None, None, []

    def create_publisher(self, kind, topic, qos):
        return types.SimpleNamespace(publish=lambda m: self.published.append(topic))

    def create_subscription(self, kind, topic, callback, qos):
        self.effort = callback

    def create_timer(self, period, callback):
        self.timer = callback

    def get_clock(self):
        return types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=0))

    def get_logger(self):
        return types.SimpleNamespace(error=self.errors.append)


def steer(squeezed=False):
    node = Node()
    cuff = types.SimpleNamespace(squeezed=squeezed)
    out = types.SimpleNamespace(active=True, position=np.zeros(7))
    return node, cuff, out, CuffSteer(node, 'left', cuff, out)


def test_zero_g_is_suppressed_only_while_a_plan_plays():
    node, _, out, _ = steer()
    node.timer()
    assert node.published == ['/robot/limb/left/suppress_cuff_interaction']
    out.active = False
    node.timer()
    assert len(node.published) == 1


def test_an_arm_that_leaves_its_setpoints_ends_steering_for_the_session():
    node, cuff, out, s = steer(squeezed=True)
    assert s.live(np.zeros(7))
    assert not s.live(np.full(7, STEER['max_tracking_error'] + 0.01))
    assert node.errors and not s.active
    cuff.squeezed = False
    assert not s.live(np.zeros(7))
    node.timer()
    assert not node.published


def test_a_released_arm_away_from_its_setpoints_is_not_a_failure():
    _, _, _, s = steer(squeezed=False)
    assert s.live(np.full(7, 1.0))


def test_the_probe_tells_a_stiff_arm_from_a_free_one():
    def rows(moved):
        row = np.zeros((10, 18))
        row[:, 1], row[:, 2], row[:, 3], row[:, 10] = 2, 1, 8.0, moved
        return np.vstack([np.zeros((10, 18)), row])

    stiff, free = verdict(rows(0.01)), verdict(rows(0.3))
    assert stiff['stiff'] and stiff['cuff_read'] and stiff['push_nm'][0] == 8.0
    assert not free['stiff']


class Loop:
    """One cycle of Teleop.spin_target on a plan, the cuff squeezed."""

    def __init__(self, steering):
        self.teleop = object.__new__(Teleop)
        self.calls = []
        t = self.teleop
        t.limb = types.SimpleNamespace(set_command_timeout=lambda s: None)
        t.kinematics = types.SimpleNamespace(tcp=lambda q: (np.zeros(3), np.eye(3)))
        t.telemetry = types.SimpleNamespace(tcp=lambda *a: None, goal=lambda *a: None)
        t.raw_output = types.SimpleNamespace(position=None, play=lambda *a: self.calls.append('play'))
        t._current_q = lambda: np.zeros(7)
        t._spin_pending = lambda: None
        t._hold = lambda: self.calls.append('hold')
        t._cuff = types.SimpleNamespace(squeezed=True)
        t._steer = types.SimpleNamespace(live=lambda q: steering) if steering is not None else None
        keys = iter([None, ESC])
        t._read_key = lambda: next(keys)
        self.target = types.SimpleNamespace(
            goal=None, plan=lambda *a: ('curve', 0.0), reset=lambda: self.calls.append('reset')
        )

    def run(self):
        self.teleop.spin_target(self.target)
        return self.calls


def test_a_squeeze_that_steers_keeps_playing_the_plan(monkeypatch):
    monkeypatch.setattr('baxter_teleop.teleop.rclpy.ok', lambda: True)
    assert Loop(True).run() == ['play', 'hold']  # the hold is the loop's exit


def test_a_squeeze_without_steering_lets_go_of_the_plan(monkeypatch):
    monkeypatch.setattr('baxter_teleop.teleop.rclpy.ok', lambda: True)
    assert Loop(None).run() == ['reset']
    assert Loop(False).run() == ['reset']
