"""Cuff steering: the admittance, its leak against a re-anchoring policy, and its way into the stream."""

import types

import numpy as np
import pytest
from std_msgs.msg import Header

from baxter_teleop.diagnostics.cuff_probe import verdict
from baxter_teleop.inputs.steer import Admittance, CuffSteer, Follow, deadzone
from baxter_teleop.outputs.raw_position import RawPositionOutput
from baxter_teleop.outputs.replay_stream import ReplayStream
from baxter_teleop.outputs.setpoint import JointLimits, JointSetpointGenerator
from baxter_teleop.settings import LIMITS, STEER, joint_names
from baxter_teleop.teleop import ESC, Teleop

DT = 0.005
KEYS = ('damping', 'deadband', 'decay', 'cutoff', 'max_rate', 'max_offset')
# The merge of multi_task_dit: ages of the chunks in the merged plan, and their weights.
AGES = np.array([0.22, 0.38, 0.54, 0.70, 0.86])
WEIGHTS = np.array([0.15, 0.24, 0.22, 0.20, 0.19])


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

    def __call__(self, dt, q_plan):
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
        self.published, self.timer, self.errors, self.warnings, self.infos = [], None, [], [], []
        self.subscribed, self.parameters, self.on_parameters = {}, {}, None

    def create_publisher(self, kind, topic, qos):
        return types.SimpleNamespace(publish=lambda m: self.published.append(topic))

    def create_subscription(self, kind, topic, callback, qos):
        self.subscribed[topic] = callback

    def declare_parameter(self, name, value, descriptor):
        self.parameters[name] = (value, descriptor)

    def add_on_set_parameters_callback(self, callback):
        self.on_parameters = callback

    def create_timer(self, period, callback):
        self.timer = callback

    def get_clock(self):
        return types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=0))

    def get_logger(self):
        return types.SimpleNamespace(error=self.errors.append, warning=self.warnings.append, info=self.infos.append)


def steer(squeezed=False):
    node = Node()
    cuff = types.SimpleNamespace(squeezed=squeezed)
    out = types.SimpleNamespace(active=True, position=np.zeros(7), commanded=np.zeros(7))
    return node, cuff, out, CuffSteer(node, 'left', cuff, out)


def test_a_held_cuff_steers_the_raw_output_without_touching_the_servos_setpoint():
    """The real CuffSteer on the real RawPositionOutput, the target chain's pair: the follow share of the
    arm's distance is sent, the setpoint the pose servo closes on stays the policy's, and a stop clears it."""
    q0 = np.array([0.08, -1.0, 1.19, 1.94, -0.67, 1.03, 0.5])
    sent = []
    acceleration = np.array(LIMITS.acceleration)
    limits = JointLimits(LIMITS.velocity, acceleration, 10 * acceleration, LIMITS.position_low, LIMITS.position_high)
    output = RawPositionOutput(JointSetpointGenerator(limits, DT), sent.append, stale_after=10.0)
    node, cuff = Node(), types.SimpleNamespace(squeezed=True)
    s = output.offset = CuffSteer(node, 'left', cuff, output)
    s.follow.share[:] = 0.9
    output.command(np.zeros(7), q0)
    s.q_measured = q0 + 0.1
    for _ in range(400):
        output.tick()
    assert s.live(s.q_measured)
    assert np.allclose(output.position, q0)
    assert np.allclose(sent[-1].position - q0, 0.09, atol=0.02)
    output.stop()
    assert not s.follow.offset.any() and s._added is None


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


def contact(node, state):
    node.subscribed['/robot/limb/left/collision_detection_state'](types.SimpleNamespace(collision_state=state))


def test_a_contact_pauses_steering_and_does_not_end_it():
    node, cuff, _, s = steer(squeezed=True)
    contact(node, True)
    assert not s.live(np.full(7, 1.0))
    assert node.warnings and not node.errors and not s.tripped
    node.timer()
    assert not node.published
    contact(node, False)
    assert s.live(np.zeros(7))


def test_a_trip_names_the_joint():
    node, _, _, s = steer(squeezed=True)
    q = np.zeros(7)
    q[2] = STEER['max_tracking_error'] + 0.01
    s.live(q)
    assert 'left_e0' in node.errors[0]


def test_softness_divides_the_damping_of_its_joints_live():
    node, _, _, s = steer()
    node.on_parameters([types.SimpleNamespace(name='steer_softness_arm', value=2.0)])
    damping = np.array(STEER['damping'], float)
    assert np.allclose(s.admittance.damping[:4], damping[:4] / 2)
    assert np.allclose(s.admittance.damping[4:], damping[4:])
    node.on_parameters([types.SimpleNamespace(name='steer_softness_wrist', value=1.5)])
    assert np.allclose(s.admittance.damping[4:], damping[4:] / 1.5)
    node.on_parameters([types.SimpleNamespace(name='steer_softness_arm', value=1.0)])
    assert np.allclose(s.admittance.damping[:4], damping[:4])
    assert len(node.infos) == 3


def test_the_admittance_is_published_only_while_steering_can_run():
    node, _, out, s = steer()
    msg = types.SimpleNamespace(
        header=Header(),
        name=joint_names('left'),
        actual_effort=[0.0] * 7,
        gravity_model_effort=[0.0] * 7,
        actual_position=[0.0] * 7,
    )
    effort = node.subscribed['/robot/limb/left/gravity_compensation_torques']
    effort(msg)
    assert node.published == ['/teleop/left/steer_state']
    out.active = False
    effort(msg)
    assert len(node.published) == 1


def follow(share=0.9, max_rate=1.0, max_offset=0.6):
    f = Follow(STEER['cutoff'], STEER['decay'], max_rate, max_offset)
    f.share[:] = share
    return f


@pytest.mark.parametrize('share', [0.0, 0.5, 0.9])
def test_following_divides_the_stiffness_by_what_it_leaves(share):
    """An arm on a spring k to its setpoint under a steady torque: it settles at torque / (k (1 - share))."""
    f, k, torque, plan = follow(share), 120.0, 6.0, np.zeros(7)
    q = plan.copy()
    for _ in range(20000):
        q = plan + f.step(DT, plan, q, True) + torque / k
    assert q[0] == pytest.approx(torque / (k * (1 - share)), rel=1e-3)


def test_a_share_of_zero_is_the_plain_position_hold():
    assert not follow(0.0).step(DT, np.zeros(7), np.full(7, 0.3), True).any()


def test_following_sets_each_joint_apart():
    node, _, _, s = steer()
    node.on_parameters([types.SimpleNamespace(name='steer_follow_wrist', value=0.8)])
    arm = STEER['follow']
    assert np.array_equal(s.follow.share, [arm, arm, arm, arm, 0.8, 0.8, 0.8])


def test_a_released_cuff_lets_the_followed_offset_decay():
    f = follow()
    f.offset[:] = 0.3
    for _ in range(int(4 * STEER['decay'] / DT)):
        f.step(DT, np.zeros(7), np.full(7, 0.5), False)
    assert np.abs(f.offset).max() < 0.3 * np.exp(-3.9)


def test_the_followed_offset_is_bounded():
    f = follow(0.9, max_rate=1.0, max_offset=0.6)
    last = np.zeros(7)
    for _ in range(2000):
        now = f.step(DT, np.zeros(7), np.full(7, 5.0), True).copy()
        assert np.abs(now - last).max() <= 1.0 * DT + 1e-12
        last = now
    assert np.abs(last).max() == pytest.approx(0.6)


def test_the_stream_passes_the_plan_and_a_stop_drops_what_followed():
    _, _, _, s = steer(squeezed=True)
    s.q_measured = np.full(7, 0.1)
    s.follow.share[:] = 0.5
    offset, rate = s(DT, np.zeros(7))
    assert offset[0] > 0 and not rate.any()
    s.reset()
    assert not s.follow.offset.any()


def test_the_tcp_offset_is_the_arm_while_held_and_the_command_after():
    """Published on kinesthetic/offset as a nudge's correction_offset: translation and rotation vector. Held, the
    arm's offset, not the share a of it the command takes; released, the command's decaying handback."""
    _, _, _, s = steer(squeezed=True)

    def tcp(q):  # a stand-in arm: q[0:3] is the position, q[3] turns about z
        c, n = np.cos(q[3]), np.sin(q[3])
        return q[:3].copy(), np.array([[c, -n, 0.0], [n, c, 0.0], [0.0, 0.0, 1.0]])

    assert not s.tcp_offset(tcp).any()  # nothing played yet
    s.q_measured = np.array([0.02, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0])
    s.follow.share[:] = 0.5
    for _ in range(400):
        added, _ = s(DT, np.zeros(7))
    assert added[0] == pytest.approx(0.5 * 0.02)  # the command takes half the way
    assert np.allclose(s.tcp_offset(tcp), [0.02, 0, 0, 0, 0, 0.1])  # held: the label is all of it
    s.cuff.squeezed = False
    added, _ = s(DT, np.zeros(7))
    assert np.allclose(s.tcp_offset(tcp), [added[0], 0, 0, 0, 0, added[3]])  # released: the handback
    s.reset()
    assert not s.tcp_offset(tcp).any()


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


def test_the_probe_catches_a_steering_sign_that_would_pull_into_the_hand():
    def rows(moved):
        row = np.zeros((10, 18))
        row[:, 1], row[:, 2], row[:, 3], row[:, 10] = 2, 1, -8.0, moved  # the spring carries -8: the hand's +8
        return row

    assert verdict(rows(0.01))['sign_agrees'] is True
    assert verdict(rows(-0.01))['sign_agrees'] is False
    assert verdict(np.zeros((10, 18)))['sign_agrees'] is None


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
