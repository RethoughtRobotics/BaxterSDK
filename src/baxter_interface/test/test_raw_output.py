"""RawJointOutput (activation, staleness, lead backstop, thread) and BaxterRawSink (messages)."""

import time

import numpy as np
import pytest

from baxter_interface.baxter_teleop.joint_setpoint import JointLimits, JointSetpointGenerator
from baxter_interface.baxter_teleop.raw_output import BaxterRawSink, RawJointOutput

DT = 0.005
VEL = np.array([1.2, 1.2, 1.2, 1.2, 3.2, 3.2, 3.2])
ACC = np.array([4.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0])
LOW = np.array([-1.70168, -2.147, -3.05418, 0.35, -3.059, -1.5708, -3.059])
HIGH = np.array([1.70168, 1.047, 3.05418, 2.618, 3.059, 2.094, 3.059])
Q0 = np.array([0.08, -1.0, 1.19, 1.94, -0.67, 1.03, 0.5])
NAMES = [f'right_{j}' for j in ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')]


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(**kwargs):
    sent, clock = [], FakeClock()
    generator = JointSetpointGenerator(JointLimits(VEL, ACC, 10 * ACC, LOW, HIGH), DT)
    return RawJointOutput(generator, sent.append, clock=clock, **kwargs), sent, clock


def ticks(output, clock, n):
    out = []
    for _ in range(n):
        clock.now += DT
        out.append(output.tick())
    return out


def test_inactive_sends_nothing():
    output, sent, clock = make()
    assert ticks(output, clock, 10) == [None] * 10
    assert sent == [] and not output.velocity.any()


def test_first_command_starts_at_rest_from_the_measured_arm():
    output, sent, clock = make()
    output.command(np.full(7, 0.5), Q0)
    first = ticks(output, clock, 1)[0]
    assert output.active and sent == [first]
    assert np.all(np.abs(first.position - Q0) < 1e-4)  # one jerk-limited step from rest
    assert np.all(first.velocity < 0.5)


def test_follows_the_commanded_velocity_while_commands_keep_coming():
    output, sent, clock = make()
    target = np.array([0.3, -0.2, 0.1, 0.0, 0.5, -0.5, 0.2])
    for _ in range(60):  # 0.6 s, a command every 2 ticks (the teleop is ~3.5 ticks per cycle)
        output.command(target, sent[-1].position if sent else Q0)
        ticks(output, clock, 2)
    np.testing.assert_allclose(sent[-1].velocity, target, atol=1e-9)


def test_position_is_the_streamed_setpoint_while_active_and_none_otherwise():
    output, sent, clock = make()
    assert output.position is None
    output.command(np.full(7, 0.5), Q0)
    ticks(output, clock, 10)
    np.testing.assert_allclose(output.position, sent[-1].position)
    output.stop()
    assert output.position is None


def test_stale_commands_decelerate_to_a_hold():
    output, sent, clock = make(stale_after=0.1)
    output.command(np.full(7, 0.5), Q0)
    ticks(output, clock, 10)  # 50 ms: still fresh
    assert sent[-1].velocity[0] > 0
    ticks(output, clock, 200)  # control loop stalled for 1 s
    np.testing.assert_allclose(sent[-1].velocity, 0.0, atol=1e-9)
    held = sent[-1].position.copy()
    ticks(output, clock, 10)
    np.testing.assert_allclose(sent[-1].position, held)


def test_stop_halts_streaming_and_the_next_command_restarts_from_the_arm():
    output, sent, clock = make()
    output.command(np.full(7, 0.5), Q0)
    ticks(output, clock, 20)
    output.stop()
    n = len(sent)
    assert ticks(output, clock, 5) == [None] * 5 and len(sent) == n
    q_now = Q0 + 0.05
    output.command(np.zeros(7), q_now)
    np.testing.assert_allclose(ticks(output, clock, 1)[0].position, q_now, atol=1e-9)


def test_setpoint_never_leads_the_measured_arm_by_more_than_max_lead():
    output, sent, clock = make(max_lead=0.1, stale_after=10.0)
    output.command(np.full(7, 1.0), Q0)  # the arm is blocked: measured stays at Q0
    ticks(output, clock, 200)
    assert np.all(np.abs(np.array([s.position for s in sent]) - Q0) <= 0.1 + 1e-12)


def test_thread_streams_at_the_generator_rate_and_closes():
    sent = []
    generator = JointSetpointGenerator(JointLimits(VEL, ACC, 10 * ACC, LOW, HIGH), DT)
    output = RawJointOutput(generator, sent.append).start()
    try:
        output.command(np.zeros(7), Q0)
        time.sleep(0.25)
        output.command(np.zeros(7), Q0)  # keep it fresh
        time.sleep(0.25)
    finally:
        output.close()
    assert 70 <= len(sent) <= 110  # ~100 at 200 Hz, loose for CI scheduling
    n = len(sent)
    time.sleep(0.05)
    assert len(sent) == n  # closed: no more sends


class FakePublisher:
    def __init__(self, topic):
        self.topic, self.messages = topic, []

    def publish(self, msg):
        self.messages.append(msg)


class FakeNode:
    def __init__(self):
        self.publishers = {}

    def create_publisher(self, msg_type, topic, qos):
        self.publishers[topic] = FakePublisher(topic)
        return self.publishers[topic]


@pytest.fixture
def ros_msgs():
    return pytest.importorskip('baxter_core_msgs.msg'), pytest.importorskip('trajectory_msgs.msg')


def test_sink_sends_raw_position_and_inverse_dynamics_in_limb_order(ros_msgs):
    node = FakeNode()
    limb_order = list(reversed(NAMES))  # the feedforward point must follow the limb's order
    sink = BaxterRawSink(node, 'right', NAMES, limb_order, inverse_dynamics=True)
    output, _, clock = make()
    output.send = sink
    output.command(np.full(7, 0.5), Q0)
    sp = ticks(output, clock, 3)[-1]
    cmd = node.publishers['/robot/limb/right/joint_command'].messages[-1]
    ff = node.publishers['/robot/limb/right/inverse_dynamics_command'].messages[-1]
    assert cmd.mode == ros_msgs[0].JointCommand.RAW_POSITION_MODE and list(cmd.names) == NAMES
    np.testing.assert_allclose(cmd.command, sp.position)
    np.testing.assert_allclose(ff.positions, sp.position[::-1])
    np.testing.assert_allclose(ff.velocities, sp.velocity[::-1])
    np.testing.assert_allclose(ff.accelerations, sp.acceleration[::-1])


def test_sink_without_inverse_dynamics_sends_only_the_position(ros_msgs):
    node = FakeNode()
    BaxterRawSink(node, 'right', NAMES, NAMES, inverse_dynamics=False)
    assert list(node.publishers) == ['/robot/limb/right/joint_command']
