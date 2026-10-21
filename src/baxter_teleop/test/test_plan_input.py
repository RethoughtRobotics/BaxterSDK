"""Plan input and replay stream: a timed TCP plan played as replay plays a demo; no robot or ROS graph."""

import types

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation

from baxter_teleop.inputs.plan import PlanTarget
from baxter_teleop.kinematics import ArmKinematics
from baxter_teleop.outputs.replay_stream import ReplayStream
from baxter_teleop.replay import Trajectory

HOME = np.array([-0.1124, -1.0097, -1.1317, 1.9033, 0.5854, 1.0585, -0.7055])


def joints(t):
    """A smooth joint motion from home: every joint swinging by up to 0.2 rad."""
    return HOME + 0.2 * np.sin(np.outer(t, np.arange(1, 8) / 4.0))


def fake_node(clock):
    subs = {}
    node = types.SimpleNamespace(
        create_subscription=lambda kind, topic, cb, qos: subs.setdefault(topic, cb),
        get_clock=lambda: types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=int(clock[0] * 1e9))),
    )
    return node, subs


def plan_msg(kinematics, t):
    msg = Path()
    for tk, q in zip(t, joints(t - t[0])):
        p, R = kinematics.tcp(q)
        pose = PoseStamped()
        pose.header.stamp.sec, pose.header.stamp.nanosec = int(tk), int(round((tk % 1) * 1e9))
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, p)
        o = pose.pose.orientation
        o.x, o.y, o.z, o.w = map(float, Rotation.from_matrix(R).as_quat())
        msg.poses.append(pose)
    return msg


def test_the_stream_sends_replays_sample_and_nothing_once_stopped():
    sent, clock = [], [100.0]
    stream = ReplayStream(sent.append, lambda: clock[0])
    t = np.linspace(0.0, 1.0, 26)
    trajectory = Trajectory(t, joints(t))
    stream.play(trajectory, 99.5)
    setpoint = stream.tick()
    q, qd, qdd = trajectory(0.5)
    np.testing.assert_allclose(setpoint.position, q)
    np.testing.assert_allclose(setpoint.velocity, qd)
    np.testing.assert_allclose(setpoint.acceleration, qdd)
    stream.stop()
    assert stream.tick() is None and stream.position is None and len(sent) == 1


def test_a_plan_becomes_a_replay_trajectory_along_its_tcp():
    clock = [10.0]
    node, subs = fake_node(clock)
    kinematics = ArmKinematics('left')
    target = PlanTarget(node, '/policy/left/path', kinematics)
    t = 10.0 + 0.04 * np.arange(17)  # a 0.64 s plan from now, as the runtime publishes
    subs['/policy/left/path'](plan_msg(kinematics, t))
    trajectory, t0 = target.plan(HOME, None)
    assert t0 == 10.0 and target._moving and target.goal is not None
    q, _, _ = trajectory(0.32)
    planned = kinematics.tcp(joints(np.array([0.32]))[0])[0]
    np.testing.assert_allclose(kinematics.tcp(q)[0], planned, atol=1e-3)  # on the plan's TCP within 1 mm
    assert target.plan(HOME, None)[0] is trajectory  # the same plan is not rebuilt


def test_a_stale_or_reset_plan_holds():
    clock = [10.0]
    node, subs = fake_node(clock)
    kinematics = ArmKinematics('left')
    target = PlanTarget(node, '/policy/left/path', kinematics)
    subs['/policy/left/path'](plan_msg(kinematics, 10.0 + 0.04 * np.arange(17)))
    clock[0] = 11.0  # 0.36 s past the plan's end
    assert target.plan(HOME, None) is None and not target._moving
    clock[0] = 10.1
    target.reset()
    assert target.plan(HOME, None) is None


def test_a_new_plan_takes_over_without_a_velocity_jump():
    """Consecutive plans along one motion: at the switch the played velocity is continuous."""
    clock = [10.0]
    node, subs = fake_node(clock)
    kinematics = ArmKinematics('left')
    target = PlanTarget(node, '/policy/left/path', kinematics)
    first = 10.0 + 0.04 * np.arange(17)
    subs['/policy/left/path'](plan_msg(kinematics, first))
    clock[0] = 10.04
    old, old_t0 = target.plan(HOME, None)
    q_now = old(10.04 - old_t0)[0]
    subs['/policy/left/path'](
        plan_msg_from(kinematics, 10.04 + 0.04 * np.arange(17), 10.0)
    )  # the same motion, a step on
    new, new_t0 = target.plan(HOME, q_now)
    np.testing.assert_allclose(new(10.06 - new_t0)[1], old(10.06 - old_t0)[1], atol=0.02)  # rad/s


def plan_msg_from(kinematics, t, origin):
    """plan_msg for times t of the motion that started at `origin`."""
    msg = Path()
    for tk, q in zip(t, joints(t - origin)):
        p, R = kinematics.tcp(q)
        pose = PoseStamped()
        pose.header.stamp.sec, pose.header.stamp.nanosec = int(tk), int(round((tk % 1) * 1e9))
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, p)
        o = pose.pose.orientation
        o.x, o.y, o.z, o.w = map(float, Rotation.from_matrix(R).as_quat())
        msg.poses.append(pose)
    return msg
