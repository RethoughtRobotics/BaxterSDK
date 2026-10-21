"""The expert taking over mid-attempt (CupTask.replan) and its chunks (datagen.expert), without the sim;
the bag reading on a bag written here."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from baxter_mujoco import scenes
from baxter_mujoco.datagen import expert
from baxter_mujoco.datagen.cup import GRASP_QUAT, GRASP_Z, GRIP_CLOSED, HOVER, RELEASE_Z, WALL, CupTask

HOME = (np.array([0.574, 0.225, 0.103]), Rotation.from_quat([-0.054, 0.997, -0.001, 0.011]))
LAYOUT = [0.75, 0.15, 0.6, -0.15, 0.0]
UPRIGHT = np.array([0.0, 0.0, 0.0, 1.0])
DOWN = Rotation.from_quat(GRASP_QUAT)
TO_WALL = np.r_[WALL * DOWN.apply([0.0, 1.0, 0.0])[:2], 0.0]
CUP, BOX = np.array([*LAYOUT[:2], scenes.TABLE_TOP]), np.array([*LAYOUT[2:4], scenes.TABLE_TOP])
GRASP = CUP + TO_WALL + [0, 0, GRASP_Z - scenes.TABLE_TOP]  # the TCP at the grasp
RELEASE = np.r_[BOX[:2] + TO_WALL[:2], RELEASE_Z]


def objects(cup=CUP, cup_quat=UPRIGHT, box=BOX):
    return {'cup': (np.asarray(cup, float), cup_quat), 'container': (np.asarray(box, float), UPRIGHT)}


def held_cup(tcp):
    """The cup's pose held at the TCP."""
    return np.asarray(tcp) - TO_WALL - [0, 0, GRASP_Z - scenes.TABLE_TOP]


def test_replan_at_home_is_the_plan():
    replan = CupTask().replan(HOME, 0.0, objects(), HOME, LAYOUT)
    plan = CupTask().plan(LAYOUT, HOME)
    assert replan.wait == 0 and replan.grip == 0
    assert len(replan.waypoints) == len(plan)
    for a, b in zip(replan.waypoints, plan):
        assert np.allclose(a.position, b.position) and (a.grip, a.slow, a.hold) == (b.grip, b.slow, b.hold)


def test_chunk_from_home_follows_the_plan():
    c = expert.expert_chunk(CupTask(), HOME, 0.0, objects(), HOME, layout=LAYOUT)
    assert c.shape == (expert.HORIZON, 8)
    assert np.allclose(c[0, :3], HOME[0]) and np.all(c[:, 4] >= 0) and np.allclose(c[:, 7], 0.0)
    hover = np.r_[GRASP[:2], GRASP_Z + HOVER]
    d = np.linalg.norm(c[:, :3] - HOME[0], axis=1)
    assert np.all(np.diff(d) >= -1e-9)  # straight on towards the cup
    toward = (c[-1, :3] - HOME[0]) / np.linalg.norm(c[-1, :3] - HOME[0])
    assert np.allclose(toward, (hover - HOME[0]) / np.linalg.norm(hover - HOME[0]), atol=1e-3)
    assert np.allclose(np.linalg.norm(c[:, 3:7], axis=1), 1.0)


def test_an_arm_on_the_experts_move_continues_it():
    """Taken over on the approach, the chunk is the full plan's own, from that time."""
    full = expert.chunk(CupTask().replan(HOME, 0.0, objects(), HOME, LAYOUT), HOME, 0.0, GRIP_CLOSED, horizon=60)
    k = 20
    velocity = (full[k, :3] - full[k - 1, :3]) * expert.FPS
    tcp = (full[k, :3], Rotation.from_quat(full[k, 3:7]))
    later = expert.expert_chunk(CupTask(), tcp, 0.0, objects(), HOME, velocity, LAYOUT, horizon=30)
    assert np.abs(later[:, :3] - full[k : k + 30, :3]).max() < 2e-3


def test_carrying_goes_on_to_the_container_and_lets_go():
    tcp = (np.array([0.68, 0.0, GRASP_Z + HOVER]), DOWN)
    replan = CupTask().replan(tcp, GRIP_CLOSED, objects(cup=held_cup(tcp[0])), HOME, LAYOUT)
    assert replan.grip == 1.0 and replan.wait == 0
    assert np.allclose(replan.waypoints[0].position, np.r_[RELEASE[:2], GRASP_Z + HOVER])
    opening = next(w for w in replan.waypoints if w.grip == 0.0)
    assert np.allclose(opening.position, RELEASE)
    c = expert.expert_chunk(CupTask(), tcp, GRIP_CLOSED, objects(cup=held_cup(tcp[0])), HOME, layout=LAYOUT)
    assert np.allclose(c[:, 7], GRIP_CLOSED, atol=1e-6)


def test_held_low_lifts_first_and_at_the_release_opens():
    low = (GRASP + [0, 0, 0.03], DOWN)
    replan = CupTask().replan(low, GRIP_CLOSED, objects(cup=held_cup(low[0])), HOME, LAYOUT)
    assert np.allclose(replan.waypoints[0].position, np.r_[GRASP[:2], GRASP_Z + HOVER])
    at = (RELEASE, DOWN)
    replan = CupTask().replan(at, GRIP_CLOSED, objects(cup=held_cup(RELEASE)), HOME, LAYOUT)
    assert replan.grip == 0.0 and replan.wait > 0
    c = expert.expert_chunk(CupTask(), at, GRIP_CLOSED, objects(cup=held_cup(RELEASE)), HOME, layout=LAYOUT)
    assert c[-1, 7] < 0.1 and np.allclose(c[:10, :3], RELEASE)  # opens, holding still


def test_closing_at_the_grasp_finishes_the_close():
    at = (GRASP, DOWN)
    replan = CupTask().replan(at, 0.3, objects(), HOME, LAYOUT)
    assert replan.grip == 1.0 and 0 < replan.wait < 1.0
    c = expert.expert_chunk(CupTask(), at, 0.3, objects(), HOME, layout=LAYOUT)
    assert c[-1, 7] == pytest.approx(GRIP_CLOSED, abs=0.01) and np.all(np.diff(c[:, 7]) >= -1e-9)


def test_missed_grasp_opens_rises_and_grasps_again():
    beside = (GRASP + [0.06, 0.0, 0.0], DOWN)  # closed beside the cup
    replan = CupTask().replan(beside, GRIP_CLOSED, objects(), HOME, LAYOUT)
    assert replan.grip == 0.0 and replan.wait > 0
    first, *rest = replan.waypoints
    assert np.allclose(first.position, np.r_[beside[0][:2], GRASP_Z + HOVER]) and first.slow
    assert any(w.grip == 1.0 and np.allclose(w.position, GRASP) for w in rest)
    c = expert.expert_chunk(CupTask(), beside, GRIP_CLOSED, objects(), HOME, layout=LAYOUT)
    assert c[-1, 7] < c[0, 7] and c[-1, 2] >= c[0, 2]


def test_released_in_the_container_goes_home():
    tcp = (RELEASE + [0, 0, 0.02], DOWN)
    inside = BOX + [0, 0, 0.005]
    replan = CupTask().replan(tcp, 0.0, objects(cup=inside), HOME, LAYOUT)
    assert np.allclose(replan.waypoints[-1].position, HOME[0]) and all(w.grip == 0 for w in replan.waypoints)


def test_no_plan_for_a_fallen_cup():
    lying = Rotation.from_euler('x', 90, degrees=True).as_quat()
    assert CupTask().replan(HOME, 0.0, objects(cup_quat=lying), HOME, LAYOUT) is None
    assert expert.expert_chunk(CupTask(), HOME, 0.0, objects(cup=CUP - [0, 0, 0.5]), HOME) is None


def write_bag(path, frames=150, rate=100.0):
    """An evaluator bag of an attempt in which the arm stays at home: TCP and gripper at `rate`, objects
    and the engage as the evaluator's bags have them."""
    import rosbag2_py
    from baxter_core_msgs.msg import EndEffectorState
    from geometry_msgs.msg import Pose, PoseArray, PoseStamped
    from rclpy.serialization import serialize_message
    from std_msgs.msg import Bool

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap', storage_preset_profile='zstd_fast'),
        rosbag2_py.ConverterOptions('', ''),
    )
    topics = {
        '/teleop/left/tcp': 'geometry_msgs/msg/PoseStamped',
        '/robot/end_effector/left_gripper/state': 'baxter_core_msgs/msg/EndEffectorState',
        '/sim/objects': 'geometry_msgs/msg/PoseArray',
        '/policy/left/engaged': 'std_msgs/msg/Bool',
    }
    for i, (name, kind) in enumerate(topics.items()):
        writer.create_topic(rosbag2_py.TopicMetadata(i, name, kind, 'cdr'))
    t0 = 1_000_000_000

    def pose(p, q):
        m = Pose()
        m.position.x, m.position.y, m.position.z = map(float, p)
        m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w = map(float, q)
        return m

    writer.write('/policy/left/engaged', serialize_message(Bool(data=False)), t0)
    writer.write('/policy/left/engaged', serialize_message(Bool(data=True)), t0 + 300_000_000)
    for k in range(int(frames / 25 * rate)):
        t = t0 + int(k * 1e9 / rate)
        tcp = PoseStamped()
        tcp.pose = pose(HOME[0], HOME[1].as_quat())
        writer.write('/teleop/left/tcp', serialize_message(tcp), t)
        writer.write('/robot/end_effector/left_gripper/state', serialize_message(EndEffectorState(position=100.0)), t)
        if k % 10 == 0:
            objs = PoseArray(poses=[pose(CUP, UPRIGHT), pose(BOX, UPRIGHT)])
            writer.write('/sim/objects', serialize_message(objs), t)
    writer.write('/policy/left/engaged', serialize_message(Bool(data=False)), t0 + int(5.5e9))
    del writer


def test_expert_npz_from_a_bag(tmp_path):
    bag = tmp_path / 'seed_0500'
    write_bag(bag)
    path = expert.write(bag, CupTask(), horizon=8)
    z = np.load(path)
    n = len(z['t'])
    assert n == int(5.2 * expert.FPS) + 1  # engage (0.3 s) to release (5.5 s)
    assert np.all(np.diff(z['t']) == 40_000_000) and z['t'][0] == 1_300_000_000
    assert z['chunk'].shape == (n, 8, 8) and z['valid'].all()
    assert np.allclose(z['tcp'][:, :3], HOME[0]) and np.allclose(z['gripper'], 0.0)
    assert np.allclose(z['cup'][:, :3], CUP) and np.allclose(z['container'][:, :3], BOX)
    assert np.allclose(z['home'][:3], HOME[0])
    assert json.loads(str(z['meta']))['horizon'] == 8
    first = expert.expert_chunk(CupTask(), HOME, 0.0, objects(), HOME, layout=LAYOUT, horizon=8)
    assert np.allclose(z['chunk'][0], first, atol=1e-6)
