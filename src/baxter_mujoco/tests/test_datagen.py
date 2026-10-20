"""The scripted tasks' pure parts: layouts, plans and success, without ROS or the sim."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from baxter_mujoco import scenes
from baxter_mujoco.datagen.cup import GRASP_Z, CupTask
from baxter_mujoco.datagen.motion import min_jerk, move_duration

HOME = (np.array([0.574, 0.225, 0.103]), Rotation.from_quat([-0.054, 0.997, -0.001, 0.011]))


def test_min_jerk_ends():
    assert min_jerk(0.0) == 0.0 and min_jerk(1.0) == pytest.approx(1.0)


def test_slow_moves_take_longer():
    goal = (HOME[0] + [0.2, 0, 0], HOME[1])
    assert move_duration(HOME, goal, slow=True) > move_duration(HOME, goal)


def test_layouts_reproducible_and_inside_real_ranges():
    a, b = CupTask().sample(np.random.default_rng(3)), CupTask().sample(np.random.default_rng(3))
    assert a == b
    (x0, x1), (y0, y1) = scenes.CUP_RANGE
    assert x0 <= a[0] <= x1 and y0 <= a[1] <= y1
    assert np.hypot(a[0] - a[2], a[1] - a[3]) >= scenes.MIN_SEPARATION


def test_plan_ends_at_home_and_grasps_at_real_height():
    layout = CupTask().sample(np.random.default_rng(0))
    plan = CupTask().plan(layout, HOME)
    assert np.allclose(plan[-1].position, HOME[0]) and plan[-1].grip == 0.0
    closing = next(w for w in plan if w.grip == 1.0)
    assert closing.position[2] == pytest.approx(GRASP_Z)
    assert np.allclose(closing.position[:2], layout[:2])


def test_success_only_inside_container():
    layout = [0.8, 0.2, 0.7, -0.1, 0.0]
    task = CupTask()
    assert task.success(layout, {'cup': [0.7, -0.1, -0.055], 'container': [0.7, -0.1, -0.06]})
    assert not task.success(layout, {'cup': [0.8, 0.2, -0.06], 'container': [0.7, -0.1, -0.06]})
