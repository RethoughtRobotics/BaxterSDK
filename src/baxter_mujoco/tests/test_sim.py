import numpy as np

from baxflow.sim import make_env


def test_baxter_steps():
    env = make_env(cameras=["frontview"])
    obs = env.reset()
    assert env.action_dim == 14  # 2 arms x (6-DoF OSC pose + gripper)
    obs, *_ = env.step(np.zeros(env.action_dim))
    assert obs["frontview_image"].shape == (96, 96, 3)
    assert "robot0_left_eef_pos" in obs and "robot0_right_eef_pos" in obs
