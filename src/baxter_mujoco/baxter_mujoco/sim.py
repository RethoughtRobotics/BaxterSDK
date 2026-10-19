"""Baxter simulation in MuJoCo via robosuite.

Run `uv run python -m baxflow.sim` to open the viewer.
"""

import numpy as np
import robosuite as suite


def make_env(task="TwoArmLift", render=False, cameras=None, image_size=96, **kwargs):
    """Bimanual Baxter env. `cameras` (e.g. ["frontview"]) enables image observations."""
    return suite.make(
        task,
        robots="Baxter",
        env_configuration="single-robot",
        has_renderer=render,
        has_offscreen_renderer=bool(cameras),
        use_camera_obs=bool(cameras),
        camera_names=cameras or "frontview",
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=20,
        **kwargs,
    )


if __name__ == "__main__":
    env = make_env(render=True)
    env.reset()
    while True:
        env.step(np.zeros(env.action_dim))  # zero delta = hold pose
        env.render()
