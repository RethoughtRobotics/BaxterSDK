# BaxFlow-T
Push-T with bi-manual flow matching using Mujoco
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2ed81375-3203-4841-811c-fdd0be3d791d" />

## Setup

Bimanual Baxter in MuJoCo via [robosuite](https://robosuite.ai/docs/modules/robots.html). Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # create .venv with robosuite + mujoco
./launch_sim.sh                  # open the viewer (Baxter holding its home pose)
uv run pytest                    # smoke test
```

`baxflow.sim.make_env()` returns the Baxter env: 14-D action (per arm: 6-D OSC pose delta + gripper).
MuJoCo is pinned to 3.3.x — robosuite 1.5.2 breaks on newer releases.

## ROS 2 simulator

With ROS 2 Kilted and the `bringbackbaxter/ros2_ws` workspace built, run:

```bash
./launch_sim.sh --ros
```

This exposes the simulated Baxter through the same joint command, state, and endpoint topics as
the robot. Set `BAXTER_WS` if the workspace is not at `../bringbackbaxter/ros2_ws`.
