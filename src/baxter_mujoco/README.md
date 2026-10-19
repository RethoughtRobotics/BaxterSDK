# baxter_mujoco

Baxter in MuJoCo (both arms, electric grippers, via [robosuite](https://robosuite.ai/docs/modules/robots.html)),
exposed on the real robot's ROS 2 interface: the same joint command, joint state and endpoint topics, so
the teleop, `tuck_arms` and RViz drive it exactly like the robot. Baxter's joint loop is emulated as
computed torque, so its tracking lag is close to the real arm's.

This is a [uv](https://docs.astral.sh/uv/) project, not a colcon package (`COLCON_IGNORE`): robosuite and
MuJoCo are not available through rosdep. MuJoCo is pinned to 3.3.x because robosuite 1.5.2 breaks on newer
releases.

```bash
cd src/baxter_mujoco
uv sync                      # .venv with robosuite and MuJoCo
./launch_sim.sh              # viewer only, Baxter holding its home pose
./launch_sim.sh --ros        # the simulated robot on ROS 2 (needs this workspace built)
uv run pytest                # smoke test
```

`--ros` options: `--no-viewer`, `--empty` (scene object moved out of reach, for controller tests),
`--no-table`. Use a separate `ROS_DOMAIN_ID` from the real robot.

`python -m baxter_mujoco.contacts <bag>/joints.npz` replays recorded joint angles kinematically and lists
the contacts the arms made.
