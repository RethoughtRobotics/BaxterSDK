# Headless sim image

Closed-loop policy evaluations in the MuJoCo sim on a GPU machine with no display, several at once.
Simulation only: the sim, the teleop servo following a policy's targets, a RethoughtAI policy server and
`baxter_mujoco.datagen.evaluate`. Nothing for the real robot.

The image holds the dependencies (ROS 2 Kilted, Zenoh, apt Pinocchio, EGL, uv). The code and its venvs
come from the `FinalProject` checkout (this repository beside `rethought_ai`), mounted at its own path;
the first run builds the workspace and the `.venv-ros` and `baxter_mujoco` venvs into it.

```bash
docker build -t baxter-sim:latest docker/sim
docker/sim/eval.sh --gpu 1 --name red_r1 /path/to/checkpoint \
    --cup-color red --seeds 1000-1004 --out ~/FinalProject/data/sim_eval/x/red_r1.jsonl \
    --bags ~/FinalProject/data/sim_eval/x/red_r1_bags --bag-images
```

Each container has its own network, so its Zenoh router and ROS domain are its own: start as many as the
GPU and cores allow. The stack's logs go to `~/.cache/baxter_sim/<name>/`. There is no viewer; the bags
hold the cameras.
