#!/bin/bash
# Build the workspace and its venvs inside container/baxter_sim.def's image, once (and after a pull that
# changes a package): colcon, .venv-ros, baxter_mujoco's uv env and RethoughtAI's (recorder, adapter,
# lerobot). Everything is built against the image's Python 3.12 (ROS Kilted's), so run the stack with
# the same image.
#   apptainer exec --nv --bind <dir with baxter_sdk and rethought_ai> baxter_sim.sif container/setup.sh
set -euo pipefail
WS_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
RAI_DIR="${RAI_DIR:-$(realpath -m "${WS_DIR}/../rethought_ai")}"
[[ -d "$RAI_DIR" ]] || { echo "No RethoughtAI at ${RAI_DIR}: clone it there or set RAI_DIR" >&2; exit 1; }
# The ROS Python everywhere: a uv-downloaded one could not import rclpy
export UV_PYTHON=/usr/bin/python3.12 UV_PYTHON_DOWNLOADS=never

set +u
source /opt/ros/kilted/setup.bash
set -u
cd "$WS_DIR"
# What the sim, the teleop and the recorder need; not MoveIt or the demos
colcon build --packages-up-to baxter_teleop baxter_tools baxter_interface baxter_external_devices \
    baxter_description rethink_ee_description baxter_core_msgs \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
[[ -x .venv-ros/bin/python ]] || /usr/bin/python3.12 -m venv --system-site-packages .venv-ros
.venv-ros/bin/pip install -q -r src/baxter_teleop/requirements.txt
(cd src/baxter_mujoco && uv sync --group ros)
(cd "$RAI_DIR" && uv sync)
echo "Built ${WS_DIR} and ${RAI_DIR}"
