#!/bin/bash
# Wrapper script to run diff_ik_teleop from the ROS 2 workspace venv

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}" && pwd)"

# Activate venv
source "${WS_DIR}/ros2_ws/.venv/bin/activate"

# Source ROS 2 setup
source "${WS_DIR}/ros2_ws/install/setup.bash"

# Run the node
"${WS_DIR}/ros2_ws/.venv/bin/python3" -m baxter_examples.diff_ik_teleop "$@"
