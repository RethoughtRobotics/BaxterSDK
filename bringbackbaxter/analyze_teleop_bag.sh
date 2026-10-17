#!/bin/bash
# Analyze teleop bags from record_teleop.sh: per-segment and summary metrics (bag_report.py), a
# side-by-side comparison for several bags, and for sim bags the contacts the arms made (BaxFlow-T).
# Usage: ./analyze_teleop_bag.sh <bag> [<bag> ...]     e.g. ./analyze_teleop_bag.sh bags/*_baseline bags/*_closed_loop

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAXFLOW_DIR="${BAXFLOW_DIR:-$(realpath "${SCRIPT_DIR}/../BaxFlow-T")}"
set +u
source /opt/ros/kilted/setup.bash
source "${SCRIPT_DIR}/ros2_ws/install/setup.bash"
set -u

"${SCRIPT_DIR}/ros2_ws/.venv-ros/bin/python" -m baxter_interface.baxter_teleop.bag_report "$@" 2> >(grep -v -e jax -e GPU >&2)
for bag in "$@"; do
    if [[ "$bag" == *_sim_* && -d "$BAXFLOW_DIR" ]]; then
        echo -e "\n=== $(basename "$bag")"
        npz="$(realpath "$bag")/joints.npz"
        (cd "$BAXFLOW_DIR" && uv run python -m baxflow.contacts "$npz" 2> >(grep -v robosuite >&2))
    fi
done
