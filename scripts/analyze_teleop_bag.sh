#!/bin/bash
# Analyze teleop bags from record_teleop.sh: per-segment and summary metrics (bag_report.py), a
# side-by-side comparison for several bags, and for sim bags the contacts the arms made (baxter_mujoco).
# Usage: ./analyze_teleop_bag.sh <bag> [<bag> ...]     e.g. ./analyze_teleop_bag.sh ../data/bags/*_baseline ../data/bags/*_closed_loop

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="${SIM_DIR:-$(realpath "${SCRIPT_DIR}/../src/baxter_mujoco")}"
set +u
source /opt/ros/kilted/setup.bash
source "${SCRIPT_DIR}/../install/setup.bash"
set -u

"${SCRIPT_DIR}/../.venv-ros/bin/python" -m baxter_interface.baxter_teleop.bag_report "$@" 2> >(grep -v -e jax -e GPU >&2)
for bag in "$@"; do
    if [[ "$bag" == *_sim_* && -d "$SIM_DIR" ]]; then
        echo -e "\n=== $(basename "$bag")"
        npz="$(realpath "$bag")/joints.npz"
        (cd "$SIM_DIR" && uv run python -m baxter_mujoco.contacts "$npz" 2> >(grep -v robosuite >&2))
    fi
done
