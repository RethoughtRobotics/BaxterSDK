#!/bin/bash
# Run trajectory_test once per IK solver on the simulated Baxter (no objects in reach), record each
# run, and write a report (figures, tables) comparing them.
# Usage: ./compare_ik.sh [solver ...]   (default: dls bounded)  ->  data/ik_comparisons/<time>_ik_comparison/
# Stops any running ROS processes first (teardown.sh) and again at the end.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_DIR="${SIM_DIR:-$(realpath "${SCRIPT_DIR}/../src/baxter_mujoco")}"
SOLVERS=("${@:-dls}")
[[ $# -eq 0 ]] && SOLVERS=(dls bounded)
set +u
source /opt/ros/kilted/setup.bash
source "${SCRIPT_DIR}/../install/setup.bash"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ZENOH_ROUTER_CHECK_ATTEMPTS=-1
unset ROS_DOMAIN_ID
LOG="$(mktemp -d)"
trap '"${SCRIPT_DIR}/teardown.sh" >/dev/null' EXIT

bags=()
for ik in "${SOLVERS[@]}"; do
    echo "== ${ik}: starting the sim"
    "${SCRIPT_DIR}/teardown.sh" >/dev/null
    setsid ros2 run rmw_zenoh_cpp rmw_zenohd >/dev/null 2>&1 &
    (cd "$SIM_DIR" && setsid ./launch_sim.sh --ros --no-viewer --empty >"${LOG}/sim_${ik}.log" 2>&1 &)
    until timeout 5 ros2 topic echo --once /robot/joint_states >/dev/null 2>&1; do sleep 1; done
    ros2 run baxter_tools tuck_arms -u >/dev/null 2>&1
    setsid "${SCRIPT_DIR}/record_teleop.sh" "trajectory_${ik}" >"${LOG}/record_${ik}.log" 2>&1 &
    sleep 4
    "${SCRIPT_DIR}/../.venv-ros/bin/python" -m baxter_interface.baxter_teleop.trajectory_test --ik "$ik" \
        2>"${LOG}/teleop_${ik}.log" | grep -v '^\['
    pkill -INT -f '[r]os2 bag record' || true
    until grep -q '^Saved ' "${LOG}/record_${ik}.log"; do sleep 1; done
    bags+=("$(grep '^Saved ' "${LOG}/record_${ik}.log" | cut -d' ' -f2)")
done

DATA_DIR="${DATA_DIR:-$(realpath "${SCRIPT_DIR}/../../data")}"  # FinalProject/data: every recording and result
REPORT="${DATA_DIR}/ik_comparisons/$(date +%Y%m%d_%H%M%S)_ik_comparison"
"${SCRIPT_DIR}/../.venv-ros/bin/python" -m baxter_interface.baxter_teleop.ik_report "$REPORT" "${bags[@]}" \
    2> >(grep -v -e jax -e GPU >&2)
{
    echo -e "\n## Contacts\n"
    echo "Kinematic replay of the recorded joint angles in the MuJoCo scene (baxter_mujoco.contacts):"
    for bag in "${bags[@]}"; do
        echo -e "\n\`$(basename "$bag")\`:\n\n\`\`\`"
        (cd "$SIM_DIR" && uv run python -m baxter_mujoco.contacts "${bag}/joints.npz" 2>/dev/null | sed '/^$/d')
        echo '```'
    done
} >> "${REPORT}/README.md"
echo "Report: ${REPORT}"
