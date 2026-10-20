#!/bin/bash
# Compare IK solvers on the sim: one scripted trajectory each, then a report with figures.
# Usage: scripts/compare_ik.sh [SOLVER ...]
#   SOLVER  dls or bounded (default: both); report in data/ik_comparisons/<time>_ik_comparison/

# Stops any running ROS processes first (teardown.sh) and again at the end. It runs on domain 0
# with no guard: never run it while the bridge or the robot is up.

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
set -e
SOLVERS=("${@:-dls}")
[[ $# -eq 0 ]] && SOLVERS=(dls bounded)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
unset ROS_DOMAIN_ID
LOG="$(mktemp -d)"
trap '"${SCRIPT_DIR}/teardown.sh" >/dev/null' EXIT

bags=()
for ik in "${SOLVERS[@]}"; do
    echo "== ${ik}: starting the sim"
    "${SCRIPT_DIR}/teardown.sh" >/dev/null
    setsid ros2 run rmw_zenoh_cpp rmw_zenohd >/dev/null 2>&1 &
    (cd "$SIM_DIR" && setsid ./launch_sim.sh --ros --no-viewer --empty >"${LOG}/sim_${ik}.log" 2>&1 &)
    deadline=$((SECONDS + 120))
    until timeout 5 ros2 topic echo --once /robot/joint_states >/dev/null 2>&1; do
        ((SECONDS < deadline)) || { echo "ERROR: no /robot/joint_states from the sim in 120 s (${LOG}/sim_${ik}.log)" >&2; exit 1; }
        sleep 1
    done
    ros2 run baxter_tools tuck_arms -u >/dev/null 2>&1
    setsid "${SCRIPT_DIR}/record_teleop.sh" "trajectory_${ik}" >"${LOG}/record_${ik}.log" 2>&1 &
    sleep 4
    "$PY" -m baxter_teleop.diagnostics.scripted_trajectory --ik "$ik" \
        2>"${LOG}/teleop_${ik}.log" | grep -v '^\['
    pkill -INT -f '[r]os2 bag record' || true
    for _ in $(seq 60); do grep -q '^Saved ' "${LOG}/record_${ik}.log" && break; sleep 1; done
    grep -q '^Saved ' "${LOG}/record_${ik}.log" || { echo "ERROR: the ${ik} bag was not saved in 60 s (${LOG}/record_${ik}.log)" >&2; exit 1; }
    bags+=("$(grep '^Saved ' "${LOG}/record_${ik}.log" | cut -d' ' -f2)")
done

REPORT="${DATA_DIR}/ik_comparisons/$(date +%Y%m%d_%H%M%S)_ik_comparison"
"$PY" -m baxter_teleop.diagnostics.ik_comparison "$REPORT" "${bags[@]}"
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
