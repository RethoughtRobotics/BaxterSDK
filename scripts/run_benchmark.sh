#!/bin/bash
# Control benchmark on the robot: one-factor sweeps of the teleop's control parameters, each run the
# ISO 9283-style protocol (pose, linear, circle) recorded to its own bag, then the report.
#
# Usage: ./run_benchmark.sh [experiment_dir] [benchmark options]
#   default dir: experiments/control_benchmark_<date>   (rerun with the same dir to resume)
#   e.g. ./run_benchmark.sh --dry-run                      plan and duration only
#        ./run_benchmark.sh experiments/cb --sweeps gain    one sweep
#        ./run_benchmark.sh experiments/cb --repeats 1      quick pass
# Runs on ROS_DOMAIN_ID as exported (unset = 0, the real robot). Ctrl+C stops after holding the arm;
# rerunning resumes. Raw outputs bypass collision avoidance: keep the 30 cm area to the arm's right clear.
# Each run aborts itself (and holds) on tracking error or contact force (trajectory_test --safety).

set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
ROOT="${DATA_DIR}/benchmarks/control_benchmark_$(date +%Y%m%d)"
if [[ $# -gt 0 && "$1" != -* ]]; then ROOT="$(realpath -m "$1")"; shift; fi
export BAXTER_SCRIPTS="$SCRIPT_DIR"

if [[ " $* " != *" --dry-run "* ]]; then
    if ! timeout 5 ros2 topic echo --once /robot/joint_states >/dev/null 2>&1; then
        echo "No /robot/joint_states on ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}: start the bridge (or the sim) first." >&2
        exit 1
    fi
    if pgrep -f "[c]artesian_delta_teleop.*--arm right|[t]rajectory_test" >/dev/null; then
        echo "A teleop is already driving the right arm: close it first." >&2
        exit 1
    fi
fi
"$PY" -m baxter_interface.baxter_teleop.benchmark "$ROOT" "$@" 2> >(grep -v -e jax -e GPU >&2)
[[ " $* " == *" --dry-run "* ]] && exit 0
echo "== Report"
"$PY" -m baxter_interface.baxter_teleop.benchmark_report "$ROOT" 2> >(grep -v -e jax -e GPU >&2) \
    || echo "Report failed; rerun: $PY -m baxter_interface.baxter_teleop.benchmark_report $ROOT"
echo "Results: ${ROOT}/README.md"
