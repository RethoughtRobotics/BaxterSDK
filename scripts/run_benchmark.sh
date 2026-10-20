#!/bin/bash
# Control benchmark on the robot: one-factor sweeps of the teleop's control parameters.
# Usage: scripts/run_benchmark.sh [DIR] [benchmark options]
#   DIR        results folder (default: data/benchmarks/control_benchmark_<date>; reuse to resume)
#   --dry-run  plan and duration only
#   --sweeps   e.g. gain; --repeats N   (all: python -m baxter_teleop.diagnostics.benchmark -h)

# Each run plays the ISO 9283-style protocol (pose, linear, circle), recorded to its own bag, then
# the report. Runs on ROS_DOMAIN_ID as exported (unset = 0, the robot). Ctrl+C stops after holding
# the arm; rerunning resumes. Raw outputs bypass collision avoidance: keep the 30 cm to the arm's
# right clear. Each run aborts itself (and holds) on tracking error or contact force.

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
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
    if pgrep -f "[-]m baxter_teleop( .*)? --arm right|[s]cripted_trajectory" >/dev/null; then
        echo "A teleop is already driving the right arm: close it first." >&2
        exit 1
    fi
fi
"$PY" -m baxter_teleop.diagnostics.benchmark "$ROOT" "$@"
[[ " $* " == *" --dry-run "* ]] && exit 0
echo "== Report"
"$PY" -m baxter_teleop.benchmark_report "$ROOT" \
    || echo "Report failed; rerun: $PY -m baxter_teleop.benchmark_report $ROOT"
echo "Results: ${ROOT}/README.md"
