#!/bin/bash
# Test the teleop end to end on the sim (circle, VR, keyboard) against a stored baseline.
# Usage: scripts/run_sim_regression.sh [--only circle,vr,keys] [--headless] [--write-baseline]
#   --only            scenarios to run (default: all)
#   --headless        no MuJoCo viewer
#   --write-baseline  store this run as the baseline (src/baxter_teleop/test/sim_baseline.yaml)
#   --domain N        ROS_DOMAIN_ID (default: 43)

# Exit status 1 if any metric is out of tolerance. Details: python -m
# baxter_teleop.diagnostics.sim_regression -h.

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec "$PY" -m baxter_teleop.diagnostics.sim_regression "$@"
