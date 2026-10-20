#!/bin/bash
# End-to-end regression test of the teleop on the MuJoCo sim (circle, VR, keyboard), against
# src/baxter_teleop/test/sim_baseline.yaml. The MuJoCo viewer opens when a display is available.
# Usage: ./run_sim_regression.sh [--only circle,vr,keys] [--headless] [--write-baseline] [--domain 43]
# Exit status 1 if any metric is out of tolerance. Runs on ROS_DOMAIN_ID 43 unless --domain says otherwise.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec "$PY" -m baxter_teleop.diagnostics.sim_regression "$@"
