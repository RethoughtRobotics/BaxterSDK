#!/bin/bash
# Replay a recorded demo on the arms (joint space) by pose, by force, or both, and compare.
# Usage: scripts/replay_demo.sh EPISODE [--mode pose|force|both] [options]
#   EPISODE         bag folder, e.g. ../rethought_ai/recorder/data/SweepTrash_Kin/episode_000
#   --mode          pose (stiff joint servo), force (torque mode), both (default)
#   --out DIR       write the comparison report (default: data/replays/<time>_<episode>/)
#   --arms, --omega, --speed, --lead, --until, --no-gripper   (python -m baxter_teleop.replay -h)

# Runs on ROS_DOMAIN_ID as exported (unset = 0, the robot; the sim on its domain). Raw position and
# torque modes bypass Baxter's collision avoidance: keep the workspace clear.

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[[ $# -ge 1 ]] || { sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 1; }
OUT=()
[[ " $* " == *" --out "* ]] || OUT=(--out "${DATA_DIR}/replays/$(date +%Y%m%d_%H%M%S)_$(basename "$1")")
exec "$PY" -m baxter_teleop.replay "$@" "${OUT[@]}"
