#!/bin/bash
# Report how well the arm followed the operator in recorded teleop bags (and sim contacts).
# Usage: scripts/analyze_teleop_bag.sh BAG [BAG ...]
#   BAG  a folder from record_teleop.sh; several are compared side by side

# Per-segment and summary metrics (diagnostics/bag_report.py); for sim bags, the contacts the
# arms made (baxter_mujoco.contacts).

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
set -u

"$PY" -m baxter_teleop.diagnostics.bag_report "$@"
for bag in "$@"; do
    if [[ "$bag" == *_sim_* && -d "$SIM_DIR" ]]; then
        echo -e "\n=== $(basename "$bag")"
        npz="$(realpath "$bag")/joints.npz"
        (cd "$SIM_DIR" && uv run python -m baxter_mujoco.contacts "$npz" 2> >(grep -v robosuite >&2))
    fi
done
