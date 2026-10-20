#!/bin/bash
# Enable Baxter's motors (the robot must be up: baxter_start).
# Usage: scripts/enable_robot.sh [-d|-r|-s|-S]
#   (none)  enable
#   -d      disable
#   -r      reset (after an e-stop or error)
#   -s      print the robot's state
#   -S      stop

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
exec ros2 run baxter_tools enable_robot "${@:--e}"
