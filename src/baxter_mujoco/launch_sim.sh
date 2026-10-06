#!/usr/bin/env bash
# Open the MuJoCo viewer with Baxter (both arms).
#   ./launch_sim.sh          holding its home pose
#   ./launch_sim.sh --ros    as a robot on the real Baxter ROS 2 interface (baxflow/baxter_ros.py), so
#                            bringbackbaxter's teleop, tuck_arms and RViz drive it like the real one.
#                            Options: --no-viewer, --empty (pot moved out of reach, for controller
#                            tests). Needs ROS 2 Kilted and bringbackbaxter's workspace (BAXTER_WS).
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" != --ros ]]; then
    exec uv run python -m baxflow.sim "$@"
fi
shift
BAXTER_WS="${BAXTER_WS:-$(realpath ../bringbackbaxter/ros2_ws)}"
set +u  # ROS setup scripts read unset variables
source /opt/ros/kilted/setup.bash
source "${BAXTER_WS}/install/setup.bash"
set -u
# Same middleware as bringbackbaxter's run_teleop.sh, on the default domain the teleop uses.
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ZENOH_ROUTER_CHECK_ATTEMPTS=-1
unset ROS_DOMAIN_ID
pgrep -f rmw_zenohd >/dev/null || { ros2 run rmw_zenoh_cpp rmw_zenohd >/dev/null 2>&1 & trap 'kill $!' EXIT; }
uv run --group ros python -m baxflow.baxter_ros "$@"
