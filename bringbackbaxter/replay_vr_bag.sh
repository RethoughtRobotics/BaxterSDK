#!/bin/bash
# Replay a VR teleop bag through the current (tool-frame) and proposed (base-frame) translation
# mappings from the untuck pose, print a per-stroke report, and loop both side by side in RViz.
# Usage: ./replay_vr_bag.sh <bag_dir> [--arm left|right] [--linear-speed 1.4] [--rate 0.5] ...
# Runs on its own ROS domain, so it never touches a live robot.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${SCRIPT_DIR}/ros2_ws"
BAG="$(realpath "${1:?usage: $0 <bag_dir> [vr_replay options]}")"
shift
source /opt/ros/kilted/setup.bash
source "${WS_DIR}/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ROS_DOMAIN_ID=42 ZENOH_ROUTER_CHECK_ATTEMPTS=-1  # wait for the router

pids=()
PARAMS="$(mktemp --suffix=.yaml)"
trap 'kill "${pids[@]}" 2>/dev/null; rm -f "$PARAMS"' EXIT
pgrep -f rmw_zenohd >/dev/null || { ros2 run rmw_zenoh_cpp rmw_zenohd & pids+=($!); }
# The same URDF the teleop's frax model loads (resolve_baxter_urdf).
URDF="$(ros2 pkg prefix baxter_description)/share/baxter_description/urdf/baxter.urdf"
python3 -c 'import sys, yaml; print(yaml.safe_dump({"/**": {"ros__parameters": {"robot_description": open(sys.argv[1]).read()}}}))' "$URDF" > "$PARAMS"
for run in current proposed; do
    ros2 run robot_state_publisher robot_state_publisher --ros-args -r __ns:=/${run} \
        --params-file "$PARAMS" -p frame_prefix:=${run}/ &
    pids+=($!)
done
rviz2 -d "${WS_DIR}/src/baxter_interface/config/vr_replay.rviz" &
pids+=($!)
"${WS_DIR}/.venv-ros/bin/python" -m baxter_interface.baxter_teleop.vr_replay "$BAG" "$@"
