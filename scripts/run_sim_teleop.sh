#!/bin/bash
# Teleop the simulated Baxter (baxter_mujoco on the real robot's ROS 2 interface), no hardware:
# starts the sim and live RViz, untucks the arms with tuck_arms -u, then opens run_teleop.sh.
# Usage: ./run_sim_teleop.sh [--vr] [--table|--objects] [--left|--right] [other run_teleop.sh options]
#   --vr        control with live VR (requires libsurvive_ros2 to be started separately)
#   --table     keep the table (it stops the hands about 12 cm below the base frame)
#   --objects   keep the table and the scene's objects (the pot); by default neither is there
# RViz shows the sim's table and objects (/sim/scene).
# Runs on $ROS_DOMAIN_ID if exported (e.g. export ROS_DOMAIN_ID=55 to keep the sim apart from a
# real robot on the default domain). Ctrl+C here stops everything this script started. Refuses to
# run while a real robot is up on that domain; a sim left over from an earlier run (e.g. after
# kill -9) is stopped and replaced.

set -e

INPUT=keyboard
ARM=right
SIM_ARGS=(--empty --no-table)
ARGS=()
for a in "$@"; do
    case "$a" in
        --vr) INPUT=vr ;;
        --table) SIM_ARGS=(--empty) ;;
        --objects) SIM_ARGS=() ;;
        --left | --right) ARM="${a#--}"; ARGS+=("$a") ;;
        *) ARGS+=("$a") ;;
    esac
done

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# ROS_DOMAIN_ID: kept as exported (unset = 0, the real robot); export another to run apart

groups=()
cleanup() {
    for g in "${groups[@]}"; do kill -INT -- "-$g" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP  # exit through cleanup on Ctrl+C, kill or a closed terminal
start() {  # run in its own process group, so cleanup stops all of its children
    setsid "$@" &
    groups+=($!)
}
wait_for_topic() {  # topic, seconds (each try allows 5 s: the ros2 CLI alone takes ~2 s to start)
    local deadline=$((SECONDS + $2))
    while ((SECONDS < deadline)); do
        timeout 5 ros2 topic echo --once "$1" >/dev/null 2>&1 && return 0
        sleep 0.5
    done
    return 1
}

pgrep -f rmw_zenohd >/dev/null || start ros2 run rmw_zenoh_cpp rmw_zenohd
if wait_for_topic /robot/joint_states 5; then
    if ! ros2 node list 2>/dev/null | grep -qx /baxter_sim; then
        echo 'A robot is already publishing /robot/joint_states; not starting the sim.'
        exit 1
    fi
    echo 'Stopping a simulated Baxter left over from an earlier run...'
    pkill -INT -f 'baxter_mujoco[.]baxter_ros' || true
    sleep 3
fi

echo "Starting the simulated Baxter..."
start "${SIM_DIR}/launch_sim.sh" --ros "${SIM_ARGS[@]}"
wait_for_topic /robot/joint_states 60 || { echo 'The sim did not come up (see above).'; exit 1; }
start "${SCRIPT_DIR}/view_teleop_frames.sh" --live
ros2 run baxter_tools tuck_arms -u

if [[ "$INPUT" == vr ]]; then
    wait_for_topic "/vive/${ARM}/target" 5 \
        || { echo "No /vive/${ARM}/target on ROS domain ${ROS_DOMAIN_ID:-0}. Start libsurvive_ros2 from a shell" \
                  "with the same ROS_DOMAIN_ID (running driver: domain $(docker inspect --format \
                  '{{range .Config.Env}}{{println .}}{{end}}' libsurvive_ros2-libsurvive_ros2-1 2>/dev/null \
                  | sed -n 's/^ROS_DOMAIN_ID=//p' | grep . || echo 0)), then press A."; exit 1; }
fi

"${SCRIPT_DIR}/run_teleop.sh" --input "$INPUT" "${ARGS[@]}"
echo "Sim and RViz are running. Ctrl+C here to stop them."
wait "${groups[@]}"
