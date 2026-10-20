#!/bin/bash
# Teleop the simulated Baxter: starts the sim and RViz, untucks, then opens run_teleop.sh.
# Usage: scripts/run_sim_teleop.sh [--vr] [--table|--objects|--cup] [--cameras] [--headless] [run_teleop.sh options]
#   --vr       live VR (start libsurvive_ros2 first); default keyboard
#   --table    keep the table
#   --objects  keep the table and its objects
#   --cup      the BlueCupRedContainer scene: real table height, cup, container
#   --cameras  the robot's cameras (and the lab's third_eye with --cup), as for recording
#   --headless no MuJoCo viewer
#   run_teleop.sh options, e.g. --left, --policy (scripts/run_teleop.sh -h)

# The sim is baxter_mujoco on the real robot's ROS 2 interface. RViz shows the sim's table and
# objects (/sim/scene). Runs on $ROS_DOMAIN_ID if exported (e.g. 55 to keep apart from a robot on
# 0). Ctrl+C here stops everything this script started. Refuses to run while a real robot is up on
# that domain; a sim left from an earlier run is stopped and replaced.

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
set -e

INPUT=keyboard
ARM=right
SIM_ARGS=(--empty --no-table)
SIM_OPTS=()
ARGS=()
for a in "$@"; do
    case "$a" in
        --vr) INPUT=vr ;;
        --policy) INPUT=policy; ARGS+=("$a") ;;  # run_teleop.sh points the teleop at the policy
        --table) SIM_ARGS=(--empty) ;;
        --objects) SIM_ARGS=() ;;
        --cup) SIM_ARGS=(--scene cup) ;;
        --headless) SIM_OPTS+=(--no-viewer) ;;
        --cameras) SIM_OPTS+=(--cameras) ;;
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
start "${SIM_DIR}/launch_sim.sh" --ros "${SIM_ARGS[@]}" "${SIM_OPTS[@]}"
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

if [[ "$INPUT" == policy ]]; then
    "${SCRIPT_DIR}/run_teleop.sh" "${ARGS[@]}"
else
    "${SCRIPT_DIR}/run_teleop.sh" --input "$INPUT" "${ARGS[@]}"
fi
echo "Sim and RViz are running. Ctrl+C here to stop them."
wait "${groups[@]}"
