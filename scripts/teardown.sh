#!/bin/bash
# Stop everything this stack started (sim, teleop, policy, recorders, camera, RViz, VR, routers).
# Usage: scripts/teardown.sh [--dry-run]
#   --dry-run  list what would be stopped

# Also removes stale ROS middleware shared memory from /dev/shm. The bridge container (the robot
# link) keeps running: stop it from its own terminal. The robot is not powered down: with the
# bridge and teleop gone it holds position.

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBSURVIVE_DIR="${LIBSURVIVE_DIR:-$(realpath "${SCRIPT_DIR}/../../libsurvive_ros2")}"
DRY=false
[[ "${1:-}" == --dry-run ]] && DRY=true

# Only what this stack starts (BaxterSDK, RethoughtAI, the camera, RViz, recorders, routers), never
# every ROS process on the machine: other ROS programs (a micro-ROS agent, Foxglove) are left alone.
PATTERN='baxter_sdk/(install|build|\.venv)|rethought_ai/(\.venv|launch)|policy\.launch\.py'
PATTERN+='|baxter_mujoco[.](baxter_ros|sim)|baxter_interface|baxter_teleop|baxter_tools|baxter_examples|move_group'
PATTERN+='|robot_state_publisher|joint_state_publisher|realsense2_camera|rs_launch\.py|rviz2'
PATTERN+='|ros2 bag (record|play)|rmw_zenohd|ros2cli\.daemon'
PATTERN+='|(run_sim_teleop|run_teleop|launch_sim|view_teleop_frames|record_teleop)\.sh'

# Never stop this script or the shells that started it.
keep=" $$ "
p=$PPID
while [[ "$p" -gt 1 ]]; do keep+="$p "; p=$(ps -o ppid= -p "$p" | tr -d ' '); done
targets() { for pid in $(pgrep -f -- "$PATTERN"); do [[ "$keep" == *" $pid "* ]] || echo "$pid"; done; }

mapfile -t pids < <(targets)
if $DRY; then
    echo "Would stop:"
    (( ${#pids[@]} )) && ps -o pid=,cmd= -p "${pids[*]}" | cut -c1-150 || echo "  no ROS processes"
else
    for sig in INT TERM KILL; do  # politely first, so nodes and the sim run their own shutdown
        (( ${#pids[@]} )) || break
        kill -"$sig" "${pids[@]}" 2>/dev/null
        for _ in 1 2 3 4 5; do sleep 1; mapfile -t pids < <(targets); (( ${#pids[@]} )) || break; done
    done
    (( ${#pids[@]} )) && echo "Could not stop: ${pids[*]}"
fi

if [[ -f "${LIBSURVIVE_DIR}/docker-compose.yml" ]] \
    && [[ -n "$(docker compose --project-directory "$LIBSURVIVE_DIR" ps -q 2>/dev/null)" ]]; then
    if $DRY; then echo "Would stop the libsurvive_ros2 VR container"
    else docker compose --project-directory "$LIBSURVIVE_DIR" stop >/dev/null 2>&1 && echo "Stopped the VR container"; fi
fi

shm=$(find /dev/shm -maxdepth 1 -user "$USER" \( -name 'fastrtps_*' -o -name 'sem.fastrtps_*' \
    -o -name 'fast_datasharing*' -o -name 'iceoryx*' -o -name '*zenoh*' -o -name 'cyclonedds*' \) 2>/dev/null)
if [[ -n "$shm" ]]; then
    if $DRY; then echo "Would remove from /dev/shm:"; echo "$shm"
    else echo "$shm" | xargs rm -f && echo "Removed $(echo "$shm" | wc -l) ROS shared-memory files"; fi
fi

$DRY || echo "Teardown done: no ROS processes, daemons or routers left (ros2 restarts its daemon on next use)."
