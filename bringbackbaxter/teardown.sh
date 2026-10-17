#!/bin/bash
# Stop everything ROS on this machine for the Baxter setup, real or sim, and reset to a clean state:
#   sim robot (BaxFlow-T), teleop, tuck_arms, RViz, robot/joint state publishers, VR replay,
#   ros1_bridge (real robot link), the libsurvive VR container, Zenoh routers and ROS 2 daemons;
#   then removes stale ROS middleware shared memory from /dev/shm.
# The robot itself is not powered down: with the bridge and teleop gone it holds position.
# Usage: ./teardown.sh [--dry-run]   (--dry-run lists what would be stopped)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBSURVIVE_DIR="${LIBSURVIVE_DIR:-$(realpath "${SCRIPT_DIR}/../libsurvive_ros2")}"
DRY=false
[[ "${1:-}" == --dry-run ]] && DRY=true

PATTERN='/opt/ros/|ros2_ws/(install|build|\.venv)|ros2cli\.daemon|rmw_zenohd|rviz2|robot_state_publisher'
PATTERN+='|joint_state_publisher|ros1_bridge|dynamic_bridge|baxflow[.](baxter_ros|sim)|baxter_interface'
PATTERN+='|baxter_tools|(run_sim_teleop|run_teleop|launch_sim|view_teleop_frames|replay_vr_bag|run_bridge)\.sh'

# Never stop this script or the shells that started it.
keep=" $$ "
p=$PPID
while [[ "$p" -gt 1 ]]; do keep+="$p "; p=$(ps -o ppid= -p "$p" | tr -d ' '); done
targets() { for pid in $(pgrep -f -- "$PATTERN"); do [[ "$keep" == *" $pid "* ]] || echo "$pid"; done; }

pids=$(targets)
if $DRY; then
    echo "Would stop:"
    [[ -n "$pids" ]] && ps -o pid=,cmd= -p $pids | cut -c1-150 || echo "  no ROS processes"
else
    for sig in INT TERM KILL; do  # politely first, so nodes and the sim run their own shutdown
        [[ -z "$pids" ]] && break
        kill -"$sig" $pids 2>/dev/null
        for _ in 1 2 3 4 5; do sleep 1; pids=$(targets); [[ -z "$pids" ]] && break; done
    done
    [[ -n "$pids" ]] && echo "Could not stop: $pids"
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
