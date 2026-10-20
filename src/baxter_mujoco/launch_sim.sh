#!/usr/bin/env bash
# Open the MuJoCo viewer with Baxter (both arms).
#   ./launch_sim.sh          holding its home pose
#   ./launch_sim.sh --ros    as a robot on the real Baxter ROS 2 interface (baxter_mujoco/baxter_ros.py), so
#                            the teleop, tuck_arms and RViz drive it like the real one.
#                            Options: --no-viewer, --empty (pot moved out of reach, for controller
#                            tests), --scene cup, --cameras (the robot's cameras, and the lab's in
#                            the cup scene: baxter_mujoco/cameras.py, its own process). Needs ROS 2
#                            Kilted and this workspace built (BAXTER_WS).
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-}" != --ros ]]; then
    exec uv run python -m baxter_mujoco.sim "$@"
fi
shift
BAXTER_WS="${BAXTER_WS:-$(realpath ../..)}"  # this BaxterSDK workspace
set +u  # ROS setup scripts read unset variables
source /opt/ros/kilted/setup.bash
source "${BAXTER_WS}/install/setup.bash"
set -u
# Same middleware as the teleop; ROS_DOMAIN_ID kept as exported (unset = 0).
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ZENOH_ROUTER_CHECK_ATTEMPTS=-1
pids=()
trap 'kill "${pids[@]}" 2>/dev/null' EXIT
pgrep -f rmw_zenohd >/dev/null || { ros2 run rmw_zenoh_cpp rmw_zenohd >/dev/null 2>&1 & pids+=($!); }
args=()
for a in "$@"; do
    if [[ "$a" == --cameras ]]; then CAMERAS=1; else args+=("$a"); fi
done
if [[ -n "${CAMERAS:-}" ]]; then
    # Offscreen through EGL: GLX offscreen contexts beside a viewer segfault Xorg on the NVIDIA 535 driver
    MUJOCO_GL=egl uv run --group ros python -m baxter_mujoco.cameras "${args[@]}" &
    pids+=($!)
fi
uv run --group ros python -m baxter_mujoco.baxter_ros "${args[@]}"
