# Sourced, never run, by the scripts here and the panes run_teleop.sh opens: ROS 2 Kilted, this
# workspace, rmw_zenoh, and the paths the scripts share. Each *_DIR can be overridden from the
# environment. ROS_DOMAIN_ID is left as exported (unset = 0, the real robot).
# shellcheck shell=bash disable=SC2034  # its variables are for the scripts that source it

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(realpath "${SCRIPT_DIR}/..")"
SIM_DIR="${SIM_DIR:-${WS_DIR}/src/baxter_mujoco}"
RAI_DIR="${RAI_DIR:-$(realpath -m "${WS_DIR}/../rethought_ai")}"
DATA_DIR="${DATA_DIR:-$(realpath -m "${WS_DIR}/../data")}"  # FinalProject/data: every recording and result
PY="${PY:-${WS_DIR}/.venv-ros/bin/python}"  # ROS Kilted's Python 3.12 plus ruckig (src/baxter_teleop/requirements.txt)

# This stack's processes, by command line: teardown.sh stops them, provenance.py records them. Only
# what this stack starts, never every ROS process on the machine (a micro-ROS agent, Foxglove).
STACK_PROCESSES='baxter_sdk/(install|build|\.venv)|rethought_ai/(\.venv|launch)|policy\.launch\.py'
STACK_PROCESSES+='|baxter_mujoco[.](baxter_ros|sim)|baxter_interface|baxter_teleop|baxter_tools|baxter_examples|move_group'
STACK_PROCESSES+='|robot_state_publisher|joint_state_publisher|realsense2_camera|rs_launch\.py|rviz2'
STACK_PROCESSES+='|ros2 bag (record|play)|rmw_zenohd|ros2cli\.daemon'
STACK_PROCESSES+='|(run_sim_teleop|run_teleop|launch_sim|view_teleop_frames|record_teleop)\.sh'
export STACK_PROCESSES

# The ROS setup files read unset variables: source them with nounset off, then restore the caller's.
[[ $- == *u* ]] && _env_nounset=1
set +u
source /opt/ros/kilted/setup.bash
source "${WS_DIR}/install/setup.bash"
[[ -n "${_env_nounset:-}" ]] && set -u
unset _env_nounset

# -1 skips rmw_zenoh's startup check for a router (unset tries once and warns, 0 waits forever).
# Either way a node joins the graph once a router is up.
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ZENOH_ROUTER_CHECK_ATTEMPTS=-1
