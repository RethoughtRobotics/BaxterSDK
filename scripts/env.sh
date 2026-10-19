# Sourced, never run, by the scripts here and the panes run_teleop.sh opens: ROS 2 Kilted, this
# workspace, rmw_zenoh, and the paths the scripts share. Each *_DIR can be overridden from the
# environment. ROS_DOMAIN_ID is left as exported (unset = 0, the real robot).
# shellcheck shell=bash disable=SC2034  # its variables are for the scripts that source it

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(realpath "${SCRIPT_DIR}/..")"
SIM_DIR="${SIM_DIR:-${WS_DIR}/src/baxter_mujoco}"
RAI_DIR="${RAI_DIR:-$(realpath -m "${WS_DIR}/../rethought_ai")}"
DATA_DIR="${DATA_DIR:-$(realpath -m "${WS_DIR}/../data")}"  # FinalProject/data: every recording and result
PY="${WS_DIR}/.venv-ros/bin/python"                          # Python 3.12 (Kilted's rclpy) with frax and jax

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
