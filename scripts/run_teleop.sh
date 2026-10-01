#!/bin/bash
# Launch Baxter cartesian-delta teleop in its own window that reads key input.
# Usage: scripts/run_teleop.sh [--arm right|left] [--linear-speed 0.2] ...
# Opens a Terminator window if a display is available, otherwise a tmux window.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARGS="${*:---arm right}"

# .venv-ros is Python 3.12 (matches ROS Kilted rclpy) with frax + jax installed
CMD="cd '${WS_DIR}' \
  && source /opt/ros/kilted/setup.bash \
  && source '${WS_DIR}/install/setup.bash' \
  && export RMW_IMPLEMENTATION=rmw_zenoh_cpp && unset ROS_DOMAIN_ID \
  && '${WS_DIR}/.venv-ros/bin/python' -m baxter_interface.baxter_teleop.cartesian_delta_teleop ${ARGS}; \
  echo; read -rp 'Teleop exited. Press Enter to close.'"

if [[ -n "$DISPLAY" ]] && command -v terminator >/dev/null; then
    terminator --title "Baxter teleop" --working-directory "${WS_DIR}" -x bash -c "$CMD" &
    echo "Teleop opened in a new Terminator window. Click it to give it keyboard focus."
elif [[ -n "$TMUX" ]]; then
    tmux new-window -n baxter-teleop "bash -c \"$CMD\""
else
    tmux new-session -s baxter-teleop "bash -c \"$CMD\""
fi
