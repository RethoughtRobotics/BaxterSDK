#!/bin/bash
# Plot the kinesthetic correction from a steer bag: active flag on top, TCP offset (x, y, z) below.
# Usage: scripts/plot_steer.sh <bag_path> [-o output.png]
#   bag_path  path to the bag directory (e.g. data/rollouts/20261111_194733_real_SteerTest)
set -euo pipefail

BAG="${1:?Usage: scripts/plot_steer.sh <bag_path> [-o output.png]}"
shift
RECAP_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../../recap")"

cd "$RECAP_DIR"
uv run recap plot "$BAG" \
  /kinesthetic/left/active.data \
  /kinesthetic/left/offset.twist.linear.x \
  /kinesthetic/left/offset.twist.linear.y \
  /kinesthetic/left/offset.twist.linear.z \
  "$@"
