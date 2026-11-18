#!/bin/bash
# View a recording vs its replay: cameras, TCP position and orientation.
# Usage: scripts/replay_viewer.sh EPISODE [--port 8051]
#   EPISODE   episode directory with a replay/ subdirectory

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECAP_VENV="$(cd "${SCRIPT_DIR}/../../recap/.venv" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../src/baxter_teleop:${PYTHONPATH:-}"
exec "${RECAP_VENV}/bin/python" -m baxter_teleop.diagnostics.replay_viewer "$@"
