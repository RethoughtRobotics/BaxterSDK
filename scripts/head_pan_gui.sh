#!/usr/bin/env bash
# Head pan slider GUI: run from any terminal with the robot bridge up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"
exec "$PY" "$SCRIPT_DIR/../src/baxter_examples/baxter_examples/head_pan_gui.py" "$@"
