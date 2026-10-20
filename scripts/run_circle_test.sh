#!/bin/bash
# Right-arm control test: untuck, record, drive 15 cm circles through the teleop, stop recording.
# Usage: scripts/run_circle_test.sh [--output velocity|raw|raw_id] [scripted_trajectory options]
#   --output  joint output stage (default: raw_id)
#   e.g. --circle-speed 0.25 --loops 2   (all: python -m baxter_teleop.diagnostics.scripted_trajectory -h)

# Default: servo gain 3/s P-only, 3 circles of 15 cm radius at 0.15 m/s in x-y toward the arm's
# outer side, ~21 s. Bag: data/bags/<time>_<real|sim>_circle_<output>/. Runs on ROS_DOMAIN_ID as
# exported (unset = 0, the robot). Ctrl+C holds the arm and still saves the bag. raw/raw_id bypass
# Baxter's collision avoidance: keep the circle clear.

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
set -e
OUTPUT=raw_id
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output) OUTPUT="$2"; shift ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if ! timeout 5 ros2 topic echo --once /robot/joint_states >/dev/null 2>&1; then
    echo "No /robot/joint_states on ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}: start the bridge (or the sim) first." >&2
    exit 1
fi
if pgrep -f "[-]m baxter_teleop( .*)? --arm right|[s]cripted_trajectory" >/dev/null; then
    echo "A teleop is already driving the right arm: close it first (two controllers would fight)." >&2
    exit 1
fi

echo "== Output ${OUTPUT}. The right arm will circle 15 cm out to its right of the home pose: keep it clear."
echo "== Moving to the home (untuck) pose"
ros2 run baxter_tools tuck_arms -u 2>&1 | grep -v '^\[INFO\]' || true

LOG="$(mktemp -d)"
setsid "${SCRIPT_DIR}/record_teleop.sh" "circle_${OUTPUT}" >"${LOG}/record.log" 2>&1 &
RECORDER=$!  # session leader: INT to its group reaches record_teleop.sh and ros2 bag record
stop_recording() {
    kill -INT -- "-${RECORDER}" 2>/dev/null || return 0
    for _ in $(seq 30); do grep -q '^Saved ' "${LOG}/record.log" && break; sleep 0.5; done
    grep -E '^(Duration|Messages|Saved)' "${LOG}/record.log" || echo "Recorder log: ${LOG}/record.log"
}
trap stop_recording EXIT
for _ in $(seq 150); do grep -q '^Recording' "${LOG}/record.log" && break; sleep 0.2; done
grep -q '^Recording' "${LOG}/record.log" || { echo "ERROR: the recorder did not start in 30 s (${LOG}/record.log)" >&2; exit 1; }
sleep 3  # let the recorder subscribe to every topic before the motion starts

echo "== Circles"
"$PY" -m baxter_teleop.diagnostics.scripted_trajectory \
    --arm right --protocol circle --output "$OUTPUT" --gain 3 --no-feedforward "${ARGS[@]}" \
    2> >(grep -v '^\[INFO\]' >&2)
sleep 1  # the hold after the last circle
echo "== Done; saving the bag"
