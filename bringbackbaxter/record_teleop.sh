#!/bin/bash
# Record a teleop session, sim or real, for offline analysis: robot state and commands, teleop
# telemetry (/teleop/{arm}/ twist, IK, VR goal, keys, config), VR input (/vive, /libsurvive),
# TF and the robot description. Skips the real robot's cameras and IO. Ctrl+C stops recording.
# While a RethoughtAI policy runs (./run_inference), it records a rollout instead: also /policy/*
# and the policy's cameras as JPEG (compress_images.py), into rollouts/, checkpoint in session.yaml.
# Usage: ./record_teleop.sh [label]   ->  sessions/bags|rollouts/<time>_<sim|real>_<label>/ (+ session.yaml)
# Environment: ALL_TOPICS=1 records every topic (IO, cuffs, every camera as JPEG: ~1 MB/s each);
#              RAW_IMAGES=1 with it keeps the raw colour images instead (~1.5 GB/min per camera);
#              SESSION=<name> files the bag under sessions/<name>/ instead (see ./bag_all);
#              BAG_PATH=<dir> records to exactly that directory.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="${1:-session}"
set +u
source /opt/ros/kilted/setup.bash
source "${SCRIPT_DIR}/ros2_ws/install/setup.bash"
export RMW_IMPLEMENTATION=rmw_zenoh_cpp ZENOH_ROUTER_CHECK_ATTEMPTS=-1  # as run_teleop.sh
# ROS_DOMAIN_ID: kept as exported (unset = 0, the real robot); export another to run apart

NODES="$(ros2 node list 2>/dev/null)"
ROBOT=real
grep -qx /baxter_sim <<< "$NODES" && ROBOT=sim
KIND=bags
grep -qx -e /policy_runtime -e /policy_server <<< "$NODES" && KIND=rollouts
ROOT="${SCRIPT_DIR}/sessions/${KIND}"  # every recording lives under sessions/
[[ -n "$SESSION" ]] && ROOT="${SCRIPT_DIR}/sessions/${SESSION}"
BAG="${ROOT}/$(date +%Y%m%d_%H%M%S)_${ROBOT}_${LABEL}"
[[ -n "$BAG_PATH" ]] && BAG="$BAG_PATH"  # exact output path (e.g. the control benchmark runner)
mkdir -p "$(dirname "$BAG")"

TOPICS='^/(tf|tf_static|robot_description|rosout)$'
TOPICS+='|^/robot/(joint_states|state|head/.*|limb/.*|end_effector/[^/]+/(state|command|properties))$'
TOPICS+='|^/robot/(range|accelerometer)/.*'  # hand IR range (distance to what's in/below the gripper), hand IMU
TOPICS+='|^/teleop/.*|^/vive/.*|^/libsurvive/.*|^/sim/.*'
CAMERAS=(/cameras/left_hand_camera/image /cameras/third_eye/color/image_raw)  # the policy's
if [[ -n "$ALL_TOPICS" ]]; then  # every colour camera up (depth stays as it is)
    mapfile -t CAMERAS < <(ros2 topic list 2>/dev/null | grep -E '^/cameras/.*/(image|image_raw)$' | grep -vi depth || true)
fi
if [[ ( "$KIND" == rollouts || -n "$ALL_TOPICS" ) && -z "$RAW_IMAGES" ]] && (( ${#CAMERAS[@]} )); then
    # cameras as JPEG on <topic>/compressed (raw: ~3 GB a minute); the raw topics are not recorded
    python3 "${SCRIPT_DIR}/compress_images.py" "${CAMERAS[@]}" & COMPRESS=$!
    RAW_CAMERAS="^($(IFS='|'; echo "${CAMERAS[*]}"))\$"
fi
if [[ "$KIND" == rollouts ]]; then
    # the policy's chunks and commands, and its cameras
    TOPICS+='|^/policy/.*|^/cameras/(left_hand_camera/image|third_eye/color/image_raw)/compressed$'
    TOPICS+='|^/cameras/(left_hand_camera|third_eye/color)/camera_info$'
    # the running checkpoint, from RethoughtAI's `inference serve` / `inference run` command lines
    RAI_DIR="$(realpath "${SCRIPT_DIR}/../RethoughtAI")"
    arg() { pgrep -af "[i]nference $1 " | grep -v "uv run" | head -1 | sed -n "s/.* $2 \([^ ]*\).*/\1/p"; }
    MODEL="$(arg serve --model)" REVISION="$(arg serve --revision)" EPISODE="$(arg run --start-at)"
    [[ -n "$MODEL" && -d "${RAI_DIR}/${MODEL}" ]] && MODEL="${RAI_DIR}/${MODEL}"
    echo "Policy running: ${MODEL:-unknown}${EPISODE:+, shadowing ${EPISODE}}"
fi

if [[ -n "$ALL_TOPICS" ]]; then
    # --all subscribes to every topic, and the recorder stops at the first whose type has no
    # installed definition (e.g. baxter_core_msgs/HomingCommand from the ROS 1 bridge): skip those
    UNKNOWN="$(ros2 topic list -t 2>/dev/null | python3 -c '
import sys
from rosidl_runtime_py.utilities import get_message
for line in sys.stdin:
    topic, _, kind = line.strip().partition(" ")
    try:
        get_message(kind.strip("[]"))
    except Exception:
        print(topic)')"
    [[ -n "$UNKNOWN" ]] && echo "Skipping topics with no installed message type: ${UNKNOWN//$'\n'/ }"
fi

session_yaml() {
    echo "label: ${LABEL}"
    [[ -n "$SESSION" ]] && echo "session: ${SESSION}"
    echo "robot: ${ROBOT}"
    echo "kind: ${KIND%s}  # rollout: a policy was running when recording started"
    echo "topics: $([[ -n "$ALL_TOPICS" ]] && echo all || echo selected)"
    echo "cameras: $([[ -n "$RAW_CAMERAS" ]] && echo "jpeg on <topic>/compressed: ${CAMERAS[*]}" || echo raw)"
    [[ -n "$UNKNOWN" ]] && echo "skipped: [${UNKNOWN//$'\n'/, }]  # no installed message type"
    echo "recorded: $(date -Iseconds)  # start"
    if [[ "$KIND" == rollouts ]]; then
        echo "policy:"
        echo "  model: ${MODEL:-unknown}"
        [[ -n "$REVISION" ]] && echo "  revision: ${REVISION}"
        [[ -f "${MODEL}/config.json" ]] && echo "  type: $(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['type'])" "${MODEL}/config.json")"
        [[ -f "${MODEL}/training_step.pt" ]] && echo "  checkpoint: $(basename "$MODEL")"
        echo "  shadow_episode: ${EPISODE:-null}  # null: the robot's own cameras"
    fi
    echo "git:"
    for repo in "$SCRIPT_DIR/ros2_ws/src" "$SCRIPT_DIR/../BaxFlow-T" "$SCRIPT_DIR/../libsurvive_ros2" "$SCRIPT_DIR/../RethoughtAI/src"; do
        [[ -d "$repo" ]] && echo "  $(basename "$(realpath "$repo/..")")/$(basename "$repo"): $(git -C "$repo" describe --always --dirty 2>/dev/null)"
    done
    return 0
}
# written as soon as the recorder creates the bag, so any way of stopping it keeps the file
( until [[ -d "$BAG" ]]; do sleep 0.2; done; session_yaml > "${BAG}/session.yaml" ) &
echo "Recording ${ROBOT} robot to ${BAG}. Ctrl+C to stop."
trap ':' INT  # let Ctrl+C stop the recorder, then finish below (an ignored INT would be inherited)
SELECT=(--regex "$TOPICS")
# --all also takes the head screen and other image transports' duplicates; never the raw cameras being compressed
[[ -n "$ALL_TOPICS" ]] && SELECT=(--all ${UNKNOWN:+--exclude-topics $UNKNOWN}
    --exclude-regex "^/robot/xdisplay\$|/theora\$${RAW_CAMERAS:+|$RAW_CAMERAS}")
ros2 bag record -s mcap --storage-preset-profile zstd_fast "${SELECT[@]}" -o "$BAG" || true
trap - INT
[[ -n "${COMPRESS:-}" ]] && { kill -INT "$COMPRESS" 2>/dev/null || true; wait "$COMPRESS" 2>/dev/null || true; }

ros2 bag info "$BAG" | sed -n '/Duration/p;/Messages/p'
echo "Saved ${BAG}"
