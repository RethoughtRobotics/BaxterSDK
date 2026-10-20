#!/bin/bash
# Record a teleop session (sim or real) to a bag for offline analysis. Ctrl+C stops.
# Usage: scripts/record_teleop.sh [LABEL]
#   LABEL  name in data/bags/<time>_<sim|real>_<LABEL>/ (default: session)
# Environment:
#   ALL_TOPICS=1     every topic (IO, cuffs, every camera as JPEG)
#   RAW_IMAGES=1     with ALL_TOPICS, raw colour images (~1.5 GB/min per camera)
#   SESSION=NAME     file the bag under data/sessions/NAME/
#   BAG_PATH=DIR     record to exactly DIR

# Records robot state and commands, teleop telemetry (/teleop/{arm}/*), VR input (/vive,
# /libsurvive), the cuffs (squeeze and gripper buttons), TF and the robot description; not the
# real robot's cameras or its other IO. While a RethoughtAI policy runs it records a rollout
# instead (data/rollouts/): also /policy/* and the policy's cameras as JPEG (compress_images.py),
# with the checkpoint in session.yaml.

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
set -e
LABEL="${1:-session}"
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# ROS_DOMAIN_ID: kept as exported (unset = 0, the real robot); export another to run apart

NODES="$(ros2 node list 2>/dev/null)"
ROBOT=real
grep -qx /baxter_sim <<< "$NODES" && ROBOT=sim
KIND=bags
grep -qx -e /policy_runtime -e /policy_server <<< "$NODES" && KIND=rollouts
ROOT="${DATA_DIR}/${KIND}"
[[ -n "$SESSION" ]] && ROOT="${DATA_DIR}/sessions/${SESSION}"
BAG="${ROOT}/$(date +%Y%m%d_%H%M%S)_${ROBOT}_${LABEL}"
[[ -n "$BAG_PATH" ]] && BAG="$BAG_PATH"  # exact output path (e.g. the control benchmark runner)
mkdir -p "$(dirname "$BAG")"

TOPICS='^/(tf|tf_static|robot_description|rosout)$'
TOPICS+='|^/robot/(joint_states|state|head/.*|limb/.*|end_effector/[^/]+/(state|command|properties))$'
TOPICS+='|^/robot/(range|accelerometer)/.*'  # hand IR range (distance to what's in/below the gripper), hand IMU
TOPICS+='|^/teleop/.*|^/vive/.*|^/libsurvive/.*|^/sim/.*'
TOPICS+='|^/robot/digital_io/(left|right)_(lower_cuff|upper_button|lower_button)/state$'  # cuff squeeze (zero-g takeover), gripper buttons
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
    echo "provenance: provenance.yaml  # commits, patches, process command lines, teleop.yaml, containers"
    return 0
}
# written as soon as the recorder creates the bag, so any way of stopping it keeps the file
# Gives up after 60 s, so a recorder that dies before creating the bag cannot leave this loop running.
( for _ in $(seq 300); do
    [[ -d "$BAG" ]] && { session_yaml > "${BAG}/session.yaml"; python3 -m baxter_tools.provenance write "$BAG"; exit; }
    sleep 0.2
done ) &
echo "Recording ${ROBOT} robot to ${BAG}. Ctrl+C to stop."
trap ':' INT  # let Ctrl+C stop the recorder, then finish below (an ignored INT would be inherited)
SELECT=(--regex "$TOPICS")
read -ra CUSTOM <<< "$(python3 -m baxter_tools.provenance custom-data)"  # short form, in metadata.yaml
# --all also takes the head screen and other image transports' duplicates; never the raw cameras being compressed
[[ -n "$ALL_TOPICS" ]] && SELECT=(--all ${UNKNOWN:+--exclude-topics $UNKNOWN}
    --exclude-regex "^/robot/xdisplay\$|/theora\$${RAW_CAMERAS:+|$RAW_CAMERAS}")
ros2 bag record -s mcap --storage-preset-profile zstd_fast "${SELECT[@]}" ${CUSTOM[0]:+--custom-data "${CUSTOM[@]}"} -o "$BAG" || true
trap - INT
[[ -n "${COMPRESS:-}" ]] && { kill -INT "$COMPRESS" 2>/dev/null || true; wait "$COMPRESS" 2>/dev/null || true; }

ros2 bag info "$BAG" | sed -n '/Duration/p;/Messages/p'
echo "Saved ${BAG}"
