#!/bin/bash
# RViz of Baxter with the TCP frames ({arm}_gripper) and the base frame.
# Usage: scripts/view_teleop_frames.sh [--live|--vive]
#   (none)   fixed untuck pose, on its own ROS domain (never touches a robot)
#   --live   the robot's or sim's joint states, plus the VR tracking tree
#   --vive   the VR tracking tree only

# libsurvive_ros2 keeps its TF on /libsurvive/tf* (off /tf, which the bridge forwards to the robot),
# with libsurvive_world parented to base. Live, the robot tree is published there too and RViz
# reads only those topics, so both trees show as one and nothing here adds to the robot's /tf.

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
CALLER_ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-}"
export ROS_DOMAIN_ID=42
case "${1:-}" in
    ''|--live|--vive) ;;
    *) echo "Unknown option '$1'. Usage: $0 [--live|--vive]" >&2; exit 2 ;;
esac
LIVE=false
VIVE_ONLY=false
[[ "${1:-}" == --vive ]] && VIVE_ONLY=true
if [[ "${1:-}" == --live || "${1:-}" == --vive ]]; then  # the caller's domain, as run_teleop.sh (unset = 0)
    LIVE=true
    if [[ -n "${CALLER_ROS_DOMAIN_ID}" ]]; then export ROS_DOMAIN_ID="${CALLER_ROS_DOMAIN_ID}"; else unset ROS_DOMAIN_ID; fi
fi

# Exact home pose: baxter_interface.settings.UNTUCK_POSITIONS (tuck_arms -u, the teleop H key),
# joint order s0 s1 e0 e1 w0 w1 w2
SETTINGS="${WS_DIR}/src/baxter_interface/baxter_interface/settings.py"
read -ra ZEROS < <(python3 - "$SETTINGS" <<'PY'
import ast, sys
untuck = next(ast.literal_eval(n.value) for n in ast.parse(open(sys.argv[1]).read()).body
              if isinstance(n, ast.Assign) and n.targets[0].id == 'UNTUCK_POSITIONS')
print(' '.join(f'-p zeros.{arm}_{j}:={q!r}' for arm, qs in untuck.items()
               for j, q in zip(['s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2'], qs)))
PY
)

pids=()
trap 'kill "${pids[@]}" 2>/dev/null; rm -f "$PARAMS" "${LIVE_RVIZ:-}"' EXIT
pgrep -f rmw_zenohd >/dev/null || { ros2 run rmw_zenoh_cpp rmw_zenohd & pids+=($!); }
# The same URDF the teleop's kinematics load (kinematics.resolve_baxter_urdf); the description launch's
# xacro differs (left_gripper ~12 cm apart), so it would show different TCP frames.
URDF="$(ros2 pkg prefix baxter_description)/share/baxter_description/urdf/baxter.urdf"
PARAMS="$(mktemp --suffix=.yaml)"
python3 -c 'import sys, yaml; print(yaml.safe_dump({"/**": {"ros__parameters": {"robot_description": open(sys.argv[1]).read()}}}))' "$URDF" > "$PARAMS"
TF_REMAP=()
$LIVE && TF_REMAP=(-r /tf:=/libsurvive/tf -r /tf_static:=/libsurvive/tf_static)
if ! $VIVE_ONLY; then
    ros2 run robot_state_publisher robot_state_publisher --ros-args --params-file "$PARAMS" \
        -r joint_states:=/robot/joint_states "${TF_REMAP[@]}" &
    pids+=($!)
fi
if ! $LIVE; then
    ros2 run joint_state_publisher joint_state_publisher --ros-args \
        -r joint_states:=/robot/joint_states "${ZEROS[@]}" &
    pids+=($!)
fi
RVIZ="${WS_DIR}/src/baxter_teleop/config/teleop_frames.rviz"
if $LIVE; then
    # Live view: right half of the desktop work area (teleop's Terminator takes the left), no frame names.
    LIVE_RVIZ="$(mktemp --suffix=.rviz)"
    python3 - "$RVIZ" "$LIVE_RVIZ" "$(xprop -root _NET_WORKAREA 2>/dev/null | tr -d ',' | awk '{print $3, $4, $5, $6}')" "$VIVE_ONLY" <<'PY'
import sys, yaml
config = yaml.safe_load(open(sys.argv[1]))
vive_only = sys.argv[4] == 'true'
for display in config['Visualization Manager']['Displays']:
    if display['Name'] == 'Frame names':
        display['Value'] = False
    if display['Name'] == 'All frames (robot + VR)':
        display['Value'] = True
    if vive_only and display['Name'].startswith(('Baxter', 'Sim scene', 'left_gripper', 'right_gripper')):
        display['Value'] = False
if sys.argv[3]:
    x, y, w, h = map(int, sys.argv[3].split())
    config['Window Geometry'].update({'X': x + w // 2, 'Y': y, 'Width': w - w // 2, 'Height': h})
yaml.safe_dump(config, open(sys.argv[2], 'w'), sort_keys=False)
PY
    RVIZ="$LIVE_RVIZ"
fi
rviz2 -d "$RVIZ" --ros-args "${TF_REMAP[@]}"
