#!/bin/bash
# Teleop Baxter's arms (VR by default) in a Terminator window (tmux without a display; left detached
# when not run from a terminal, e.g. in a cluster job).
# Usage: scripts/run_teleop.sh [--left|--right|--both] [options] [teleop options]
#   --left, --right, --both  arms (default: both)
#   --session NAME           record episodes to rethought_ai/recorder/data/NAME (implies --record)
#   --record                 run the RethoughtAI recorder below the teleop
#   --policy                 play the policy (rethought_ai scripts/run_inference) instead of VR: its plan
#                            through the replay stream (--input plan), the cuff and gripper as always
#   --policy-target          follow the policy's TCP target stream as a VR controller (servo and IK):
#                            for policies trained on teleop goals, which that servo was chasing
#   --swap                   you stand beside Baxter, not facing it
#   --no-gripper             the VR trigger and going home leave the gripper alone
#   --no-camera              do not start the third_eye RealSense
#   --no-viz                 recorder without its Rerun viewer
#   teleop options           any option of python -m baxter_teleop, listed below

# Defaults come from src/baxter_teleop/config/teleop.yaml; options given here override them. Raw
# output bypasses Baxter's collision avoidance. --policy maps the policy's targets 1:1
# (config/vr_axes_policy.yaml: no deadband or smoothing).
# --swap sets swap_hands false: x/y of the axis maps negated, each arm driven by its own hand's
# controller (default: facing Baxter, hands crossed).
# --both runs one teleop per arm side by side, each reading its own keys and /vive/{hand}/ input.
# Recording, VR or kinesthetic, is the same command (--session NAME): drive with the controllers, or
# squeeze a cuff and guide the arm by hand, its dash/circle buttons closing/opening the gripper.
# Episodes: Show starts/stops, OK saves, Back discards (left navigator), or both thumbsticks pushed
# the same way for 2 s (north start, south stop, east save, west discard). The right navigator's
# OK sends both arms home (as B). With one arm the recorder skips the other arm's topics.
# The third_eye RealSense (config/third_eye.yaml) starts in the background unless running; it keeps
# running after the teleop (teardown.sh stops it). Without one on USB it warns and goes on.

if [[ " $* " =~ \ (-h|--help)\  ]]; then
    sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"
    source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
    echo; echo 'Teleop options:'
    "$PY" -m baxter_teleop --help | sed -n '/^options:/,$p' | tail -n +2
    exit 0
fi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${SCRIPT_DIR}/.."
RAI_DIR="$(realpath "${SCRIPT_DIR}/../../rethought_ai")"
# --left / --right / --both pick the arms (default both); --arm X is passed through as before
ARMS=""
GIVEN_ARM=""
ARGS=()  # passed to every teleop, each element one argument
SWAP=""
RECORD=""
CAMERA=1
SESSION=""
POLICY=""
NO_VIZ=""
while (( $# )); do
    case "$1" in
        --left)  ARMS="left" ;;
        --right) ARMS="right" ;;
        --both)  ARMS="left right" ;;
        --arm|--arm=*)
            if [[ "$1" == --arm ]]; then
                GIVEN_ARM="${2:-}"
            else
                GIVEN_ARM="${1#*=}"
            fi
            case "$GIVEN_ARM" in
                left|right) ;;
                *) echo "ERROR: --arm needs left or right" >&2; exit 1 ;;
            esac
            ARGS+=(--arm "$GIVEN_ARM")
            [[ "$1" != --arm ]] || shift
            ;;
        --record) RECORD=1 ;;
        --session|--session=*)
            if [[ "$1" == --session ]]; then SESSION="${2:-}"; shift; else SESSION="${1#*=}"; fi
            # A name, never the next flag: "--session --policy" once created a session called --policy.
            if [[ ! "$SESSION" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
                echo "ERROR: --session needs a name (letters, digits, _ . -), got '${SESSION}'" >&2; exit 1
            fi
            RECORD=1 ;;
        --camera) CAMERA=1 ;;
        --no-camera) CAMERA="" ;;
        --policy) POLICY=plan ;;
        --policy-target) POLICY=target ;;
        --no-viz) NO_VIZ=1 ;;
        --swap)  SWAP=1; ARGS+=(--swap-hands false) ;;
        *)       ARGS+=("$1") ;;
    esac
    shift
done
[[ -z "$ARMS" && -z "$GIVEN_ARM" ]] && ARMS="left right"
# The control settings are the teleop's own defaults (teleop.yaml); --policy adds its own mapping
# (vr_axes_policy.yaml) and topics in teleop_cmd.
if [[ -n "$POLICY" && -n "$SWAP" ]]; then
    echo "ERROR: --swap with --policy would mirror the policy's targets (x/y negated)." >&2; exit 1
fi
ARGS=(--input vr "${ARGS[@]}")

# Each pane sources env.sh. The python is spelled out instead of $PY: the terminal backend runs the
# command through one more shell, which would expand $PY first, to nothing.
teleop_cmd() {  # $1: arm, or empty when --arm was given directly
    local arm="${1:-$GIVEN_ARM}" policy_args=""
    if [[ "$POLICY" == plan ]]; then
        policy_args=" --input plan --vr-config vr_axes_policy.yaml --vr-trigger-topic /policy/${arm}/trigger"
    elif [[ "$POLICY" == target ]]; then
        policy_args=" --vr-config vr_axes_policy.yaml --vr-topic /policy/${arm}/target --vr-trigger-topic /policy/${arm}/trigger"
    fi
    echo "cd '${WS_DIR}' \
  && source '${SCRIPT_DIR}/env.sh' \
  && '${WS_DIR}/.venv-ros/bin/python' -m baxter_teleop$(printf ' %q' "${ARGS[@]}")${1:+ --arm $1}${policy_args}; \
  echo; read -rp 'Teleop exited. Press Enter to close.'"
}
# RethoughtAI recorder; baxter_core_msgs (its button messages) come from the Baxter workspace
record_cmd() {
    # The single quotes go into the pane's command so a session name stays one word; ${SESSION} expands here.
    # shellcheck disable=SC2016
    echo "cd '${RAI_DIR}' \
  && source '${SCRIPT_DIR}/env.sh' \
  && source .venv/bin/activate \
  && record record baxter.yaml${SESSION:+ --session '${SESSION}'}${OTHER_ARM:+ --exclude ${OTHER_ARM}}${NO_VIZ:+ --no-viz}; \
  echo; read -rp 'Recorder exited. Press Enter to close.'"
}
read -ra ARM_LIST <<< "${ARMS:-given}"  # "given": one teleop using the --arm from ARGS
# One arm: the recorder skips the other side's topics (camera, teleop, gripper)
OTHER_ARM=""
if (( ${#ARM_LIST[@]} == 1 )); then
    case "${ARM_LIST[0]/given/$GIVEN_ARM}" in
        left)  OTHER_ARM=right ;;
        right) OTHER_ARM=left ;;
    esac
fi

# Two teleops on one arm fight (one servoing, the other holding or homing it): refuse to start one.
for arm in "${ARM_LIST[@]}"; do
    [[ "$arm" == given ]] && arm="$GIVEN_ARM"
    if running=$(pgrep -f "[-]m baxter_teleop( .*)? --arm ${arm}( |$)"); then
        echo "ERROR: a ${arm} arm teleop is already running (pid ${running//$'\n'/ }). Quit it first (ESC in its pane)." >&2
        exit 1
    fi
done

if [[ -n "$CAMERA" ]] && ! pgrep -f "[r]ealsense2_camera_node.*third_eye" >/dev/null \
    && ! grep -qs RealSense /sys/bus/usb/devices/*/product; then
    echo "WARNING: no RealSense on USB: the third_eye camera is not running. Plug it in and rerun," >&2
    echo "         or pass --no-camera. Recordings and policies (--record, --policy) need it." >&2
    CAMERA=""
fi
if [[ -n "$CAMERA" ]] && ! pgrep -f "[r]ealsense2_camera_node.*third_eye" >/dev/null; then
    CAMERA_LOG="${XDG_RUNTIME_DIR:-/tmp}/baxter_teleop/third_eye.log"
    mkdir -p "$(dirname "$CAMERA_LOG")"
    setsid bash -c "source '${SCRIPT_DIR}/env.sh' && exec ros2 launch realsense2_camera rs_launch.py camera_namespace:=cameras camera_name:=third_eye \
        config_file:='${WS_DIR}/install/baxter_interface/share/baxter_interface/config/third_eye.yaml'" \
        > "$CAMERA_LOG" 2>&1 < /dev/null &
    echo "third_eye camera starting (log: ${CAMERA_LOG})"
fi

# One script per pane, run by either terminal backend: the teleop per arm and, below, record_teleop.sh
# typed in and waiting for Enter, or with --record the RethoughtAI recorder.
RUN_DIR="${XDG_RUNTIME_DIR:-/tmp}/baxter_teleop"
mkdir -p "$RUN_DIR"
for arm in "${ARM_LIST[@]}"; do
    teleop_cmd "${arm/given/}" > "${RUN_DIR}/teleop_${arm}.sh"
done
if [[ -n "$RECORD" ]]; then
    record_cmd > "${RUN_DIR}/record_prompt.sh"
else
    cat > "${RUN_DIR}/record_prompt.sh" <<PROMPT
cd '${SCRIPT_DIR}'
echo 'Press Enter to record this session (edit the label first if you like). Ctrl+C stops recording.'
read -e -p '\$ ' -i './record_teleop.sh baseline' cmd && history -s "\$cmd" && eval "\$cmd"
exec bash
PROMPT
fi

if [[ -n "${DISPLAY:-}" ]] && command -v terminator >/dev/null; then
    # Teleop on top (one pane per arm, left arm on the left, first one focused), the record pane below.
    {
        echo '[global_config]'
        echo '  suppress_multiple_term_dialog = True'
        { sed -n '/^\[profiles\]/,/^\[[a-z]/p' ~/.config/terminator/config 2>/dev/null || true; } | sed '$d'
        cat <<LAYOUT
[layouts]
  [[teleop]]
    [[[window0]]]
      type = Window
      parent = ""
      title = Baxter teleop
      last_active_term = 00000000-0000-0000-0000-000000000010
    [[[panes]]]
      type = VPaned
      parent = window0
    [[[record]]]
      type = Terminal
      parent = panes
      order = 1
      profile = default
      uuid = 00000000-0000-0000-0000-000000000002
      command = bash ${RUN_DIR}/record_prompt.sh
LAYOUT
        TOP=panes
        if (( ${#ARM_LIST[@]} > 1 )); then
            printf '    [[[arms]]]\n      type = HPaned\n      parent = panes\n      order = 0\n'
            TOP=arms
        fi
        for i in "${!ARM_LIST[@]}"; do
            printf '    [[[teleop_%s]]]\n      type = Terminal\n      parent = %s\n      order = %d\n      profile = default\n      uuid = 00000000-0000-0000-0000-00000000001%d\n      command = bash %s/teleop_%s.sh\n' \
                "${ARM_LIST[$i]}" "$TOP" "$i" "$i" "$RUN_DIR" "${ARM_LIST[$i]}"
        done
    } > "${RUN_DIR}/terminator.config"
    # Left half of the desktop work area (RViz from run_sim_teleop.sh takes the right half).
    if read -r X Y W H < <(xprop -root _NET_WORKAREA 2>/dev/null | tr -d ',' | awk '{print $3, $4, $5, $6}'); then
        sed -i "s|      title = Baxter teleop|      title = Baxter teleop\n      position = ${X}:${Y}\n      size = $((W / 2)), ${H}\n      maximised = False|" \
            "${RUN_DIR}/terminator.config"
    fi
    terminator -u -g "${RUN_DIR}/terminator.config" -l teleop --working-directory "${WS_DIR}" &
    echo "Teleop opened in a new Terminator window (one pane per arm, recorder ready below). Click a pane to give it keyboard focus."
else
    # tmux: one window per arm, then the recorder with --record
    for arm in "${ARM_LIST[@]}"; do
        cmd="bash $(printf %q "${RUN_DIR}/teleop_${arm}.sh")"
        if [[ -n "${TMUX:-}" ]]; then
            tmux new-window -n "teleop-${arm}" "$cmd"
        elif tmux has-session -t baxter-teleop 2>/dev/null; then
            tmux new-window -t baxter-teleop -n "teleop-${arm}" "$cmd"
        else
            tmux new-session -d -s baxter-teleop -n "teleop-${arm}" "$cmd"
        fi
    done
    if [[ -n "$RECORD" ]]; then  # the teleop windows above made sure a session exists
        cmd="bash $(printf %q "${RUN_DIR}/record_prompt.sh")"
        if [[ -n "${TMUX:-}" ]]; then tmux new-window -n record "$cmd"
        else tmux new-window -t baxter-teleop -n record "$cmd"; fi
    fi
    [[ -n "${TMUX:-}" || ! -t 0 ]] || tmux attach -t baxter-teleop
fi
