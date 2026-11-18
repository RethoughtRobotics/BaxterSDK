#!/usr/bin/env bash
# Inside the sim image (docker/sim/eval.sh starts it): build what this checkout is missing, start the
# headless stack as scripts/run_sim_teleop.sh --cup --cameras --left --policy-target does, run one
# closed-loop evaluation, and stop.
#   stack.sh MODEL [evaluator args...]     e.g. stack.sh /path/ckpt --cup-color red --seeds 1000-1004 --out x.jsonl
# ADAPTER (a nudge adapter directory) runs the policy with it at alpha 1.
# The container has its own network, so its Zenoh router and ROS domain are its own.
set -eo pipefail
WS="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"  # this BaxterSDK checkout
RAI="$(realpath -m "${WS}/../rethought_ai")"
MODEL="$1"
shift
LOG="${LOG_DIR:-/tmp}/sim_stack"
mkdir -p "$LOG"

set +u
source /opt/ros/kilted/setup.bash
set -u
if [[ ! -f "${WS}/install/setup.bash" ]]; then
    (cd "$WS" && colcon build --packages-up-to baxter_teleop baxter_tools baxter_description > "${LOG}/colcon.log" 2>&1)
fi
if [[ ! -x "${WS}/.venv-ros/bin/python" ]]; then  # src/baxter_teleop/requirements.txt: ruckig only
    python3.12 -m venv --system-site-packages "${WS}/.venv-ros"
    "${WS}/.venv-ros/bin/pip" install -q -r "${WS}/src/baxter_teleop/requirements.txt"
fi
(cd "${WS}/src/baxter_mujoco" && uv sync -q --group ros)
source "${WS}/scripts/env.sh"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-43}"
if [[ -n "${ZENOH_PORT:-}" ]]; then  # a shared network (Apptainer on a cluster node): a router of our own
    cfg=/opt/ros/kilted/share/rmw_zenoh_cpp/config
    sed "s|\"tcp/\[::\]:7447\"|\"tcp/127.0.0.1:${ZENOH_PORT}\"|" "${cfg}/DEFAULT_RMW_ZENOH_ROUTER_CONFIG.json5" > "${LOG}/router.json5"
    sed "s|\"tcp/localhost:7447\"|\"tcp/127.0.0.1:${ZENOH_PORT}\"|" "${cfg}/DEFAULT_RMW_ZENOH_SESSION_CONFIG.json5" > "${LOG}/session.json5"
    if ! grep -q "127.0.0.1:${ZENOH_PORT}" "${LOG}/router.json5" || ! grep -q "127.0.0.1:${ZENOH_PORT}" "${LOG}/session.json5"; then
        echo "could not set the Zenoh port in ${LOG}" >&2
        exit 2
    fi
    export ZENOH_ROUTER_CONFIG_URI="${LOG}/router.json5" ZENOH_SESSION_CONFIG_URI="${LOG}/session.json5"
fi

pids=()
stop_all() {  # each started command's whole process group
    local p
    for p in "${pids[@]}"; do kill -INT -- "-$p" 2>/dev/null || true; done
}
trap stop_all EXIT
start() {  # $1 log name, then the command, in its own process group
    local name="$1"
    shift
    setsid "$@" > "${LOG}/${name}.log" 2>&1 &
    pids+=($!)
}
wait_for_topic() {
    local deadline=$((SECONDS + $2))
    while ((SECONDS < deadline)); do
        timeout -k 2 5 ros2 topic echo --once "$1" > /dev/null 2>&1 && return 0
        sleep 0.5
    done
    echo "no $1 within $2 s (logs: ${LOG})" >&2
    return 1
}

# Our own router on our own port: launch_sim.sh only starts one when none runs, and on a shared node
# another job's would count
[[ -z "${ZENOH_PORT:-}" ]] || start router ros2 run rmw_zenoh_cpp rmw_zenohd
start sim "${SIM_DIR}/launch_sim.sh" --ros --scene cup --no-viewer --cameras
wait_for_topic /robot/joint_states 120
ros2 run baxter_tools tuck_arms -u > "${LOG}/tuck.log" 2>&1
start teleop "${WS}/.venv-ros/bin/python" -m baxter_teleop --input vr --arm left \
    --vr-config vr_axes_policy.yaml --vr-topic /policy/left/target --vr-trigger-topic /policy/left/trigger
# shellcheck disable=SC2016  # the outer double quotes expand ${RAI} and ${MODEL}; the inner single quotes keep paths whole
start policy bash -c "cd '${RAI}' && source launch/env.sh && exec ros2 launch launch/policy.launch.py \
    model:='${MODEL}' config:=configs/baxter.yaml device:=cuda${ADAPTER:+ adapter:='${ADAPTER}' alpha:=1.0}"
cd "${WS}/src/baxter_mujoco"
uv run --group ros python -m baxter_mujoco.datagen.evaluate --task cup "$@"
