#!/usr/bin/env bash
# Run one headless closed-loop evaluation in the sim image (docker/sim/Dockerfile), on a GPU machine.
#   docker/sim/eval.sh [--gpu N] [--name NAME] MODEL [evaluator args...]
# MODEL is a checkpoint directory; the evaluator args are baxter_mujoco.datagen.evaluate's
# (--cup-color, --seeds, --out, --bags, --bag-images, ...), paths as seen on this machine. Several can
# run at once: each container has its own network, Zenoh router and ROS domain.
set -eo pipefail
IMAGE="${IMAGE:-baxter-sim:latest}"
GPU=0
NAME="baxter-sim-$$"
while [[ "${1:-}" == --* ]]; do
    case "$1" in
        --gpu) GPU="$2"; shift 2 ;;
        --name) NAME="$2"; shift 2 ;;
        *) break ;;
    esac
done
WS="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
# The Hugging Face cache where it really is (it can be a symlink out of the home directory)
HF="$(realpath -m "${HF_HOME:-$HOME/.cache/huggingface}")"
# The home directory at its own path: the checkout, its venvs (which point at /usr/bin/python3.12,
# the image's too), and the uv and Hugging Face caches. Rootless Docker: the container's root is this
# user on the host, so no --user (any other user in the container cannot read the home directory).
exec docker run --rm --name "$NAME" --runtime nvidia --gpus "device=${GPU}" \
    -e HOME="$HOME" -v "$HOME:$HOME" -w "$WS" -e HF_HOME="$HF" -v "$HF:$HF" \
    -e UV_NO_SYNC=1 \
    -e LOG_DIR="${LOG_DIR:-$HOME/.cache/baxter_sim/${NAME}}" \
    "$IMAGE" "${WS}/docker/sim/stack.sh" "$@"
