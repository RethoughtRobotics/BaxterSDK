#!/bin/bash
# Markerless calibration of the third-eye camera in Baxter's base frame, from recorded bags (EasyHeC).
# Usage: scripts/calibrate_camera.sh BAGS [HELD_OUT_BAGS] [--frames N]
#   BAGS           a bag, or a folder of episode_* bags (e.g. a recorder dataset), with the third-eye camera,
#                  /robot/joint_states, and the arms still in varied poses at times
#   HELD_OUT_BAGS  other bags, used only to check the result (mask IoU of the rendered robot)
# Writes data/calibration/third_eye/<time>/: frames, masks.jpg (check the masks), extrinsic.yaml, overlays.
# Needs ../simple_easyhec/.venv (simple-easyhec with nvdiffrast) and rethought_ai's venv (SAM3).

if [[ " $* " =~ \ (-h|--help)\  ]]; then sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 0; fi
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[[ $# -ge 1 ]] || { sed -n '2,/^$/{/^#/s/^# \{0,1\}//p}' "$0"; exit 1; }
BAG=$1; shift
HELD=""; [[ $# -ge 1 && $1 != --* ]] && { HELD=$1; shift; }
CAL="${SCRIPT_DIR}/calibrate_camera.py"
OUT="${DATA_DIR}/calibration/third_eye/$(date +%Y%m%d_%H%M%S)"
EASYHEC_DIR="${EASYHEC_DIR:-$(realpath -m "${WS_DIR}/../simple_easyhec")}"
RAI_PY="${RAI_DIR}/.venv/bin/python"

# nvdiffrast compiles its CUDA plugin on first use: CUDA 12.6 rejects gcc 14, the toolkit lacks the cuSPARSE headers
# torch includes, and building for this GPU's architecture only avoids a PTX JIT step on driver 535.
easyhec() {
  local nv="${EASYHEC_DIR}/.venv/lib/python3.11/site-packages/nvidia"
  PATH="${EASYHEC_DIR}/.venv/bin:/usr/local/cuda-12.6/bin:${PATH}" CUDA_HOME=/usr/local/cuda-12.6 \
    CPATH="$(printf "%s\n" "${nv}"/*/include | paste -sd:)" CC=gcc-13 CXX=g++-13 NVCC_CCBIN=g++-13 \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" "${EASYHEC_DIR}/.venv/bin/python" "$CAL" "$@"
}

"$PY" "$CAL" export "$OUT" "$BAG" "$@"
"$RAI_PY" "$CAL" masks "$OUT" --rai-dir "$RAI_DIR"
easyhec solve "$OUT"
if [[ -n $HELD ]]; then
  "$PY" "$CAL" export "$OUT/held_out" "$HELD" "$@"
  "$RAI_PY" "$CAL" masks "$OUT/held_out" --rai-dir "$RAI_DIR"
  easyhec verify "$OUT" "$OUT/held_out"
fi
echo "Calibration: $OUT/extrinsic.yaml"
