#!/usr/bin/env bash
# Render an exported UniSHARP gaussians.pt with CANN's fused 3DGS NPU ops.
# Pass ordinary render_unisharp_cpu.py rig options after this wrapper, e.g.
# --gaussians scene/gaussians.pt --output outputs/npu_grid9 --rig grid9
# --height 512 --width 768.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_ID="${NPU_ID:-0}"

[[ -f "${CANN_ENV}" ]] || { echo "CANN environment script was not found: ${CANN_ENV}" >&2; exit 1; }
source "${CANN_ENV}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/UniK3D:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/check_npu_env.py" --require-meta-gauss-render
exec python "${SCRIPT_DIR}/render_unisharp_cpu.py" --trajectory rig --backend ascend_fused "$@"
