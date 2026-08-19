#!/usr/bin/env bash
# Single-card Ascend NPU pre-training for UniSHARP's pinhole branch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_ID="${NPU_ID:-0}"

if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script was not found: ${CANN_ENV}" >&2
  exit 1
fi
source "${CANN_ENV}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/UniK3D:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

: "${DATA_ROOT_RE10K:?Set DATA_ROOT_RE10K to the RE10K training-data root.}"
: "${DATASET_MANIFEST_DIR:?Set DATASET_MANIFEST_DIR to the UniSHARP manifests directory.}"

python "${SCRIPT_DIR}/check_npu_env.py"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs_npu}"
RUN_NAME="${RUN_NAME:-unisharp_npu_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PINHOLE_TRAIN_SIZE="${PINHOLE_TRAIN_SIZE:-256}"
UNIK3D_BACKBONE="${UNIK3D_BACKBONE:-vits}"
INITIALIZER_STRIDE="${INITIALIZER_STRIDE:-2}"
PORTABLE_MAX_GAUSSIANS="${PORTABLE_MAX_GAUSSIANS:-8192}"
PORTABLE_MAX_GAUSSIANS_PER_TILE="${PORTABLE_MAX_GAUSSIANS_PER_TILE:-128}"

exec python -m unisharp.cli train-feature \
  --device npu \
  --renderer-backend portable \
  --portable-renderer-max-gaussians "${PORTABLE_MAX_GAUSSIANS}" \
  --portable-renderer-max-gaussians-per-tile "${PORTABLE_MAX_GAUSSIANS_PER_TILE}" \
  --out-root "${OUT_ROOT}" --run-name "${RUN_NAME}" \
  --steps "${STEPS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --pinhole-train-size "${PINHOLE_TRAIN_SIZE}" --train-resize-multiple 0 \
  --unik3d-backbone "${UNIK3D_BACKBONE}" --initializer-stride "${INITIALIZER_STRIDE}" \
  --data-root-re10k "${DATA_ROOT_RE10K}" --dataset-manifest-dir "${DATASET_MANIFEST_DIR}" \
  --dataset-weight-re10k 1 --dataset-weight-hm3d 0 --dataset-weight-sim 0 \
  --dataset-weight-wildrgbd 0 --dataset-weight-dl3dv 0 --dataset-weight-scanetpp 0 \
  --lambda-percep 0 --vis-every 0 \
  "$@"
