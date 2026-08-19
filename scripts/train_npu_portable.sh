#!/usr/bin/env bash
# Single-card Ascend NPU mixed pre-training for UniSHARP's pinhole branch.
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

: "${DATASET_MANIFEST_DIR:?Set DATASET_MANIFEST_DIR to the UniSHARP manifests directory.}"
: "${DATA_ROOT_RE10K:?Set DATA_ROOT_RE10K to the processed RE10K root.}"
: "${DATA_ROOT_WILDRGBD:?Set DATA_ROOT_WILDRGBD to the downloaded WildRGB-D root.}"
: "${DATA_ROOT_DL3DV:?Set DATA_ROOT_DL3DV to the DL3DV RGB/pose root.}"
: "${DATA_ROOT_DL3DV_DEPTH:?Set DATA_ROOT_DL3DV_DEPTH to the prepared DL3DV depth root.}"

python "${SCRIPT_DIR}/check_npu_env.py"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs_npu}"
RUN_NAME="${RUN_NAME:-unisharp_npu_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Match the native GPU training crop geometry (width x height = 1536 x 1024).
# The full gsplat-reference renderer is intentionally expensive at this size.
PINHOLE_TRAIN_HEIGHT="${PINHOLE_TRAIN_HEIGHT:-1024}"
PINHOLE_TRAIN_WIDTH="${PINHOLE_TRAIN_WIDTH:-1536}"
UNIK3D_BACKBONE="${UNIK3D_BACKBONE:-vits}"
INITIALIZER_STRIDE="${INITIALIZER_STRIDE:-2}"
# Zero disables approximate culling and follows the gsplat-classic reference
# math. Set finite values only when an explicit speed/memory trade-off is OK.
PORTABLE_MAX_GAUSSIANS="${PORTABLE_MAX_GAUSSIANS:-0}"
PORTABLE_MAX_GAUSSIANS_PER_TILE="${PORTABLE_MAX_GAUSSIANS_PER_TILE:-0}"
DATASET_WEIGHT_RE10K="${DATASET_WEIGHT_RE10K:-1}"
DATASET_WEIGHT_WILDRGBD="${DATASET_WEIGHT_WILDRGBD:-1}"
DATASET_WEIGHT_DL3DV="${DATASET_WEIGHT_DL3DV:-1}"
DATASET_FRACTION_RE10K="${DATASET_FRACTION_RE10K:-0.10}"
DATASET_FRACTION_WILDRGBD="${DATASET_FRACTION_WILDRGBD:-1.0}"
DATASET_FRACTION_DL3DV="${DATASET_FRACTION_DL3DV:-1.0}"
WILD_ROOTS_FILE="${WILD_ROOTS_FILE:-${DATASET_MANIFEST_DIR}/wildrgbd_roots.txt}"

exec python -m unisharp.cli train-feature \
  --device npu \
  --renderer-backend portable \
  --portable-renderer-max-gaussians "${PORTABLE_MAX_GAUSSIANS}" \
  --portable-renderer-max-gaussians-per-tile "${PORTABLE_MAX_GAUSSIANS_PER_TILE}" \
  --out-root "${OUT_ROOT}" --run-name "${RUN_NAME}" \
  --steps "${STEPS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --pinhole-train-height "${PINHOLE_TRAIN_HEIGHT}" --pinhole-train-width "${PINHOLE_TRAIN_WIDTH}" --train-resize-multiple 0 \
  --unik3d-backbone "${UNIK3D_BACKBONE}" --initializer-stride "${INITIALIZER_STRIDE}" \
  --data-root-re10k "${DATA_ROOT_RE10K}" \
  --data-root-wildrgbd "${DATA_ROOT_WILDRGBD}" --wild-roots-file "${WILD_ROOTS_FILE}" \
  --data-root-dl3dv "${DATA_ROOT_DL3DV}" --data-root-dl3dv-depth "${DATA_ROOT_DL3DV_DEPTH}" \
  --dataset-manifest-dir "${DATASET_MANIFEST_DIR}" \
  --dataset-weight-re10k "${DATASET_WEIGHT_RE10K}" --dataset-weight-hm3d 0 --dataset-weight-sim 0 \
  --dataset-weight-wildrgbd "${DATASET_WEIGHT_WILDRGBD}" --dataset-weight-dl3dv "${DATASET_WEIGHT_DL3DV}" --dataset-weight-scanetpp 0 \
  --dataset-fraction-re10k "${DATASET_FRACTION_RE10K}" --dataset-fraction-wildrgbd "${DATASET_FRACTION_WILDRGBD}" --dataset-fraction-dl3dv "${DATASET_FRACTION_DL3DV}" \
  --lambda-percep 0 --vis-every 0 \
  "$@"
