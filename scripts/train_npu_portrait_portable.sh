#!/usr/bin/env bash
# Ascend NPU pre-training on your local photos (single card).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_ID="${NPU_ID:-0}"
[[ -f "${CANN_ENV}" ]] || { echo "CANN environment script was not found: ${CANN_ENV}" >&2; exit 1; }
source "${CANN_ENV}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/UniK3D:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

: "${DATASET_MANIFEST_DIR:?Set DATASET_MANIFEST_DIR (normally <repo>/dataset_manifests).}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs_npu}"
RUN_NAME="${RUN_NAME:-unisharp_portrait_npu_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
UNIK3D_BACKBONE="${UNIK3D_BACKBONE:-vits}"
PINHOLE_TRAIN_HEIGHT="${PINHOLE_TRAIN_HEIGHT:-1024}"
PINHOLE_TRAIN_WIDTH="${PINHOLE_TRAIN_WIDTH:-1536}"
UNIK3D_PROGRESSIVE_UNFREEZE="${UNIK3D_PROGRESSIVE_UNFREEZE:-1}"
UNIK3D_DECODER_UNFREEZE_STEP="${UNIK3D_DECODER_UNFREEZE_STEP:-20000}"
UNIK3D_ENCODER_UNFREEZE_STEP="${UNIK3D_ENCODER_UNFREEZE_STEP:-50000}"
UNIK3D_ENCODER_LAST_N_BLOCKS="${UNIK3D_ENCODER_LAST_N_BLOCKS:-4}"
UNIK3D_ENCODER_FULL_UNFREEZE_STEP="${UNIK3D_ENCODER_FULL_UNFREEZE_STEP:-0}"
LOCAL_IMAGES_ROOT="${LOCAL_IMAGES_ROOT:-}"
LOCAL_IMAGES_MANIFEST="${LOCAL_IMAGES_MANIFEST:-${DATASET_MANIFEST_DIR}/local_images.txt}"
LOCAL_COLMAP_ROOT="${LOCAL_COLMAP_ROOT:-}"
LOCAL_MULTIVIEW_MANIFEST="${LOCAL_MULTIVIEW_MANIFEST:-${DATASET_MANIFEST_DIR}/local_multiview.jsonl}"
LOCAL_IMAGES_WEIGHT="${LOCAL_IMAGES_WEIGHT:-1.0}"
LOCAL_MULTIVIEW_WEIGHT="${LOCAL_MULTIVIEW_WEIGHT:-0.0}"

if [[ ! -f "${LOCAL_IMAGES_MANIFEST}" && -n "${LOCAL_IMAGES_ROOT}" ]]; then
  python "${SCRIPT_DIR}/prepare_local_images.py" --source-root "${LOCAL_IMAGES_ROOT}" --manifest "${LOCAL_IMAGES_MANIFEST}"
fi
if [[ ! -f "${LOCAL_MULTIVIEW_MANIFEST}" && -n "${LOCAL_COLMAP_ROOT}" ]]; then
  python "${SCRIPT_DIR}/prepare_calibrated_colmap.py" --source-root "${LOCAL_COLMAP_ROOT}" --manifest "${LOCAL_MULTIVIEW_MANIFEST}"
fi
LOCAL_ARGS=()
if [[ "${LOCAL_IMAGES_WEIGHT}" != "0" && "${LOCAL_IMAGES_WEIGHT}" != "0.0" ]]; then
  [[ -f "${LOCAL_IMAGES_MANIFEST}" ]] || { echo "Set LOCAL_IMAGES_ROOT or LOCAL_IMAGES_MANIFEST." >&2; exit 1; }
  LOCAL_ARGS+=(--local-images-manifest "${LOCAL_IMAGES_MANIFEST}" --dataset-weight-local-images "${LOCAL_IMAGES_WEIGHT}")
fi
if [[ "${LOCAL_MULTIVIEW_WEIGHT}" != "0" && "${LOCAL_MULTIVIEW_WEIGHT}" != "0.0" ]]; then
  [[ -f "${LOCAL_MULTIVIEW_MANIFEST}" ]] || { echo "Set LOCAL_COLMAP_ROOT or LOCAL_MULTIVIEW_MANIFEST." >&2; exit 1; }
  LOCAL_ARGS+=(--local-multiview-manifest "${LOCAL_MULTIVIEW_MANIFEST}" --dataset-weight-local-multiview "${LOCAL_MULTIVIEW_WEIGHT}")
fi
[[ "${#LOCAL_ARGS[@]}" -gt 0 ]] || { echo "At least one local dataset weight must be positive." >&2; exit 1; }
UNIK3D_UNFREEZE_ARGS=(
  --unik3d-decoder-unfreeze-step "${UNIK3D_DECODER_UNFREEZE_STEP}"
  --unik3d-encoder-unfreeze-step "${UNIK3D_ENCODER_UNFREEZE_STEP}"
  --unik3d-encoder-last-n-blocks "${UNIK3D_ENCODER_LAST_N_BLOCKS}"
  --unik3d-encoder-full-unfreeze-step "${UNIK3D_ENCODER_FULL_UNFREEZE_STEP}"
)
if [[ "${UNIK3D_PROGRESSIVE_UNFREEZE}" == "0" || "${UNIK3D_PROGRESSIVE_UNFREEZE}" == "false" ]]; then
  UNIK3D_UNFREEZE_ARGS+=(--no-unik3d-progressive-unfreeze)
else
  UNIK3D_UNFREEZE_ARGS+=(--unik3d-progressive-unfreeze)
fi
python "${SCRIPT_DIR}/check_npu_env.py"

exec python -m unisharp.cli train-feature \
  --device npu --renderer-backend portable \
  --out-root "${OUT_ROOT}" --run-name "${RUN_NAME}" --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --pinhole-train-height "${PINHOLE_TRAIN_HEIGHT}" --pinhole-train-width "${PINHOLE_TRAIN_WIDTH}" --train-resize-multiple 0 \
  --unik3d-backbone "${UNIK3D_BACKBONE}" --initializer-stride 2 \
  "${UNIK3D_UNFREEZE_ARGS[@]}" \
  --dataset-weight-re10k 0 --dataset-weight-hm3d 0 --dataset-weight-sim 0 --dataset-weight-wildrgbd 0 --dataset-weight-dl3dv 0 --dataset-weight-scanetpp 0 \
  --lambda-percep 0 --vis-every 0 "${LOCAL_ARGS[@]}" "$@"
