#!/usr/bin/env bash
# Ascend NPU pre-training on your local photos (single-node multi-card).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_IDS="${NPU_IDS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29531}"
[[ -f "${CANN_ENV}" ]] || { echo "CANN environment script was not found: ${CANN_ENV}" >&2; exit 1; }
source "${CANN_ENV}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_IDS}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/UniK3D:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"

: "${DATASET_MANIFEST_DIR:?Set DATASET_MANIFEST_DIR (normally <repo>/dataset_manifests).}"
IFS=',' read -r -a NPU_ID_ARRAY <<< "${NPU_IDS}"
WORLD_SIZE="${#NPU_ID_ARRAY[@]}"
[[ "${WORLD_SIZE}" -ge 2 ]] || { echo "NPU_IDS must contain at least two IDs." >&2; exit 1; }

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs_npu}"
RUN_NAME="${RUN_NAME:-unisharp_portrait_npu_ddp_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-100000}"; BATCH_SIZE="${BATCH_SIZE:-1}"; NUM_WORKERS="${NUM_WORKERS:-2}"
UNIK3D_BACKBONE="${UNIK3D_BACKBONE:-vits}"
PINHOLE_TRAIN_HEIGHT="${PINHOLE_TRAIN_HEIGHT:-1024}"; PINHOLE_TRAIN_WIDTH="${PINHOLE_TRAIN_WIDTH:-1536}"
LOCAL_IMAGES_ROOT="${LOCAL_IMAGES_ROOT:-}"
LOCAL_IMAGES_MANIFEST="${LOCAL_IMAGES_MANIFEST:-${DATASET_MANIFEST_DIR}/local_images.txt}"
LOCAL_COLMAP_ROOT="${LOCAL_COLMAP_ROOT:-}"
LOCAL_MULTIVIEW_MANIFEST="${LOCAL_MULTIVIEW_MANIFEST:-${DATASET_MANIFEST_DIR}/local_multiview.jsonl}"
LOCAL_IMAGES_WEIGHT="${LOCAL_IMAGES_WEIGHT:-1.0}"; LOCAL_MULTIVIEW_WEIGHT="${LOCAL_MULTIVIEW_WEIGHT:-0.0}"

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
python "${SCRIPT_DIR}/check_npu_env.py"

exec torchrun --standalone --nproc_per_node="${WORLD_SIZE}" --master_port="${MASTER_PORT}" -m unisharp.cli train-feature \
  --device npu --renderer-backend portable --out-root "${OUT_ROOT}" --run-name "${RUN_NAME}" --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --pinhole-train-height "${PINHOLE_TRAIN_HEIGHT}" --pinhole-train-width "${PINHOLE_TRAIN_WIDTH}" --train-resize-multiple 0 \
  --unik3d-backbone "${UNIK3D_BACKBONE}" --initializer-stride 2 \
  --dataset-weight-re10k 0 --dataset-weight-hm3d 0 --dataset-weight-sim 0 --dataset-weight-wildrgbd 0 --dataset-weight-dl3dv 0 --dataset-weight-scanetpp 0 \
  --lambda-percep 0 --vis-every 0 "${LOCAL_ARGS[@]}" "$@"
