#!/usr/bin/env bash
# Small CPU smoke-test fine-tune for the released UniSHARP ViT-L checkpoint.
# It is intentionally low-resolution and uses portable rendering; use the NPU
# wrapper for an actual fine-tuning run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/UniK3D:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

DATASET_MANIFEST_DIR="${DATASET_MANIFEST_DIR:-${REPO_ROOT}/dataset_manifests}"
FINETUNE_CHECKPOINT="${FINETUNE_CHECKPOINT:-${REPO_ROOT}/checkpoints/released/pretained_model.pt}"
[[ -f "${FINETUNE_CHECKPOINT}" ]] || {
  echo "Official UniSHARP checkpoint was not found: ${FINETUNE_CHECKPOINT}" >&2
  echo "Set FINETUNE_CHECKPOINT=/absolute/path/to/pretained_model.pt" >&2
  exit 1
}

# The official checkpoint is ViT-L. ViT-S/ViT-B are intentionally rejected:
# their parameter shapes do not match the released UniSHARP model.
UNIK3D_BACKBONE="${UNIK3D_BACKBONE:-vitl}"
[[ "${UNIK3D_BACKBONE}" == "vitl" ]] || {
  echo "Official UniSHARP checkpoint requires UNIK3D_BACKBONE=vitl (got ${UNIK3D_BACKBONE})." >&2
  exit 1
}

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs_cpu}"
RUN_NAME="${RUN_NAME:-unisharp_cpu_finetune_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PINHOLE_TRAIN_HEIGHT="${PINHOLE_TRAIN_HEIGHT:-256}"
PINHOLE_TRAIN_WIDTH="${PINHOLE_TRAIN_WIDTH:-384}"
MAX_GAUSSIANS="${MAX_GAUSSIANS:?Set MAX_GAUSSIANS to the explicit CPU portable-renderer Gaussian cap (for example 16384).}"
PORTABLE_MAX_GAUSSIANS_PER_TILE="${PORTABLE_MAX_GAUSSIANS_PER_TILE:-96}"
SAVE_EVERY="${SAVE_EVERY:-10}"

LOCAL_IMAGES_ROOT="${LOCAL_IMAGES_ROOT:-}"
LOCAL_IMAGES_MANIFEST="${LOCAL_IMAGES_MANIFEST:-}"
LOCAL_IMAGES_WEIGHT="${LOCAL_IMAGES_WEIGHT:-0.0}"
COCO_PERSON_MANIFEST="${COCO_PERSON_MANIFEST:-${DATASET_MANIFEST_DIR}/coco_person_train2017_boxes.jsonl}"
COCO_PERSON_WEIGHT="${COCO_PERSON_WEIGHT:-0.0}"
OPENIMAGES_PERSON_MANIFEST="${OPENIMAGES_PERSON_MANIFEST:-${DATASET_MANIFEST_DIR}/openimages_person_train_boxes.jsonl}"
OPENIMAGES_PERSON_WEIGHT="${OPENIMAGES_PERSON_WEIGHT:-0.0}"

DATASET_ARGS=()
add_dataset() {
  local manifest="$1" weight="$2" manifest_flag="$3" weight_flag="$4" label="$5"
  if [[ "${weight}" != "0" && "${weight}" != "0.0" ]]; then
    [[ -f "${manifest}" ]] || { echo "${label} manifest was not found: ${manifest}" >&2; exit 1; }
    DATASET_ARGS+=("${manifest_flag}" "${manifest}" "${weight_flag}" "${weight}")
  fi
}
if [[ "${LOCAL_IMAGES_WEIGHT}" != "0" && "${LOCAL_IMAGES_WEIGHT}" != "0.0" ]]; then
  if [[ -n "${LOCAL_IMAGES_ROOT}" ]]; then
    [[ -d "${LOCAL_IMAGES_ROOT}" ]] || { echo "Local image directory was not found: ${LOCAL_IMAGES_ROOT}" >&2; exit 1; }
    DATASET_ARGS+=(--local-images-root "${LOCAL_IMAGES_ROOT}" --dataset-weight-local-images "${LOCAL_IMAGES_WEIGHT}")
  else
    add_dataset "${LOCAL_IMAGES_MANIFEST}" "${LOCAL_IMAGES_WEIGHT}" --local-images-manifest --dataset-weight-local-images "Local images"
  fi
fi
add_dataset "${COCO_PERSON_MANIFEST}" "${COCO_PERSON_WEIGHT}" --coco-person-manifest --dataset-weight-coco-person "COCO Person"
add_dataset "${OPENIMAGES_PERSON_MANIFEST}" "${OPENIMAGES_PERSON_WEIGHT}" --openimages-person-manifest --dataset-weight-openimages-person "OpenImages Person"
[[ "${#DATASET_ARGS[@]}" -gt 0 ]] || { echo "Enable at least one of LOCAL_IMAGES_WEIGHT, COCO_PERSON_WEIGHT, or OPENIMAGES_PERSON_WEIGHT." >&2; exit 1; }

# Preserve the released checkpoint's stride=1 Gaussian grid; stride=2 would
# generate only one quarter of the Gaussians at equal resolution.
exec python -m unisharp.cli train-feature \
  --device cpu --renderer-backend portable \
  --portable-renderer-max-gaussians "${MAX_GAUSSIANS}" \
  --portable-renderer-max-gaussians-per-tile "${PORTABLE_MAX_GAUSSIANS_PER_TILE}" \
  --out-root "${OUT_ROOT}" --run-name "${RUN_NAME}" \
  --steps "${STEPS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --pinhole-train-height "${PINHOLE_TRAIN_HEIGHT}" --pinhole-train-width "${PINHOLE_TRAIN_WIDTH}" --train-resize-multiple 0 \
  --unik3d-backbone vitl --initializer-stride 1 \
  --init-checkpoint "${FINETUNE_CHECKPOINT}" --init-checkpoint-strict \
  --unik3d-progressive-unfreeze --unik3d-decoder-unfreeze-step 5 --unik3d-encoder-unfreeze-step 15 --unik3d-encoder-last-n-blocks 4 \
  --lr0 2e-5 --lr1 2e-6 --unik3d-lr0 5e-6 --unik3d-lr1 5e-7 --unik3d-encoder-lr0 5e-7 --unik3d-encoder-lr1 5e-8 \
  --save-every "${SAVE_EVERY}" --log-every 1 --vis-every 0 --lambda-percep 0 \
  --dataset-weight-re10k 0 --dataset-weight-hm3d 0 --dataset-weight-sim 0 \
  --dataset-weight-wildrgbd 0 --dataset-weight-dl3dv 0 --dataset-weight-scanetpp 0 \
  "${DATASET_ARGS[@]}" "$@"
