#!/usr/bin/env bash
# Ascend NPU portrait/outdoor appearance pre-training (single card).
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
COCO_PERSON_MANIFEST="${COCO_PERSON_MANIFEST:-${DATASET_MANIFEST_DIR}/coco_person_train2017_boxes.jsonl}"
WIDERFACE_MANIFEST="${WIDERFACE_MANIFEST:-${DATASET_MANIFEST_DIR}/widerface_boxes.jsonl}"
 : "${FACESCAPE_MANIFEST:?Set FACESCAPE_MANIFEST to prepared calibrated FaceScape JSONL.}"
 : "${MVHUMANNET_MANIFEST:?Set MVHUMANNET_MANIFEST to prepared calibrated MVHumanNet JSONL.}"
[[ -f "${COCO_PERSON_MANIFEST}" ]] || { echo "Missing COCO manifest: ${COCO_PERSON_MANIFEST}" >&2; exit 1; }
[[ -f "${WIDERFACE_MANIFEST}" ]] || { echo "Missing WIDER FACE manifest: ${WIDERFACE_MANIFEST}" >&2; exit 1; }
[[ -f "${FACESCAPE_MANIFEST}" ]] || { echo "Missing FaceScape manifest: ${FACESCAPE_MANIFEST}" >&2; exit 1; }
[[ -f "${MVHUMANNET_MANIFEST}" ]] || { echo "Missing MVHumanNet manifest: ${MVHUMANNET_MANIFEST}" >&2; exit 1; }
python "${SCRIPT_DIR}/check_npu_env.py"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs_npu}"
RUN_NAME="${RUN_NAME:-unisharp_portrait_npu_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
UNIK3D_BACKBONE="${UNIK3D_BACKBONE:-vits}"
PINHOLE_TRAIN_HEIGHT="${PINHOLE_TRAIN_HEIGHT:-1024}"
PINHOLE_TRAIN_WIDTH="${PINHOLE_TRAIN_WIDTH:-1536}"
COCO_WEIGHT="${COCO_WEIGHT:-1.0}"
WIDERFACE_WEIGHT="${WIDERFACE_WEIGHT:-1.0}"
FACESCAPE_WEIGHT="${FACESCAPE_WEIGHT:-1.0}"
MVHUMANNET_WEIGHT="${MVHUMANNET_WEIGHT:-1.0}"

exec python -m unisharp.cli train-feature \
  --device npu --renderer-backend portable \
  --out-root "${OUT_ROOT}" --run-name "${RUN_NAME}" --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --pinhole-train-height "${PINHOLE_TRAIN_HEIGHT}" --pinhole-train-width "${PINHOLE_TRAIN_WIDTH}" --train-resize-multiple 0 \
  --unik3d-backbone "${UNIK3D_BACKBONE}" --initializer-stride 2 \
  --coco-person-manifest "${COCO_PERSON_MANIFEST}" --widerface-manifest "${WIDERFACE_MANIFEST}" --facescape-manifest "${FACESCAPE_MANIFEST}" --mvhumannet-manifest "${MVHUMANNET_MANIFEST}" \
  --dataset-weight-re10k 0 --dataset-weight-hm3d 0 --dataset-weight-sim 0 --dataset-weight-wildrgbd 0 --dataset-weight-dl3dv 0 --dataset-weight-scanetpp 0 \
  --dataset-weight-coco-person "${COCO_WEIGHT}" --dataset-weight-widerface "${WIDERFACE_WEIGHT}" --dataset-weight-facescape "${FACESCAPE_WEIGHT}" --dataset-weight-mvhumannet "${MVHUMANNET_WEIGHT}" \
  --lambda-percep 0 --vis-every 0 "$@"
