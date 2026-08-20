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
HUMBI_MANIFEST="${HUMBI_MANIFEST:-${DATASET_MANIFEST_DIR}/humbi_train.jsonl}"
NERSEMBLE_MANIFEST="${NERSEMBLE_MANIFEST:-${DATASET_MANIFEST_DIR}/nersemble_train.jsonl}"
AISTPP_MANIFEST="${AISTPP_MANIFEST:-${DATASET_MANIFEST_DIR}/aistpp_train.jsonl}"
THUMAN_MANIFEST="${THUMAN_MANIFEST:-${DATASET_MANIFEST_DIR}/thuman_train.jsonl}"
[[ -f "${COCO_PERSON_MANIFEST}" ]] || { echo "Missing COCO manifest: ${COCO_PERSON_MANIFEST}" >&2; exit 1; }
[[ -f "${HUMBI_MANIFEST}" ]] || { echo "Missing HUMBI manifest: ${HUMBI_MANIFEST}" >&2; exit 1; }
[[ -f "${NERSEMBLE_MANIFEST}" ]] || { echo "Missing NeRSemble manifest: ${NERSEMBLE_MANIFEST}" >&2; exit 1; }
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
HUMBI_WEIGHT="${HUMBI_WEIGHT:-1.0}"
NERSEMBLE_WEIGHT="${NERSEMBLE_WEIGHT:-1.0}"
AISTPP_WEIGHT="${AISTPP_WEIGHT:-0.0}"
THUMAN_WEIGHT="${THUMAN_WEIGHT:-0.0}"

OPTIONAL_MULTIVIEW_ARGS=()
if [[ "${AISTPP_WEIGHT}" != "0" && "${AISTPP_WEIGHT}" != "0.0" ]]; then
  [[ -f "${AISTPP_MANIFEST}" ]] || { echo "Missing AIST++ manifest: ${AISTPP_MANIFEST}" >&2; exit 1; }
  OPTIONAL_MULTIVIEW_ARGS+=(--aistpp-manifest "${AISTPP_MANIFEST}" --dataset-weight-aistpp "${AISTPP_WEIGHT}")
fi
if [[ "${THUMAN_WEIGHT}" != "0" && "${THUMAN_WEIGHT}" != "0.0" ]]; then
  [[ -f "${THUMAN_MANIFEST}" ]] || { echo "Missing THuman manifest: ${THUMAN_MANIFEST}" >&2; exit 1; }
  OPTIONAL_MULTIVIEW_ARGS+=(--thuman-manifest "${THUMAN_MANIFEST}" --dataset-weight-thuman "${THUMAN_WEIGHT}")
fi

exec python -m unisharp.cli train-feature \
  --device npu --renderer-backend portable \
  --out-root "${OUT_ROOT}" --run-name "${RUN_NAME}" --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --pinhole-train-height "${PINHOLE_TRAIN_HEIGHT}" --pinhole-train-width "${PINHOLE_TRAIN_WIDTH}" --train-resize-multiple 0 \
  --unik3d-backbone "${UNIK3D_BACKBONE}" --initializer-stride 2 \
  --coco-person-manifest "${COCO_PERSON_MANIFEST}" --humbi-manifest "${HUMBI_MANIFEST}" --nersemble-manifest "${NERSEMBLE_MANIFEST}" \
  --dataset-weight-re10k 0 --dataset-weight-hm3d 0 --dataset-weight-sim 0 --dataset-weight-wildrgbd 0 --dataset-weight-dl3dv 0 --dataset-weight-scanetpp 0 \
  --dataset-weight-coco-person "${COCO_WEIGHT}" --dataset-weight-humbi "${HUMBI_WEIGHT}" --dataset-weight-nersemble "${NERSEMBLE_WEIGHT}" \
  --lambda-percep 0 --vis-every 0 "${OPTIONAL_MULTIVIEW_ARGS[@]}" "$@"
