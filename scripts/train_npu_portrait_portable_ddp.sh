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
NPU_RENDERER_BACKEND="${NPU_RENDERER_BACKEND:-ascend_fused}"
[[ "${NPU_RENDERER_BACKEND}" == "ascend_fused" || "${NPU_RENDERER_BACKEND}" == "portable" ]] || {
  echo "NPU_RENDERER_BACKEND must be ascend_fused or portable." >&2
  exit 1
}
# Fresh lightweight training may use stride=2 to control cost.  Official
# checkpoint fine-tuning sets this to 1 in its wrapper to preserve its grid.
INITIALIZER_STRIDE="${INITIALIZER_STRIDE:-2}"
PINHOLE_TRAIN_HEIGHT="${PINHOLE_TRAIN_HEIGHT:-1024}"; PINHOLE_TRAIN_WIDTH="${PINHOLE_TRAIN_WIDTH:-1536}"
UNIK3D_PROGRESSIVE_UNFREEZE="${UNIK3D_PROGRESSIVE_UNFREEZE:-1}"
UNIK3D_DECODER_UNFREEZE_STEP="${UNIK3D_DECODER_UNFREEZE_STEP:-20000}"
UNIK3D_ENCODER_UNFREEZE_STEP="${UNIK3D_ENCODER_UNFREEZE_STEP:-50000}"
UNIK3D_ENCODER_LAST_N_BLOCKS="${UNIK3D_ENCODER_LAST_N_BLOCKS:-4}"
UNIK3D_ENCODER_FULL_UNFREEZE_STEP="${UNIK3D_ENCODER_FULL_UNFREEZE_STEP:-0}"
LOCAL_IMAGES_ROOT="${LOCAL_IMAGES_ROOT:-}"
LOCAL_IMAGES_MANIFEST="${LOCAL_IMAGES_MANIFEST:-${DATASET_MANIFEST_DIR}/local_images.txt}"
LOCAL_COLMAP_ROOT="${LOCAL_COLMAP_ROOT:-}"
LOCAL_MULTIVIEW_MANIFEST="${LOCAL_MULTIVIEW_MANIFEST:-${DATASET_MANIFEST_DIR}/local_multiview.jsonl}"
LOCAL_IMAGES_WEIGHT="${LOCAL_IMAGES_WEIGHT:-1.0}"; LOCAL_MULTIVIEW_WEIGHT="${LOCAL_MULTIVIEW_WEIGHT:-0.0}"
OPENIMAGES_PERSON_MANIFEST="${OPENIMAGES_PERSON_MANIFEST:-${DATASET_MANIFEST_DIR}/openimages_person_train_boxes.jsonl}"
OPENIMAGES_PERSON_WEIGHT="${OPENIMAGES_PERSON_WEIGHT:-0.0}"
WIDERFACE_MANIFEST="${WIDERFACE_MANIFEST:-${DATASET_MANIFEST_DIR}/widerface_images.txt}"
WIDERFACE_WEIGHT="${WIDERFACE_WEIGHT:-0.0}"
CROWDHUMAN_MANIFEST="${CROWDHUMAN_MANIFEST:-${DATASET_MANIFEST_DIR}/crowdhuman_boxes.jsonl}"
CROWDHUMAN_WEIGHT="${CROWDHUMAN_WEIGHT:-0.0}"
FFHQ_MANIFEST="${FFHQ_MANIFEST:-${DATASET_MANIFEST_DIR}/ffhq_images.txt}"
FFHQ_WEIGHT="${FFHQ_WEIGHT:-0.0}"
NEUMAN_MANIFEST="${NEUMAN_MANIFEST:-${DATASET_MANIFEST_DIR}/neuman_train.jsonl}"
NEUMAN_WEIGHT="${NEUMAN_WEIGHT:-0.0}"
NERFIES_MANIFEST="${NERFIES_MANIFEST:-${DATASET_MANIFEST_DIR}/nerfies_multiview.jsonl}"
NERFIES_WEIGHT="${NERFIES_WEIGHT:-0.0}"

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
PUBLIC_ARGS=()
add_public_dataset() {
  local manifest="$1" weight="$2" manifest_flag="$3" weight_flag="$4" label="$5"
  if [[ "${weight}" != "0" && "${weight}" != "0.0" ]]; then
    [[ -f "${manifest}" ]] || { echo "${label} manifest was not found: ${manifest}" >&2; exit 1; }
    PUBLIC_ARGS+=("${manifest_flag}" "${manifest}" "${weight_flag}" "${weight}")
  fi
}
add_public_dataset "${OPENIMAGES_PERSON_MANIFEST}" "${OPENIMAGES_PERSON_WEIGHT}" --openimages-person-manifest --dataset-weight-openimages-person "OpenImages Person"
add_public_dataset "${WIDERFACE_MANIFEST}" "${WIDERFACE_WEIGHT}" --widerface-manifest --dataset-weight-widerface "WIDER FACE"
add_public_dataset "${CROWDHUMAN_MANIFEST}" "${CROWDHUMAN_WEIGHT}" --crowdhuman-manifest --dataset-weight-crowdhuman "CrowdHuman"
add_public_dataset "${FFHQ_MANIFEST}" "${FFHQ_WEIGHT}" --ffhq-manifest --dataset-weight-ffhq "FFHQ"
add_public_dataset "${NEUMAN_MANIFEST}" "${NEUMAN_WEIGHT}" --neuman-manifest --dataset-weight-neuman "NeuMan"
add_public_dataset "${NERFIES_MANIFEST}" "${NERFIES_WEIGHT}" --nerfies-manifest --dataset-weight-nerfies "Nerfies"
[[ "${#LOCAL_ARGS[@]}" -gt 0 || "${#PUBLIC_ARGS[@]}" -gt 0 ]] || { echo "Enable at least one local or public dataset weight." >&2; exit 1; }
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
CHECK_ARGS=()
[[ "${NPU_RENDERER_BACKEND}" == "ascend_fused" ]] && CHECK_ARGS+=(--require-meta-gauss-render)
python "${SCRIPT_DIR}/check_npu_env.py" "${CHECK_ARGS[@]}"

exec torchrun --standalone --nproc_per_node="${WORLD_SIZE}" --master_port="${MASTER_PORT}" -m unisharp.cli train-feature \
  --device npu --renderer-backend "${NPU_RENDERER_BACKEND}" --out-root "${OUT_ROOT}" --run-name "${RUN_NAME}" --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" --pinhole-train-height "${PINHOLE_TRAIN_HEIGHT}" --pinhole-train-width "${PINHOLE_TRAIN_WIDTH}" --train-resize-multiple 0 \
  --unik3d-backbone "${UNIK3D_BACKBONE}" --initializer-stride "${INITIALIZER_STRIDE}" \
  "${UNIK3D_UNFREEZE_ARGS[@]}" \
  --dataset-weight-re10k 0 --dataset-weight-hm3d 0 --dataset-weight-sim 0 --dataset-weight-wildrgbd 0 --dataset-weight-dl3dv 0 --dataset-weight-scanetpp 0 \
  --lambda-percep 0 --vis-every 0 "${LOCAL_ARGS[@]}" "${PUBLIC_ARGS[@]}" "$@"
