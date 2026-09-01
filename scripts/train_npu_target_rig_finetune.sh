#!/usr/bin/env bash
# Target-view-conditioned UniSHARP fine-tuning launcher.
#
# This keeps the source-image UniK3D backbone compatible with a released
# checkpoint and trains zero-initialized target-rig FiLM layers plus the normal
# Gaussian decoder.  The dataset still provides real calibrated source/target
# pairs; every target pose is encoded relative to its source pose. Pass the
# desired train-feature dataset arguments after this script: no dataset is
# selected implicitly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_ID="${NPU_ID:-0}"

[[ -f "${CANN_ENV}" ]] || { echo "CANN environment script was not found: ${CANN_ENV}" >&2; exit 1; }
source "${CANN_ENV}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/UniK3D:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
NPU_RENDERER_BACKEND="${NPU_RENDERER_BACKEND:-ascend_fused}"
[[ "${NPU_RENDERER_BACKEND}" == "ascend_fused" || "${NPU_RENDERER_BACKEND}" == "portable" ]] || {
  echo "NPU_RENDERER_BACKEND must be ascend_fused or portable." >&2
  exit 1
}

CHECK_ARGS=()
[[ "${NPU_RENDERER_BACKEND}" == "ascend_fused" ]] && CHECK_ARGS+=(--require-meta-gauss-render)
python "${SCRIPT_DIR}/check_npu_env.py" "${CHECK_ARGS[@]}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:?Set INIT_CHECKPOINT to a released UniSHARP checkpoint.}"
[[ -f "${INIT_CHECKPOINT}" ]] || { echo "Checkpoint was not found: ${INIT_CHECKPOINT}" >&2; exit 1; }
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs_npu}"
RUN_NAME="${RUN_NAME:-target_rig_finetune}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PINHOLE_TRAIN_HEIGHT="${PINHOLE_TRAIN_HEIGHT:-1024}"
PINHOLE_TRAIN_WIDTH="${PINHOLE_TRAIN_WIDTH:-1536}"
TARGET_RIG_EMBED_DIM="${TARGET_RIG_EMBED_DIM:-128}"
TARGET_RIG_TRANSLATION_SCALE="${TARGET_RIG_TRANSLATION_SCALE:-1.0}"
UNIK3D_BACKBONE="${UNIK3D_BACKBONE:-vitl}"
INITIALIZER_STRIDE="${INITIALIZER_STRIDE:-1}"
[[ "${INITIALIZER_STRIDE}" == "1" ]] || {
  echo "Official UniSHARP target-rig fine-tuning requires INITIALIZER_STRIDE=1 (got ${INITIALIZER_STRIDE})." >&2
  exit 1
}
PORTABLE_RENDERER_ARGS=()
if [[ "${NPU_RENDERER_BACKEND}" == "portable" ]]; then
  : "${MAX_GAUSSIANS:?Set MAX_GAUSSIANS when NPU_RENDERER_BACKEND=portable (for example 65536).}"
  PORTABLE_RENDERER_ARGS=(--portable-renderer-max-gaussians "${MAX_GAUSSIANS}")
fi

exec python -m unisharp.cli train-feature \
  --device npu --renderer-backend "${NPU_RENDERER_BACKEND}" \
  "${PORTABLE_RENDERER_ARGS[@]}" \
  --out-root "${OUT_ROOT}" --run-name "${RUN_NAME}" \
  --steps "${STEPS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --no-init-checkpoint-strict \
  --pinhole-train-height "${PINHOLE_TRAIN_HEIGHT}" \
  --pinhole-train-width "${PINHOLE_TRAIN_WIDTH}" \
  --train-resize-multiple 0 \
  --unik3d-backbone "${UNIK3D_BACKBONE}" --initializer-stride "${INITIALIZER_STRIDE}" \
  --target-rig-conditioning \
  --target-rig-embedding-dim "${TARGET_RIG_EMBED_DIM}" \
  --target-rig-translation-scale "${TARGET_RIG_TRANSLATION_SCALE}" \
  --lambda-percep 0 --vis-every 0 \
  --dataset-weight-re10k 0 --dataset-weight-hm3d 0 --dataset-weight-sim 0 \
  --dataset-weight-wildrgbd 0 --dataset-weight-dl3dv 0 --dataset-weight-scanetpp 0 \
  --dataset-weight-coco-person 0 --dataset-weight-widerface 0 --dataset-weight-openimages-person 0 \
  --dataset-weight-crowdhuman 0 --dataset-weight-ffhq 0 --dataset-weight-neuman 0 \
  --dataset-weight-nerfies 0 --dataset-weight-local-images 0 --dataset-weight-local-multiview 0 \
  "$@"
