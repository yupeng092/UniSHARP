#!/usr/bin/env bash
# Low-resolution CPU correctness run for target-rig-conditioned fine-tuning.
# Pass the dataset flags accepted by `python -m unisharp.cli train-feature`
# after this script. No dataset (including RE10K) is selected implicitly.
# It validates calibrated-pair loading, forward/backward and checkpoint output;
# use train_npu_target_rig_finetune.sh for an actual fine-tuning job.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/UniK3D:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:?Set INIT_CHECKPOINT to a released UniSHARP checkpoint.}"
[[ -f "${INIT_CHECKPOINT}" ]] || { echo "Checkpoint was not found: ${INIT_CHECKPOINT}" >&2; exit 1; }

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs_cpu}"
RUN_NAME="${RUN_NAME:-target_rig_cpu_smoketest_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-10}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PINHOLE_TRAIN_HEIGHT="${PINHOLE_TRAIN_HEIGHT:-256}"
PINHOLE_TRAIN_WIDTH="${PINHOLE_TRAIN_WIDTH:-384}"
TARGET_RIG_EMBED_DIM="${TARGET_RIG_EMBED_DIM:-128}"
TARGET_RIG_TRANSLATION_SCALE="${TARGET_RIG_TRANSLATION_SCALE:-1.0}"
MAX_GAUSSIANS="${MAX_GAUSSIANS:?Set MAX_GAUSSIANS to the explicit CPU portable-renderer Gaussian cap (for example 16384).}"
MAX_GAUSSIANS_PER_TILE="${MAX_GAUSSIANS_PER_TILE:-96}"

exec python -m unisharp.cli train-feature \
  --device cpu --renderer-backend portable \
  --portable-renderer-max-gaussians "${MAX_GAUSSIANS}" --portable-renderer-max-gaussians-per-tile "${MAX_GAUSSIANS_PER_TILE}" \
  --out-root "${OUT_ROOT}" --run-name "${RUN_NAME}" \
  --steps "${STEPS}" --batch-size 1 --num-workers "${NUM_WORKERS}" \
  --pinhole-train-height "${PINHOLE_TRAIN_HEIGHT}" --pinhole-train-width "${PINHOLE_TRAIN_WIDTH}" --train-resize-multiple 0 \
  --unik3d-backbone vitl --initializer-stride 1 \
  --init-checkpoint "${INIT_CHECKPOINT}" --no-init-checkpoint-strict \
  --target-rig-conditioning --target-rig-embedding-dim "${TARGET_RIG_EMBED_DIM}" --target-rig-translation-scale "${TARGET_RIG_TRANSLATION_SCALE}" \
  --save-every "${STEPS}" --log-every 1 --vis-every 0 --lambda-percep 0 \
  --dataset-weight-re10k 0 --dataset-weight-hm3d 0 --dataset-weight-sim 0 \
  --dataset-weight-wildrgbd 0 --dataset-weight-dl3dv 0 --dataset-weight-scanetpp 0 \
  --dataset-weight-coco-person 0 --dataset-weight-widerface 0 --dataset-weight-openimages-person 0 \
  --dataset-weight-crowdhuman 0 --dataset-weight-ffhq 0 --dataset-weight-neuman 0 \
  --dataset-weight-nerfies 0 --dataset-weight-local-images 0 --dataset-weight-local-multiview 0 \
  "$@"
