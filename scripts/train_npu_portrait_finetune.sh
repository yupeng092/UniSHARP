#!/usr/bin/env bash
# Fine-tune the released UniSHARP ViT-L checkpoint on Ascend NPU.
# This wrapper reuses the portable NPU renderer and does not require CUDA gsplat.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FINETUNE_CHECKPOINT="${FINETUNE_CHECKPOINT:-${REPO_ROOT}/checkpoints/released/pretained_model.pt}"
[[ -f "${FINETUNE_CHECKPOINT}" ]] || {
  echo "Official UniSHARP checkpoint was not found: ${FINETUNE_CHECKPOINT}" >&2
  echo "Set FINETUNE_CHECKPOINT=/absolute/path/to/pretained_model.pt" >&2
  exit 1
}

# The released UniSHARP checkpoint was trained with UniK3D ViT-L. Do not set
# this wrapper to vits/vitb: strict checkpoint loading intentionally rejects
# architecture mismatches.
UNIK3D_BACKBONE="${UNIK3D_BACKBONE:-vitl}"
[[ "${UNIK3D_BACKBONE}" == "vitl" ]] || {
  echo "Official UniSHARP checkpoint requires UNIK3D_BACKBONE=vitl (got ${UNIK3D_BACKBONE})." >&2
  exit 1
}
export UNIK3D_BACKBONE

# The released checkpoint's config uses stride=1.  Do not silently create a
# lower-density (one-quarter Gaussian count) fine-tuned checkpoint.
INITIALIZER_STRIDE="${INITIALIZER_STRIDE:-1}"
[[ "${INITIALIZER_STRIDE}" == "1" ]] || {
  echo "Official UniSHARP checkpoint fine-tuning requires INITIALIZER_STRIDE=1 (got ${INITIALIZER_STRIDE})." >&2
  exit 1
}
export INITIALIZER_STRIDE

STEPS="${STEPS:-30000}"
UNIK3D_PROGRESSIVE_UNFREEZE="${UNIK3D_PROGRESSIVE_UNFREEZE:-1}"
UNIK3D_DECODER_UNFREEZE_STEP="${UNIK3D_DECODER_UNFREEZE_STEP:-5000}"
UNIK3D_ENCODER_UNFREEZE_STEP="${UNIK3D_ENCODER_UNFREEZE_STEP:-15000}"
UNIK3D_ENCODER_LAST_N_BLOCKS="${UNIK3D_ENCODER_LAST_N_BLOCKS:-4}"
UNIK3D_ENCODER_FULL_UNFREEZE_STEP="${UNIK3D_ENCODER_FULL_UNFREEZE_STEP:-0}"
export STEPS UNIK3D_PROGRESSIVE_UNFREEZE UNIK3D_DECODER_UNFREEZE_STEP
export UNIK3D_ENCODER_UNFREEZE_STEP UNIK3D_ENCODER_LAST_N_BLOCKS UNIK3D_ENCODER_FULL_UNFREEZE_STEP

# Fine-tuning starts with a fresh optimizer and deliberately lower learning rates.
FINETUNE_LR0="${FINETUNE_LR0:-2e-5}"
FINETUNE_LR1="${FINETUNE_LR1:-2e-6}"
FINETUNE_UNIK3D_LR0="${FINETUNE_UNIK3D_LR0:-5e-6}"
FINETUNE_UNIK3D_LR1="${FINETUNE_UNIK3D_LR1:-5e-7}"
FINETUNE_ENCODER_LR0="${FINETUNE_ENCODER_LR0:-5e-7}"
FINETUNE_ENCODER_LR1="${FINETUNE_ENCODER_LR1:-5e-8}"

exec bash "${SCRIPT_DIR}/train_npu_portrait_portable.sh" \
  --init-checkpoint "${FINETUNE_CHECKPOINT}" \
  --init-checkpoint-strict \
  --lr0 "${FINETUNE_LR0}" --lr1 "${FINETUNE_LR1}" \
  --unik3d-lr0 "${FINETUNE_UNIK3D_LR0}" --unik3d-lr1 "${FINETUNE_UNIK3D_LR1}" \
  --unik3d-encoder-lr0 "${FINETUNE_ENCODER_LR0}" --unik3d-encoder-lr1 "${FINETUNE_ENCODER_LR1}" \
  "$@"
