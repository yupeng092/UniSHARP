#!/usr/bin/env bash
# Target-view-conditioned UniSHARP fine-tuning launcher.
#
# This keeps the source-image UniK3D backbone compatible with a released
# checkpoint and trains zero-initialized target-rig FiLM layers plus the normal
# Gaussian decoder.  The dataset still provides real calibrated source/target
# pairs; every target pose is encoded relative to its source pose.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:?Set INIT_CHECKPOINT to a released UniSHARP checkpoint.}"
OUT_ROOT="${OUT_ROOT:-outputs}"
RUN_NAME="${RUN_NAME:-target_rig_finetune}"
PINHOLE_TRAIN_HEIGHT="${PINHOLE_TRAIN_HEIGHT:-1024}"
PINHOLE_TRAIN_WIDTH="${PINHOLE_TRAIN_WIDTH:-1536}"
TARGET_RIG_EMBED_DIM="${TARGET_RIG_EMBED_DIM:-128}"
TARGET_RIG_TRANSLATION_SCALE="${TARGET_RIG_TRANSLATION_SCALE:-1.0}"

exec bash scripts/train_npu_portable.sh \
  --init-checkpoint "${INIT_CHECKPOINT}" \
  --no-init-checkpoint-strict \
  --out-root "${OUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --pinhole-train-height "${PINHOLE_TRAIN_HEIGHT}" \
  --pinhole-train-width "${PINHOLE_TRAIN_WIDTH}" \
  --train-resize-multiple 0 \
  --target-rig-conditioning \
  --target-rig-embedding-dim "${TARGET_RIG_EMBED_DIM}" \
  --target-rig-translation-scale "${TARGET_RIG_TRANSLATION_SCALE}" \
  "$@"
