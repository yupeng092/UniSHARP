#!/usr/bin/env bash
# CPU pre-training for the pinhole RE10K branch.  The portable renderer is
# differentiable but intentionally conservative; it is for correctness and
# smoke tests, not high-throughput production training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/UniK3D:${PYTHONPATH:-}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

: "${DATA_ROOT_RE10K:?Set DATA_ROOT_RE10K to the RE10K training-data root.}"
: "${DATASET_MANIFEST_DIR:?Set DATASET_MANIFEST_DIR to the UniSHARP manifests directory.}"

OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/outputs_cpu}"
RUN_NAME="${RUN_NAME:-unisharp_cpu_$(date +%Y%m%d_%H%M%S)}"
STEPS="${STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PINHOLE_TRAIN_SIZE="${PINHOLE_TRAIN_SIZE:-128}"
UNIK3D_BACKBONE="${UNIK3D_BACKBONE:-vits}"
INITIALIZER_STRIDE="${INITIALIZER_STRIDE:-2}"
PORTABLE_MAX_GAUSSIANS="${PORTABLE_MAX_GAUSSIANS:-4096}"
PORTABLE_MAX_GAUSSIANS_PER_TILE="${PORTABLE_MAX_GAUSSIANS_PER_TILE:-96}"

exec python -m unisharp.cli train-feature \
  --device cpu \
  --renderer-backend portable \
  --portable-renderer-max-gaussians "${PORTABLE_MAX_GAUSSIANS}" \
  --portable-renderer-max-gaussians-per-tile "${PORTABLE_MAX_GAUSSIANS_PER_TILE}" \
  --out-root "${OUT_ROOT}" --run-name "${RUN_NAME}" \
  --steps "${STEPS}" --batch-size "${BATCH_SIZE}" --num-workers "${NUM_WORKERS}" \
  --pinhole-train-size "${PINHOLE_TRAIN_SIZE}" --train-resize-multiple 0 \
  --unik3d-backbone "${UNIK3D_BACKBONE}" --initializer-stride "${INITIALIZER_STRIDE}" \
  --data-root-re10k "${DATA_ROOT_RE10K}" --dataset-manifest-dir "${DATASET_MANIFEST_DIR}" \
  --dataset-weight-re10k 1 --dataset-weight-hm3d 0 --dataset-weight-sim 0 \
  --dataset-weight-wildrgbd 0 --dataset-weight-dl3dv 0 --dataset-weight-scanetpp 0 \
  --lambda-percep 0 --vis-every 0 \
  "$@"
