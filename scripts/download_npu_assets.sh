#!/usr/bin/env bash
# Download public pretrained assets on the Ascend NPU host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/UniK3D:${PYTHONPATH:-}"
exec python "${SCRIPT_DIR}/download_npu_assets.py" "$@"
