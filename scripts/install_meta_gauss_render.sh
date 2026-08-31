#!/usr/bin/env bash
# Build/install CANN's fused 3DGS operators for UniSHARP's ascend_fused backend.
#
# Download the CANN recipe snapshot on a machine with internet access, upload
# it to the NPU server, then pass its extracted root with --source.  This
# avoids requiring git on the server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_ROOT=""
WHEEL=""
INSTALL_BUILD_DEPS=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/install_meta_gauss_render.sh --source /path/to/cann-recipes-embodied-ai
  bash scripts/install_meta_gauss_render.sh --wheel /path/to/meta_gauss_render-*.whl

The source tree must contain ops/ascendc/build.sh.  Download the source ZIP
locally from https://gitcode.com/cann/cann-recipes-embodied-ai, upload and
extract it on the NPU server.  CANN must already be initialised in this shell.

Options:
  --python PATH              Python executable (default: python)
  --source DIR               Extracted CANN recipes repository; builds its wheel
  --wheel PATH               Existing meta_gauss_render wheel; installs it directly
  --install-build-deps       Install CANN recipe build-time Python dependencies
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --source) SOURCE_ROOT="$2"; shift 2 ;;
    --wheel) WHEEL="$2"; shift 2 ;;
    --install-build-deps) INSTALL_BUILD_DEPS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "${PYTHON_BIN}" || -n "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]] || {
  echo "Python executable not found: ${PYTHON_BIN}" >&2; exit 1;
}
[[ -n "${SOURCE_ROOT}" || -n "${WHEEL}" ]] || { usage >&2; exit 2; }
[[ -z "${SOURCE_ROOT}" || -z "${WHEEL}" ]] || { echo "Use --source or --wheel, not both." >&2; exit 2; }

if [[ -n "${WHEEL}" ]]; then
  [[ -f "${WHEEL}" ]] || { echo "Wheel not found: ${WHEEL}" >&2; exit 1; }
  "${PYTHON_BIN}" -m pip install --force-reinstall "${WHEEL}"
else
  ASCENDC_DIR="${SOURCE_ROOT}/ops/ascendc"
  [[ -f "${ASCENDC_DIR}/build.sh" ]] || {
    echo "Expected ${SOURCE_ROOT}/ops/ascendc/build.sh. This is not the CANN embodied-ai recipe root." >&2
    exit 1
  }
  if [[ "${INSTALL_BUILD_DEPS}" == "1" ]]; then
    "${PYTHON_BIN}" -m pip install numpy==1.23 decorator sympy scipy attrs cloudpickle psutil synr==0.5.0 tornado cmake pyyaml expecttest protobuf
  fi
  (
    cd "${ASCENDC_DIR}"
    bash build.sh --python="$(${PYTHON_BIN} -c 'import sys; print(sys.version_info.major.__str__()+"."+sys.version_info.minor.__str__())')"
  )
  mapfile -t WHEELS < <(find "${ASCENDC_DIR}/dist" -maxdepth 1 -type f -name 'meta_gauss_render-*.whl' | sort)
  [[ "${#WHEELS[@]}" -gt 0 ]] || { echo "Build completed without a meta_gauss_render wheel." >&2; exit 1; }
  "${PYTHON_BIN}" -m pip install --force-reinstall "${WHEELS[-1]}"
fi

"${PYTHON_BIN}" - <<'PY'
import meta_gauss_render
required = (
    "projection_three_dims_gaussian_fused",
    "flash_gaussian_build_mask",
    "gaussian_sort",
    "calc_render",
    "get_render_schedule",
)
missing = [name for name in required if not hasattr(meta_gauss_render, name)]
if missing:
    raise SystemExit("Installed meta_gauss_render is incomplete: " + ", ".join(missing))
print("meta_gauss_render installed:", meta_gauss_render.__file__)
PY

echo "Ready. Start UniSHARP NPU training with the default NPU_RENDERER_BACKEND=ascend_fused."
