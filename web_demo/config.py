"""Server-only settings for the local UniSHARP web demonstration."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "REPO_ROOT": REPO_ROOT,
    "JOB_ROOT": REPO_ROOT / "outputs" / "web_demo",
    "PYTHON_EXECUTABLE": Path(sys.executable),
    "CHECKPOINT_PATH": REPO_ROOT / "checkpoints" / "released" / "pretained_model.pt",
    "CAMERA_RIG_PATH": REPO_ROOT / "configs" / "camera_rig_small10.json",
    "BIND_HOST": "127.0.0.1",
    "PORT": 5000,
    "MAX_UPLOAD_BYTES": 15 * 1024 * 1024,
    "MAX_LONG_EDGE": 768,
    "RENDER_WIDTH": 768,
    "RENDER_HEIGHT": 512,
    "THREADS": 8,
}
