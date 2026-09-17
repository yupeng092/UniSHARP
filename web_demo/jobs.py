"""Safe local job storage for the UniSHARP web demonstration."""

from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename


_JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
_NEXT_PHASES = {
    "queued": {"inference", "failed"},
    "inference": {"rendering", "failed"},
    "rendering": {"complete", "failed"},
    "complete": set(),
    "failed": set(),
}


class UploadValidationError(ValueError):
    """Raised when a browser upload does not meet the image policy."""


class JobManager:
    """Persist browser jobs below one configured root without path escape."""

    def __init__(self, config: dict[str, Any], *, start_worker: bool = True) -> None:
        self.root = Path(config["JOB_ROOT"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = int(config["MAX_UPLOAD_BYTES"])
        self.repo_root = Path(config["REPO_ROOT"]).resolve()
        self.python = Path(config["PYTHON_EXECUTABLE"])
        self.checkpoint = Path(config["CHECKPOINT_PATH"])
        self.camera_rig = Path(config["CAMERA_RIG_PATH"])
        self.max_long_edge = int(config["MAX_LONG_EDGE"])
        self.render_width = int(config["RENDER_WIDTH"])
        self.render_height = int(config["RENDER_HEIGHT"])
        self.threads = int(config["THREADS"])
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker: threading.Thread | None = None
        if start_worker:
            self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="unisharp-web-demo")
            self._worker.start()

    def create_job(self, original_name: str, stream: BinaryIO) -> dict[str, Any]:
        """Validate and store an uploaded image in a newly created job directory."""
        name = secure_filename(original_name)
        if not name or Path(name).suffix.lower() not in _IMAGE_SUFFIXES:
            raise UploadValidationError("upload must be a JPG, PNG, or WebP image")
        job_id = uuid.uuid4().hex
        job_dir = self._job_dir(job_id)
        upload_dir = job_dir / "upload"
        upload_dir.mkdir(parents=True)
        upload_path = upload_dir / name
        remaining = self.max_upload_bytes
        with upload_path.open("wb") as destination:
            while chunk := stream.read(min(1024 * 1024, remaining + 1)):
                remaining -= len(chunk)
                if remaining < 0:
                    destination.close()
                    upload_path.unlink(missing_ok=True)
                    raise UploadValidationError("image is too large")
                destination.write(chunk)
        try:
            with Image.open(upload_path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            upload_path.unlink(missing_ok=True)
            raise UploadValidationError("upload must contain a valid image") from exc
        job = {
            "id": job_id,
            "phase": "queued",
            "created_at": self._timestamp(),
            "updated_at": self._timestamp(),
            "upload_relative_path": f"upload/{name}",
            "error": None,
            "result": None,
        }
        self._write_job(job_dir, job)
        return job

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Load one persisted job after validating the opaque browser ID."""
        job_path = self._job_dir(job_id) / "job.json"
        if not job_path.is_file():
            raise FileNotFoundError("job does not exist")
        value = json.loads(job_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("id") != job_id:
            raise ValueError("invalid job data")
        return value

    def safe_job_file(self, job_id: str, relative_path: str) -> Path:
        """Resolve a result path only when it remains inside its job directory."""
        job_dir = self._job_dir(job_id).resolve()
        candidate = (job_dir / relative_path).resolve()
        if not candidate.is_relative_to(job_dir):
            raise ValueError("requested file is outside job directory")
        return candidate

    def set_phase(self, job_id: str, phase: str, *, error: str | None = None) -> dict[str, Any]:
        """Perform one legal phase transition and durably persist it."""
        job = self.get_job(job_id)
        current = str(job["phase"])
        if phase not in _NEXT_PHASES.get(current, set()):
            raise ValueError(f"invalid job transition: {current} to {phase}")
        job["phase"] = phase
        job["updated_at"] = self._timestamp()
        job["error"] = error if phase == "failed" else None
        self._write_job(self._job_dir(job_id), job)
        return job

    def enqueue(self, job_id: str) -> None:
        """Place one validated queued job behind any currently running job."""
        if self.get_job(job_id)["phase"] != "queued":
            raise ValueError("only queued jobs can be enqueued")
        self._queue.put(job_id)

    def run_job(self, job_id: str) -> None:
        """Run inference, rig render, and GIF creation in a single worker."""
        job_dir = self._job_dir(job_id)
        upload = self.safe_job_file(job_id, self.get_job(job_id)["upload_relative_path"])
        inference_root = job_dir / "inference"
        render_root = job_dir / "render"
        try:
            self.set_phase(job_id, "inference")
            self._run("inference", [
                self.python, self.repo_root / "scripts" / "infer_unisharp_cpu.py",
                "--checkpoint", self.checkpoint,
                "--image", upload,
                "--out-dir", inference_root,
                "--max-long-edge", str(self.max_long_edge),
                "--threads", str(self.threads),
            ])
            gaussian_paths = list(inference_root.glob("*/gaussians.pt"))
            if len(gaussian_paths) != 1:
                raise RuntimeError("inference did not produce exactly one Gaussian export")

            self.set_phase(job_id, "rendering")
            self._run("rendering", [
                self.python, self.repo_root / "scripts" / "render_unisharp_cpu.py",
                "--gaussians", gaussian_paths[0], "--output", render_root,
                "--trajectory", "rig", "--camera-file", self.camera_rig,
                "--camera-orientation", "look_at", "--backend", "torch",
                "--height", str(self.render_height), "--width", str(self.render_width),
                "--threads", str(self.threads), "--no-save-gaussians",
            ])
            report_path = render_root / "multiview_report.json"
            report = self._read_report(report_path)
            views = self.views_from_report(report, render_root=render_root)
            for view in views:
                if not self.safe_job_file(job_id, f"render/{view['file']}").is_file():
                    raise RuntimeError("renderer did not produce every RGB view")

            gif_path = render_root / "multiview.gif"
            self._run("GIF encoding", [
                self.python, self.repo_root / "scripts" / "make_multiview_gif.py",
                "--render-report", report_path, "--source-image", upload,
                "--camera-rig", self.camera_rig, "--output", gif_path, "--ping-pong",
            ])
            if not gif_path.is_file():
                raise RuntimeError("GIF encoder did not produce an animation")
            job = self.get_job(job_id)
            job.update({
                "phase": "complete", "updated_at": self._timestamp(), "error": None,
                "result": {"views": views, "gif": "render/multiview.gif", "source": job["upload_relative_path"]},
            })
            self._write_job(job_dir, job)
        except subprocess.CalledProcessError as exc:
            self._fail_if_active(job_id, f"{self.get_job(job_id)['phase']} failed (exit code {exc.returncode})")
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            self._fail_if_active(job_id, "rendering job failed; inspect local job files for details")

    def _worker_loop(self) -> None:
        while True:
            self.run_job(self._queue.get())
            self._queue.task_done()

    def _run(self, stage: str, command: list[str | Path]) -> None:
        subprocess.run(
            [str(value) for value in command], cwd=self.repo_root, check=True,
            capture_output=True, text=True,
        )

    @staticmethod
    def _read_report(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("render report is not an object")
        return value

    def _fail_if_active(self, job_id: str, message: str) -> None:
        job = self.get_job(job_id)
        if job["phase"] not in {"complete", "failed"}:
            self.set_phase(job_id, "failed", error=message[:240])

    def views_from_report(self, report: dict[str, Any], *, render_root: Path | None = None) -> list[dict[str, str]]:
        """Convert report RGB paths to browser-safe paths relative to ``render_root``."""
        cameras = report.get("cameras")
        if not isinstance(cameras, list) or not cameras:
            raise ValueError("render report does not contain cameras")
        views: list[dict[str, str]] = []
        for camera in cameras:
            if not isinstance(camera, dict):
                raise ValueError("render report contains an invalid camera")
            name, rgb = camera.get("name"), camera.get("rgb")
            if not isinstance(name, str) or not name or not isinstance(rgb, str) or not rgb:
                raise ValueError("render report camera is missing name or RGB path")
            source_path = Path(rgb)
            if source_path.is_absolute():
                if render_root is None:
                    raise ValueError("absolute render report RGB path needs a render directory")
                resolved_root = Path(render_root).resolve()
                resolved_path = source_path.resolve()
                if not resolved_path.is_relative_to(resolved_root):
                    raise ValueError("render report RGB path is outside render directory")
                path = PurePosixPath(resolved_path.relative_to(resolved_root).as_posix())
            else:
                path = PurePosixPath(rgb.replace("\\", "/"))
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("render report RGB path must be relative")
            views.append({"name": name, "file": path.as_posix()})
        return views

    def _job_dir(self, job_id: str) -> Path:
        if not _JOB_ID.fullmatch(job_id):
            raise ValueError("invalid job ID")
        return self.root / job_id

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _write_job(job_dir: Path, job: dict[str, Any]) -> None:
        temporary = job_dir / "job.json.tmp"
        temporary.write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(job_dir / "job.json")
