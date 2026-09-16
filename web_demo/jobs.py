"""Safe local job storage for the UniSHARP web demonstration."""

from __future__ import annotations

import json
import re
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

    def __init__(self, config: dict[str, Any]) -> None:
        self.root = Path(config["JOB_ROOT"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_upload_bytes = int(config["MAX_UPLOAD_BYTES"])

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

    def views_from_report(self, report: dict[str, Any]) -> list[dict[str, str]]:
        """Convert a renderer report into relative, browser-safe view records."""
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
