"""HTTP routes for the local UniSHARP demonstration."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, abort, current_app, jsonify, render_template, request, send_file, url_for

from .jobs import JobManager, UploadValidationError


bp = Blueprint("web_demo", __name__)


def _manager() -> JobManager:
    return current_app.extensions["job_manager"]


def _job_or_404(job_id: str) -> dict[str, Any]:
    try:
        return _manager().get_job(job_id)
    except (FileNotFoundError, ValueError):
        abort(404)


def _public_job(job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result")
    public: dict[str, Any] = {"id": job["id"], "phase": job["phase"], "error": job.get("error"), "result": None}
    if not isinstance(result, dict):
        return public
    views = result.get("views", [])
    public["result"] = {
        "source": url_for("web_demo.job_file", job_id=job["id"], relative_path=result["source"]),
        "gif": url_for("web_demo.job_file", job_id=job["id"], relative_path=result["gif"]),
        "views": [
            {
                "name": view["name"],
                "url": url_for("web_demo.job_file", job_id=job["id"], relative_path=f"render/{view['file']}"),
            }
            for view in views
        ],
    }
    return public


def _allowed_files(job: dict[str, Any]) -> set[str]:
    allowed = {str(job["upload_relative_path"])}
    result = job.get("result")
    if isinstance(result, dict):
        allowed.add(str(result["gif"]))
        allowed.update(f"render/{view['file']}" for view in result.get("views", []))
    return allowed


@bp.get("/")
def index():
    """Serve the local image-upload entry point."""
    return render_template("index.html")


@bp.post("/api/jobs")
def create_job():
    """Validate one multipart image then queue its CPU rendering work."""
    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return jsonify({"error": "choose one image before generating views"}), 400
    try:
        job = _manager().create_job(uploaded.filename, uploaded.stream)
        _manager().enqueue(job["id"])
    except UploadValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"id": job["id"], "phase": job["phase"], "job_url": url_for("web_demo.job_page", job_id=job["id"])}), 201


@bp.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    """Return only browser-safe job status and generated result URLs."""
    return jsonify(_public_job(_job_or_404(job_id)))


@bp.get("/jobs/<job_id>")
def job_page(job_id: str):
    """Serve the job result page shell after validating its ID."""
    _job_or_404(job_id)
    return render_template("job.html", job_id=job_id)


@bp.get("/job-files/<job_id>/<path:relative_path>")
def job_file(job_id: str, relative_path: str):
    """Serve only source and completed renderer files from a contained job."""
    job = _job_or_404(job_id)
    if relative_path not in _allowed_files(job):
        abort(404)
    try:
        path = _manager().safe_job_file(job_id, relative_path)
    except ValueError:
        abort(404)
    if not path.is_file():
        abort(404)
    return send_file(path)
