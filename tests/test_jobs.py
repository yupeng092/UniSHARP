from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from web_demo.jobs import JobManager, UploadValidationError


def png_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), "white")
    result = BytesIO()
    image.save(result, "PNG")
    return result.getvalue()


def test_create_job_rejects_a_non_image(app):
    manager = JobManager(app.config)
    with pytest.raises(UploadValidationError, match="valid image"):
        manager.create_job("scene.jpg", BytesIO(b"not-an-image"))


def test_create_job_writes_upload_inside_its_job_directory(app):
    manager = JobManager(app.config)
    job = manager.create_job("../scene.png", BytesIO(png_bytes()))
    assert job["phase"] == "queued"
    assert Path(job["upload_relative_path"]).name == "scene.png"
    assert (app.config["JOB_ROOT"] / job["id"] / job["upload_relative_path"]).is_file()


def test_safe_job_file_rejects_traversal(app):
    manager = JobManager(app.config)
    job = manager.create_job("scene.png", BytesIO(png_bytes()))
    with pytest.raises(ValueError, match="outside job"):
        manager.safe_job_file(job["id"], "../../README.md")


def test_set_phase_records_error_for_failed_job(app):
    manager = JobManager(app.config)
    job = manager.create_job("scene.png", BytesIO(png_bytes()))
    updated = manager.set_phase(job["id"], "failed", error="renderer exited")
    assert updated["phase"] == "failed"
    assert updated["error"] == "renderer exited"


def test_view_manifest_uses_report_camera_names_and_rgb_paths(app):
    manager = JobManager(app.config)
    report = {"cameras": [{"name": "left", "rgb": "rgb/00_left.png"}]}
    assert manager.views_from_report(report) == [{"name": "left", "file": "rgb/00_left.png"}]
