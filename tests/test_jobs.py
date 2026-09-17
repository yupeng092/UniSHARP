from io import BytesIO
from pathlib import Path
import subprocess
from unittest.mock import patch

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


def test_view_manifest_normalizes_absolute_rgb_path_inside_render_directory(app):
    manager = JobManager(app.config)
    render_root = app.config["JOB_ROOT"] / "render"
    report = {"cameras": [{"name": "left", "rgb": str(render_root / "rgb" / "00_left.png")}]}
    assert manager.views_from_report(report, render_root=render_root) == [{"name": "left", "file": "rgb/00_left.png"}]


def test_view_manifest_rejects_absolute_rgb_path_outside_render_directory(app, tmp_path):
    manager = JobManager(app.config)
    report = {"cameras": [{"name": "left", "rgb": str(tmp_path / "outside.png")}]}
    with pytest.raises(ValueError, match="outside render directory"):
        manager.views_from_report(report, render_root=app.config["JOB_ROOT"] / "render")


def test_worker_runs_inference_render_and_gif_in_order(app):
    manager = JobManager(app.config, start_worker=False)
    job = manager.create_job("scene.png", BytesIO(png_bytes()))
    report = app.config["JOB_ROOT"] / job["id"] / "render" / "multiview_report.json"

    def completed(command, **_):
        script = Path(command[1]).name
        if script == "infer_unisharp_cpu.py":
            gaussian = app.config["JOB_ROOT"] / job["id"] / "inference" / "upload_scene" / "gaussians.pt"
            gaussian.parent.mkdir(parents=True, exist_ok=True)
            gaussian.write_bytes(b"gaussians")
        if script == "render_unisharp_cpu.py":
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text('{"cameras":[{"name":"left","rgb":"rgb/00_left.png"}]}', encoding="utf-8")
            (report.parent / "rgb").mkdir(exist_ok=True)
            (report.parent / "rgb" / "00_left.png").write_bytes(png_bytes())
        if script == "make_multiview_gif.py":
            (report.parent / "multiview.gif").write_bytes(b"GIF89a")
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("web_demo.jobs.subprocess.run", side_effect=completed) as run:
        manager.run_job(job["id"])

    assert [Path(call.args[0][1]).name for call in run.call_args_list] == [
        "infer_unisharp_cpu.py",
        "render_unisharp_cpu.py",
        "make_multiview_gif.py",
    ]
    assert manager.get_job(job["id"])["phase"] == "complete"


def test_worker_records_bounded_failure_without_absolute_path(app):
    manager = JobManager(app.config, start_worker=False)
    job = manager.create_job("scene.png", BytesIO(png_bytes()))
    failure = subprocess.CalledProcessError(1, ["infer"], stderr="bad checkpoint at C:\\private\\path")
    with patch("web_demo.jobs.subprocess.run", side_effect=failure):
        manager.run_job(job["id"])
    result = manager.get_job(job["id"])
    assert result["phase"] == "failed"
    assert "inference failed" in result["error"]
    assert "C:\\private" not in result["error"]
