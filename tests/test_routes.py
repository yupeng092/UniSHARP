from io import BytesIO

from PIL import Image


def png_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), "white")
    result = BytesIO()
    image.save(result, "PNG")
    return result.getvalue()


def test_post_job_returns_created_job_url(client, monkeypatch):
    monkeypatch.setattr("web_demo.routes.JobManager.enqueue", lambda *_: None)
    response = client.post("/api/jobs", data={"image": (BytesIO(png_bytes()), "room.png")})
    assert response.status_code == 201
    assert response.json["phase"] == "queued"
    assert response.json["job_url"].startswith("/jobs/")


def test_status_does_not_expose_host_paths(client, app):
    manager = app.extensions["job_manager"]
    job = manager.create_job("room.png", BytesIO(png_bytes()))
    response = client.get(f"/api/jobs/{job['id']}")
    assert response.status_code == 200
    assert "C:\\" not in response.get_data(as_text=True)


def test_job_files_reject_parent_path(client, app):
    manager = app.extensions["job_manager"]
    job = manager.create_job("room.png", BytesIO(png_bytes()))
    assert client.get(f"/job-files/{job['id']}/../../README.md").status_code == 404


def test_home_page_contains_upload_control(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b'name="image"' in response.data
    assert b"Generate views" in response.data


def test_job_page_contains_status_and_navigator_shell(client, app):
    manager = app.extensions["job_manager"]
    job = manager.create_job("room.png", BytesIO(png_bytes()))
    response = client.get(f"/jobs/{job['id']}")
    assert response.status_code == 200
    assert b'id="job-status"' in response.data
    assert b'id="view-navigator"' in response.data
