# UniSHARP Local Web Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a localhost-only Flask demo where one uploaded image is processed by the existing UniSHARP CPU scripts and its ten rendered views can be selected by thumbnail or drag navigation.

**Architecture:** A Flask application owns safe job directories and a single-worker queue. Its worker invokes the three existing CLI scripts, derives browser-safe results from `multiview_report.json`, and persists state in `job.json`. Static HTML/CSS/JavaScript implements upload, polling, thumbnails, and a finite-view drag navigator; it never claims continuous browser-side 3D rendering.

**Tech Stack:** Python 3.13, Flask, Pillow, pytest, Node.js built-in `node:test`, vanilla HTML/CSS/JavaScript.

---

## Planned file structure

```text
web_demo/
├── __init__.py                 # application factory
├── config.py                   # immutable server-only paths and limits
├── jobs.py                     # validation, state persistence, queue worker, CLI orchestration
├── routes.py                   # Flask routes and safe file serving
├── templates/
│   ├── index.html              # upload page
│   └── job.html                # status/result page shell
└── static/
    ├── app.css                 # responsive visual design
    ├── upload.js               # multipart upload and redirect
    ├── job.js                  # status polling and result rendering
    └── view_navigator.js       # pure pointer-delta → named-view selection
tests/
├── conftest.py                 # temporary app/job-root fixture
├── test_jobs.py                # validation, paths, transitions, reports
├── test_routes.py              # upload/status/file endpoint tests
└── web/
    └── view_navigator.test.mjs # browser navigation logic under node:test
scripts/
└── run_web_demo.py             # local server entry point
```

### Task 1: Add web-demo dependencies and server-only configuration

**Files:**
- Modify: `requirements.txt`
- Create: `web_demo/__init__.py`
- Create: `web_demo/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_app.py`

- [ ] **Step 1: Write the failing application-factory test**

```python
# tests/conftest.py
from pathlib import Path
import pytest

from web_demo import create_app

@pytest.fixture()
def app(tmp_path: Path):
    app = create_app({
        "TESTING": True,
        "JOB_ROOT": tmp_path / "jobs",
        "REPO_ROOT": tmp_path / "repo",
        "MAX_UPLOAD_BYTES": 1024 * 1024,
    })
    return app

@pytest.fixture()
def client(app):
    return app.test_client()
```

Create `tests/test_app.py` with:

```python
def test_factory_defaults_to_loopback_host(app):
    assert app.config["BIND_HOST"] == "127.0.0.1"
    assert app.config["JOB_ROOT"].name == "jobs"
```

- [ ] **Step 2: Verify that the test fails because `web_demo` is missing**

Run: `python -m pytest tests/test_app.py -q`

Expected: import failure for `web_demo`.

- [ ] **Step 3: Add Flask and implement the minimal configuration and factory**

Append `Flask>=3.1,<4` to `requirements.txt`. Create `web_demo/config.py` with:

```python
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
```

Create `web_demo/__init__.py` with a `create_app(overrides=None)` factory that
loads `DEFAULTS`, applies overrides, creates `JOB_ROOT`, sets Flask
`MAX_CONTENT_LENGTH`, creates one `JobManager`, and registers the routes
blueprint. Do not read Python/checkpoint/rig paths from the request.

- [ ] **Step 4: Verify the factory test passes**

Run: `python -m pytest tests/test_app.py -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit the tested foundation**

```powershell
git add requirements.txt web_demo/__init__.py web_demo/config.py tests/conftest.py tests/test_app.py
git commit -m "Add web demo application factory"
```

### Task 2: Build test-first job validation, containment, and durable state

**Files:**
- Create: `web_demo/jobs.py`
- Create: `tests/test_jobs.py`

- [ ] **Step 1: Write failing tests for upload validation and containment**

```python
from io import BytesIO
from pathlib import Path
import pytest
from PIL import Image

from web_demo.jobs import JobManager, UploadValidationError

def png_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), "white")
    result = BytesIO(); image.save(result, "PNG")
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
```

- [ ] **Step 2: Verify the tests fail because the job module is missing**

Run: `python -m pytest tests/test_jobs.py -q`

Expected: import failure for `web_demo.jobs`.

- [ ] **Step 3: Implement the minimal job model**

Implement `JobManager` with four public methods: `create_job(original_name,
stream)`, which returns the new JSON job object; `get_job(job_id)`, which
returns a persisted object after ID validation; `safe_job_file(job_id,
relative_path)`, which returns a contained `Path`; and `set_phase(job_id,
phase, *, error=None)`, which persists and returns the transitioned object.

Generate an ID with `uuid.uuid4().hex`; allow only `[0-9a-f]{32}`. Use
`werkzeug.utils.secure_filename`, require extensions `{.jpg, .jpeg, .png,
.webp}`, verify with `PIL.Image.open(upload_path).verify()`, and enforce
`MAX_UPLOAD_BYTES` while copying the stream. Write a compact JSON object to
`job.json` via a same-directory temporary file followed by `Path.replace()`.
`safe_job_file` must resolve both the job root and candidate then require the
candidate to be within the resolved job root with `candidate.is_relative_to`.

- [ ] **Step 4: Add phase/report tests before implementing those paths**

```python
def test_set_phase_records_error_only_for_failed_job(app):
    manager = JobManager(app.config)
    job = manager.create_job("scene.png", BytesIO(png_bytes()))
    updated = manager.set_phase(job["id"], "failed", error="renderer exited")
    assert updated["phase"] == "failed"
    assert updated["error"] == "renderer exited"

def test_view_manifest_uses_only_report_camera_names_and_rgb_paths(app, tmp_path):
    manager = JobManager(app.config)
    report = {"cameras": [{"name": "left", "rgb": "rgb/00_left.png"}]}
    assert manager.views_from_report(report) == [{"name": "left", "file": "rgb/00_left.png"}]
```

- [ ] **Step 5: Verify the new tests fail for missing transition/report behavior**

Run: `python -m pytest tests/test_jobs.py -q`

Expected: failures naming `set_phase` and `views_from_report`.

- [ ] **Step 6: Implement strict transitions and report parsing**

Allow only `queued → inference → rendering → complete` and any non-terminal
phase to `failed`; reject other transitions. Implement
`views_from_report(report)` to accept a non-empty `cameras` list, retain only
non-empty string `name` and `rgb` values, confirm each RGB path is a relative
POSIX-like path without `..`, and return `[{"name": name, "file": rgb}]` in
report order. Add result data to complete jobs with exactly three fields:
`views` (the parsed list), `gif` (the string `render/multiview.gif`), and
`source` (the upload-relative path).

- [ ] **Step 7: Run the complete job-module suite**

Run: `python -m pytest tests/test_jobs.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit job safety and state management**

```powershell
git add web_demo/jobs.py tests/test_jobs.py
git commit -m "Add safe web demo job management"
```

### Task 3: Add a single-process rendering worker that reuses existing CLIs

**Files:**
- Modify: `web_demo/jobs.py`
- Modify: `tests/test_jobs.py`

- [ ] **Step 1: Write failing process-orchestration tests**

```python
from unittest.mock import patch

def test_worker_runs_inference_render_and_gif_in_order(app):
    manager = JobManager(app.config, start_worker=False)
    job = manager.create_job("scene.png", BytesIO(png_bytes()))
    report = app.config["JOB_ROOT"] / job["id"] / "render" / "multiview_report.json"

    def completed(command, **_):
        if "render_unisharp_cpu.py" in map(str, command):
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text('{"cameras":[{"name":"left","rgb":"rgb/00_left.png"}]}')
            (report.parent / "rgb").mkdir(exist_ok=True)
            (report.parent / "rgb" / "00_left.png").write_bytes(png_bytes())
        if "make_multiview_gif.py" in map(str, command):
            (report.parent / "multiview.gif").write_bytes(b"GIF89a")
        return subprocess.CompletedProcess(command, 0, "", "")

    with patch("web_demo.jobs.subprocess.run", side_effect=completed) as run:
        manager.run_job(job["id"])

    assert [Path(call.args[0][1]).name for call in run.call_args_list] == [
        "infer_unisharp_cpu.py", "render_unisharp_cpu.py", "make_multiview_gif.py"
    ]
    assert manager.get_job(job["id"])["phase"] == "complete"
```

- [ ] **Step 2: Verify it fails because `run_job` does not exist**

Run: `python -m pytest tests/test_jobs.py::test_worker_runs_inference_render_and_gif_in_order -q`

Expected: failure naming `run_job`.

- [ ] **Step 3: Implement process construction and one-at-a-time worker**

Give `JobManager` a `queue.Queue[str]`, a `threading.Lock`, and one daemon
thread started unless `start_worker=False`. `enqueue(job_id)` appends the ID;
the worker calls `run_job(job_id)` serially. Build commands as argument lists,
with `cwd=REPO_ROOT`, `check=True`, `capture_output=True`, `text=True`:

```python
[python, "scripts/infer_unisharp_cpu.py", "--checkpoint", checkpoint,
 "--image", upload, "--out-dir", inference_root, "--max-long-edge", "768",
 "--threads", "8"]
[python, "scripts/render_unisharp_cpu.py", "--gaussians", gaussian_path,
 "--output", render_root, "--trajectory", "rig", "--camera-file", rig,
 "--camera-orientation", "look_at", "--backend", "torch", "--height", "512",
 "--width", "768", "--threads", "8", "--no-save-gaussians"]
[python, "scripts/make_multiview_gif.py", "--render-report", report_path,
 "--source-image", upload, "--camera-rig", rig, "--output", gif_path,
 "--ping-pong"]
```

Locate the inferred Gaussian by reading the one created slug directory under
`inference/`, not by trusting the browser filename. Before completion require
the report, every selected RGB file, and the GIF to exist. On
`CalledProcessError`, malformed report, or missing output, record only a
bounded last stderr line in `job.json` and set `failed`.

- [ ] **Step 4: Verify worker behavior and error state**

Add a test whose patched inference process raises
`subprocess.CalledProcessError(1, ["infer"], stderr="bad checkpoint")` and
assert the phase is `failed` and the error contains `inference failed` but not
an absolute path. Then run:

`python -m pytest tests/test_jobs.py -q`

Expected: all job tests pass.

- [ ] **Step 5: Commit the worker**

```powershell
git add web_demo/jobs.py tests/test_jobs.py
git commit -m "Run UniSHARP jobs from web demo queue"
```

### Task 4: Implement and test the HTTP API and safe result serving

**Files:**
- Create: `web_demo/routes.py`
- Modify: `web_demo/__init__.py`
- Modify: `tests/test_routes.py`

- [ ] **Step 1: Write failing API tests**

```python
from io import BytesIO

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
```

- [ ] **Step 2: Verify the route tests fail**

Run: `python -m pytest tests/test_routes.py -q`

Expected: 404/missing endpoint failures.

- [ ] **Step 3: Implement minimal routes**

Use a Flask blueprint with four endpoint functions: a POST handler at
`/api/jobs` named `create_job`; a GET handler at `/api/jobs/<job_id>` named
`job_status`; a GET handler at `/jobs/<job_id>` named `job_page`; and a GET
handler at `/job-files/<job_id>/<path:relative_path>` named `job_file`.

`create_job` requires the `image` form part, translates
`UploadValidationError` to a JSON 400, calls `enqueue`, and returns a 201 JSON
object with ID, phase, and a job-page URL produced with
`url_for("web_demo.job_page", job_id=job_id)`.
`job_status` returns a public projection of `job.json`, mapping result relative
paths to `/job-files/<job_id>/<relative_path>` URLs. `job_file` calls `safe_job_file`, uses
`send_from_directory`, and returns 404 for a missing/disallowed file. Register
an explicit `RequestEntityTooLarge` handler that returns `{error: "image is too
large"}` with 413.

- [ ] **Step 4: Verify all API tests pass**

Run: `python -m pytest tests/test_routes.py tests/test_jobs.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit the HTTP boundary**

```powershell
git add web_demo/routes.py web_demo/__init__.py tests/test_routes.py
git commit -m "Expose UniSHARP web demo API"
```

### Task 5: Implement and test finite-view drag navigation

**Files:**
- Create: `web_demo/static/view_navigator.js`
- Create: `tests/web/view_navigator.test.mjs`

- [ ] **Step 1: Write failing Node tests for drag classification**

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { viewForDrag } from "../../web_demo/static/view_navigator.js";

test("horizontal drag chooses the matching side", () => {
  assert.equal(viewForDrag(80, 5), "right");
  assert.equal(viewForDrag(-80, 5), "left");
});
test("dominant diagonal drag chooses a diagonal render", () => {
  assert.equal(viewForDrag(70, -70), "upper_right");
  assert.equal(viewForDrag(-70, 70), "lower_left");
});
test("short drag preserves the current named view", () => {
  assert.equal(viewForDrag(8, 5, "back"), "back");
});
```

- [ ] **Step 2: Verify tests fail because the module is missing**

Run: `node --test tests/web/view_navigator.test.mjs`

Expected: module-not-found failure.

- [ ] **Step 3: Implement the pure classifier and element attachment helper**

Export `viewForDrag(dx, dy, currentName, threshold = 32)`. Return current name
below threshold. Use a diagonal when both absolute components exceed threshold
and their ratio is between 0.5 and 2.0; use `upper_*` for negative `dy`,
`lower_*` for positive `dy`. Otherwise choose left/right if `abs(dx) >=
abs(dy)`, or up/down. Export `attachViewNavigator(element, onSelect)` which
uses `pointerdown`, `pointerup`, and `setPointerCapture`; it calls `onSelect`
only when the classifier changes selection.

- [ ] **Step 4: Verify JavaScript tests pass**

Run: `node --test tests/web/view_navigator.test.mjs`

Expected: three passing tests.

- [ ] **Step 5: Commit navigator logic**

```powershell
git add web_demo/static/view_navigator.js tests/web/view_navigator.test.mjs
git commit -m "Add draggable rendered-view navigator"
```

### Task 6: Build the responsive upload, status, and result UI

**Files:**
- Create: `web_demo/templates/index.html`
- Create: `web_demo/templates/job.html`
- Create: `web_demo/static/app.css`
- Create: `web_demo/static/upload.js`
- Create: `web_demo/static/job.js`
- Modify: `web_demo/routes.py`

- [ ] **Step 1: Write failing page-render tests**

```python
def test_home_page_contains_upload_control(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b'name="image"' in response.data
    assert b"Generate views" in response.data

def test_job_page_contains_status_and_navigator_shell(client, app):
    manager = app.extensions["job_manager"]
    job = manager.create_job("room.png", BytesIO(png_bytes()))
    response = client.get(f"/jobs/{job['id']}")
    assert b'id="job-status"' in response.data
    assert b'id="view-navigator"' in response.data
```

- [ ] **Step 2: Verify page tests fail**

Run: `python -m pytest tests/test_routes.py -q`

Expected: missing home page or expected HTML-marker failures.

- [ ] **Step 3: Implement the page templates and client behavior**

Add `GET /` that renders `index.html`. The upload page must have a labelled
file input, drag/drop target, client-side file type/size message, disabled
submit during the POST, and plain-language local CPU note. `upload.js` posts
`FormData` to `/api/jobs` and redirects to `job_url` on success.

The job page emits only its job ID in `data-job-id`; `job.js` fetches its status
every two seconds. For incomplete jobs it displays `Queued`, `Inferring 3D
scene`, or `Rendering 10 views`. For complete jobs it builds thumbnail buttons
from `result.views`, sets the main image source via result URLs, connects
`attachViewNavigator`, shows the selected direction in an `aria-live` label,
and exposes a GIF download anchor. For failure it stops polling and shows the
bounded error plus an upload-page retry link.

`app.css` must support 320px-wide screens, visible focus rings, a 16:10 main
viewer, an overflow-scrolling thumbnail row, and `cursor: grab/grabbing` on
the view navigator. Use no front-end build step and no remote CDN assets.

- [ ] **Step 4: Verify all template/API tests and browser logic tests pass**

Run:

```powershell
python -m pytest tests -q
node --test tests/web/view_navigator.test.mjs
```

Expected: all tests pass.

- [ ] **Step 5: Commit the user interface**

```powershell
git add web_demo/templates web_demo/static web_demo/routes.py tests/test_routes.py
git commit -m "Add UniSHARP web demo interface"
```

### Task 7: Add a safe local launcher and user documentation

**Files:**
- Create: `scripts/run_web_demo.py`
- Modify: `README.md`
- Modify: `tests/test_routes.py`

- [ ] **Step 1: Write failing launcher configuration test**

```python
def test_loopback_is_default_and_non_loopback_requires_explicit_flag():
    from scripts.run_web_demo import parse_args
    assert parse_args([]).host == "127.0.0.1"
    with pytest.raises(SystemExit):
        parse_args(["--host", "0.0.0.0"])
```

- [ ] **Step 2: Verify it fails because the launcher is missing**

Run: `python -m pytest tests/test_routes.py::test_loopback_is_default_and_non_loopback_requires_explicit_flag -q`

Expected: import failure for `scripts.run_web_demo`.

- [ ] **Step 3: Implement launcher and documentation**

Implement `parse_args(argv)` with `--host` defaulting to `127.0.0.1`,
`--port` defaulting to 5000, and `--allow-network` required when host is not
loopback. `main()` creates the app with host/port overrides and calls
`app.run(host=args.host, port=args.port, debug=False, threaded=True)`.

Add a README section that installs dependencies, starts the local service with:

```powershell
python scripts\run_web_demo.py
```

Document `http://127.0.0.1:5000`, the CPU-only queue, 768×512 demo output,
expected multi-minute latency, finite-view drag semantics, output directory,
and that public deployment requires a reverse proxy, HTTPS, authentication,
upload limits, and a GPU-capable worker design. Do not document `0.0.0.0` as a
normal startup path.

- [ ] **Step 4: Verify the complete automated suite and a server startup**

Run:

```powershell
python -m pytest tests -q
node --test tests/web/view_navigator.test.mjs
python scripts\run_web_demo.py --help
```

Expected: all tests pass and help shows the loopback default.

- [ ] **Step 5: Perform the manual smoke test**

Run `python scripts\run_web_demo.py`, open `http://127.0.0.1:5000`, upload
`dataset/20260831094111_73_2.jpg`, wait for completion, confirm each of the ten
thumbnail images can be selected, drag to left/right/up/down/diagonal views,
open the GIF download, then stop the server. Confirm the job files are under
`outputs/web_demo/<job-id>/`.

- [ ] **Step 6: Commit the launcher and documentation**

```powershell
git add scripts/run_web_demo.py README.md tests/test_routes.py
git commit -m "Document local UniSHARP web demo"
```
