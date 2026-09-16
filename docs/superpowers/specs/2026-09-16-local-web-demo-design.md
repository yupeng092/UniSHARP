# UniSHARP Local Interactive Web Demo Design

## Goal

Provide a local browser demo in which a user uploads one image, waits for the
existing UniSHARP CPU inference and named-camera renderer to complete, then
browses the resulting ten novel views by dragging the main image or selecting
a thumbnail.

## Scope and deployment boundary

The first release is a local demonstration service. It listens on
`127.0.0.1` by default and is not configured for public Internet access,
authentication, multiple machines, multi-user quotas, or persistent job
recovery. A later public deployment can put a reverse proxy, authentication,
and a GPU worker queue in front of the same job interface.

The service runs on the current Windows host and invokes the existing CPU
scripts. CPU inference and high-resolution rendering are intentionally slow;
the first demo therefore uses a 768 x 512 render and makes the waiting state
visible rather than promising real-time generation.

## User experience

1. The home page explains the local-demo limitation and provides a drag/drop
   uploader that accepts one JPG, JPEG, PNG, or WebP image.
2. After validation, the page creates one job and shows an isolated job page
   with queued, inference, rendering, complete, or failed state. It polls a
   JSON status endpoint for progress.
3. On completion, the main canvas displays a selected rendered view with a
   compact direction label. Ten thumbnail buttons remain available below it.
4. Pointer dragging on the main canvas maps to the nearest pre-rendered view:
   horizontal drags choose left/right; vertical drags choose up/down; diagonal
   drags choose the four diagonal views; forward/back are available from the
   thumbnail strip. A click or a short drag does not invent a continuous,
   unseen view.
5. The completion page also exposes the generated GIF and a download link for
   it. It may show the source image for comparison.

The drag interaction is deliberately a navigator for the finite renderer
outputs. It is not a browser-resident 3D Gaussian renderer and must not claim
continuous, physically correct novel-view synthesis between frames.

## Architecture

### Web server

A small Python server provides HTML, static assets, and a JSON API. It owns
job directories at:

```text
outputs/web_demo/<job-id>/
├── upload/<sanitized-original-filename>
├── inference/<image-slug>/gaussians.pt
├── render/rgb/*.png
├── render/multiview_report.json
├── render/multiview.gif
└── job.json
```

The server launches the existing commands as child processes instead of
duplicating UniSHARP model or renderer behavior:

```powershell
python scripts\infer_unisharp_cpu.py ...
python scripts\render_unisharp_cpu.py --trajectory rig ...
python scripts\make_multiview_gif.py ...
```

Jobs run one at a time. This avoids concurrent CPU models exhausting memory
and makes state unambiguous on the demonstration machine. The worker writes
`job.json` atomically at every phase transition. It records sanitized source
metadata, phase, timestamps, command exit status, concise error text, and
result-relative URLs; it never exposes arbitrary host paths to the browser.

### API

The minimum API is:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/jobs` | Validate one multipart upload and queue a job. |
| `GET` | `/api/jobs/{job_id}` | Return current phase, error summary, and result URLs. |
| `GET` | `/jobs/{job_id}` | Serve the job results page. |
| `GET` | `/job-files/{job_id}/{relative_path}` | Serve only approved files beneath that job directory. |

The server rejects missing images, unsupported types, oversize uploads, invalid
job identifiers, and traversal attempts. A status response reveals only job
state and approved relative URLs.

### Front end

The front end uses static HTML, CSS, and vanilla JavaScript to keep the demo
dependency-light. It has distinct modules for upload submission, job polling,
and the view navigator. The navigator receives a view manifest from the API,
uses pointer events, and updates the selected view and accessible label.
Thumbnails remain keyboard-operable buttons.

## Rendering configuration

The demo uses the committed `configs/camera_rig_small10.json`, with
`--camera-orientation look_at`, `--backend torch`, and `--no-save-gaussians`.
It uses width 768 and height 512. This keeps output practical on CPU while
maintaining the established conservative baselines. The exact Python
interpreter, checkpoint path, and maximum inference edge are configured in a
server-side settings module rather than accepted from the browser.

## Failure behavior

- Upload validation errors are returned directly to the uploader.
- A child-process failure marks the job `failed`, retains a short non-sensitive
  message for the UI, and leaves its files for local diagnosis.
- Missing expected render files after a zero exit status also marks the job
  `failed`.
- Polling stops at `complete` or `failed`; a job page shows a retry link that
  returns the user to upload rather than silently rerunning work.

## Testing and acceptance criteria

- Unit tests cover upload validation, job-ID/path containment, state
  transitions, render-manifest parsing, and drag-direction selection.
- API tests create a temporary job root, submit a small valid image, and mock
  child processes at the process boundary; they verify successful status and
  safely reported failures without running the model.
- A manual smoke test runs the service locally, uploads
  `dataset/20260831094111_73_2.jpg`, waits for a finished job, checks all ten
  thumbnails, drags among horizontal/vertical/diagonal outputs, and opens the
  GIF.
- The README documents installation, startup, the local-only default, expected
  CPU latency, and how to enable a public deployment only after adding the
  required security and compute infrastructure.

## Explicit non-goals

- No public hosting, login, billing, or multi-user scheduler.
- No browser-side continuous Gaussian Splatting in this release.
- No diffusion completion, GPU job execution, or changing UniSHARP inference
  and renderer mathematics.
