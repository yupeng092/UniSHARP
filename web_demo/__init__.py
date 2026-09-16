"""Flask application factory for the localhost-only UniSHARP demo."""

from __future__ import annotations

from typing import Any

from flask import Flask

from .config import DEFAULTS


def create_app(overrides: dict[str, Any] | None = None) -> Flask:
    """Create the web demo application without accepting request-side settings."""
    app = Flask(__name__)
    app.config.from_mapping(DEFAULTS)
    if overrides:
        app.config.update(overrides)
    app.config["JOB_ROOT"].mkdir(parents=True, exist_ok=True)
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_UPLOAD_BYTES"]
    return app
