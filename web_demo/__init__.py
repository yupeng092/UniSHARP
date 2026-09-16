"""Flask application factory for the localhost-only UniSHARP demo."""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

from .config import DEFAULTS
from .jobs import JobManager


def create_app(overrides: dict[str, Any] | None = None) -> Flask:
    """Create the web demo application without accepting request-side settings."""
    app = Flask(__name__)
    app.config.from_mapping(DEFAULTS)
    if overrides:
        app.config.update(overrides)
    app.config["JOB_ROOT"].mkdir(parents=True, exist_ok=True)
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_UPLOAD_BYTES"]
    app.extensions["job_manager"] = JobManager(app.config, start_worker=not app.config["TESTING"])

    from .routes import bp

    app.register_blueprint(bp)

    @app.errorhandler(RequestEntityTooLarge)
    def upload_too_large(_: RequestEntityTooLarge):
        return jsonify({"error": "image is too large"}), 413

    return app
