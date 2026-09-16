from pathlib import Path

import pytest

from web_demo import create_app


@pytest.fixture()
def app(tmp_path: Path):
    return create_app(
        {
            "TESTING": True,
            "JOB_ROOT": tmp_path / "jobs",
            "REPO_ROOT": tmp_path / "repo",
            "MAX_UPLOAD_BYTES": 1024 * 1024,
        }
    )


@pytest.fixture()
def client(app):
    return app.test_client()
