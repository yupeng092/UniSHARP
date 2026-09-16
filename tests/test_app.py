def test_factory_defaults_to_loopback_host(app):
    assert app.config["BIND_HOST"] == "127.0.0.1"
    assert app.config["JOB_ROOT"].name == "jobs"
