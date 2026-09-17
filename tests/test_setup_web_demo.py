from scripts.setup_web_demo import parse_args


def test_setup_can_prepare_and_start_with_explicit_port():
    args = parse_args(["--start", "--port", "5001"])
    assert args.start is True
    assert args.port == 5001


def test_setup_defaults_to_localhost_only():
    assert parse_args([]).host == "127.0.0.1"
