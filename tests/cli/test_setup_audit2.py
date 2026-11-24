"""Audit findings in `serve`, `init` and `main`: a port taken after the first look, a config
truncated mid-write, Ctrl-C and a closed pipe, and a JSON-errors flag left over from a prior call."""
import json
import socket

import pytest

from verdict import cli, config
from verdict.cli import parser, setup


def _hold(host):
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, 0))
    s.listen()
    return s, s.getsockname()[1]


def test_a_wildcard_listener_makes_loopback_not_free():
    s, port = _hold("0.0.0.0")
    try:
        assert setup._port_free("127.0.0.1", port) is False
    finally:
        s.close()
    assert setup._port_free("127.0.0.1", port) is True


def test_serve_rechecks_the_port_after_the_load_and_does_not_announce(monkeypatch, capsys):
    import uvicorn

    from verdict.backends import mlx

    class Engine:
        agent = tokenizer = None

    class Backend:
        name = "fake"
        engine = Engine()

        def __init__(self, *a, **k):
            pass

    calls = []

    def free(host, port):
        calls.append(1)
        return len(calls) == 1  # free before the load, taken after

    monkeypatch.setattr(mlx, "MlxBackend", Backend)
    monkeypatch.setattr(setup, "_port_free", free)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("bound a taken port"))
    assert cli.main(["serve", "--port", "0", "--model", "x"]) == 2
    io = capsys.readouterr()
    assert "serving" not in io.out
    assert "cannot serve" in io.err


def test_a_lost_bind_inside_uvicorn_is_a_failure_not_an_answer(monkeypatch, capsys):
    import types

    def run(*a, **k):
        raise SystemExit(1)

    args = types.SimpleNamespace(host="127.0.0.1", port=1, log_level="warning")
    assert setup._run(types.SimpleNamespace(run=run), object(), args) == 2


OLD = '[server]\nurl = "http://127.0.0.1:1"\n'


def _config_target(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATHS", (tmp_path / "absent.toml",))
    monkeypatch.setattr(setup, "_port_free", lambda *a: True)
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "c.toml"
    target.write_text(OLD)
    return target


def test_init_never_leaves_a_truncated_config(monkeypatch, tmp_path):
    from pathlib import Path

    target = _config_target(tmp_path, monkeypatch)
    original = Path.write_text

    def interrupted(self, text, *a, **k):
        original(self, text[:5], *a, **k)
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "write_text", interrupted)
    assert cli.main(["init", "--yes", "--out", str(target)]) == 130
    assert target.read_text() == OLD
    assert [p.name for p in tmp_path.iterdir()] == ["c.toml"]


def test_init_writes_through_a_replace(monkeypatch, tmp_path):
    target = _config_target(tmp_path, monkeypatch)
    assert cli.main(["init", "--yes", "--out", str(target)]) == 0
    assert config.load(target).url.startswith("http://")
    assert [p.name for p in tmp_path.iterdir()] == ["c.toml"]


def test_init_refuses_to_overwrite_without_a_terminal(monkeypatch, tmp_path, capsys):
    target = _config_target(tmp_path, monkeypatch)
    monkeypatch.setattr("sys.stdin", open("/dev/null"))
    assert cli.main(["init", "--out", str(target)]) == 2
    assert str(target) in capsys.readouterr().err
    assert target.read_text() == OLD


def test_ctrl_c_exits_130_without_a_traceback(monkeypatch, capsys):
    def boom(args):
        raise KeyboardInterrupt

    monkeypatch.setattr(setup, "_config_cmd", boom)
    assert cli.main(["config"]) == 130
    assert "Traceback" not in capsys.readouterr().err


def test_a_closed_pipe_is_not_an_answer(monkeypatch, capsys):
    def boom(args):
        raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setattr(setup, "_config_cmd", boom)
    assert cli.main(["config"]) == 2
    assert capsys.readouterr().err == ""


def test_json_errors_is_reset_at_the_start_of_every_call(capsys):
    from verdict.cli import support

    support.JSON_ERRORS = True  # left by a previous in-process call
    with pytest.raises(SystemExit):
        cli.main(["config", "--bogus-flag"])  # argparse fails before this call sets its own value
    assert support.JSON_ERRORS is False
    capsys.readouterr()
    assert cli.main(["decide", "{}", "--questions", "nope-not-a-preset"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]
    assert cli.main(["config", "--json"]) == 0
    assert support.JSON_ERRORS is True
    cli.main(["questions"])
    assert support.JSON_ERRORS is False
