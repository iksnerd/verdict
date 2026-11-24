"""Fourth audit of the CLI: what a scorecard is stamped from, the pin and the family probe,
`update` under a second Ctrl-C, IPv6 ports, atomic writes through symlinks, and a 207 batch."""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import socket
import stat
import types

import pytest

from verdict import cli, client, config, library
from verdict.cli import inference, parser, setup, update
from verdict.evaluation import bench
from conftest import typed
from fakes import KIND_STATES as STATES, RISKY_BANK as BANK, noul

# --- 1. bench: the scorecard is stamped from what answered -----------------------------------------

def _bench(monkeypatch, local_engine, tmp_path, decide, suites=("sst2-yesno",), extra=()):
    monkeypatch.setattr(client, "decide", decide)
    monkeypatch.setattr(client, "capabilities", lambda *a, **k: (_ for _ in ()).throw(
        client.NoServer("no capabilities")))
    local_engine()
    monkeypatch.setattr(bench, "fetch", lambda s: [])
    monkeypatch.setattr(bench, "sample", lambda s, rows: [("t", "x")])

    def score(s, items, ask):
        ask({"text": "t"}, {"q": s.question})
        return {"metric": "accuracy", "value": 0.5, "ci": [0.4, 0.6], "n": 1, "chance": 0.5}

    monkeypatch.setattr(bench, "score", score)
    out = tmp_path / "card.json"
    argv = ["bench", "--pause", "0", "--out", str(out)]
    for s in suites:
        argv += ["--suite", s]
    return argv + list(extra), out


def _served(state, questions, url=None, model=None, **flags):
    return {"model": "served-model", "answers": typed(questions, lambda k: noul(0.5))}


def _down(state, questions, url=None, model=None, **flags):
    raise client.NoServer("down")


def test_bench_refuses_to_stamp_a_scorecard_from_a_server_run(monkeypatch, tmp_path, capsys, local_engine):
    argv, out = _bench(monkeypatch, local_engine, tmp_path, _served)
    assert cli.main(argv) == 2
    assert not out.exists()
    err = capsys.readouterr().err
    assert "served-model" in err and "in-process" in err


def test_bench_in_process_still_writes_its_scorecard(monkeypatch, tmp_path, capsys, local_engine):
    argv, out = _bench(monkeypatch, local_engine, tmp_path, _down)
    assert cli.main(argv) == 0
    card = json.loads(out.read_text())
    assert card["bits"] == config.load().bits and "model" in card


def test_bench_warns_when_its_askers_did_not_share_a_source(monkeypatch, tmp_path, capsys, local_engine):
    calls = []

    def decide(state, questions, url=None, **kw):
        calls.append(1)
        return _served(state, questions) if len(calls) == 1 else _down(state, questions)

    argv, out = _bench(monkeypatch, local_engine, tmp_path, decide, suites=("sst2-yesno", "bg-yesno"))
    argv.remove("--out"); argv.remove(str(out))
    assert cli.main(argv) == 0
    assert "not all answered by the same source" in capsys.readouterr().err


# --- 3. the pin and the family probe are one state machine -----------------------------------------

def test_once_pinned_in_process_the_family_is_never_probed_again(monkeypatch, capsys, local_engine):
    probes, caps = [], []
    monkeypatch.setattr(client, "decide", _down)
    monkeypatch.setattr(client, "capabilities",
                        lambda *a, **k: caps.append(1) or {"family": "jev", "model": "m"})
    local_engine()
    monkeypatch.setattr("sys.stdin", io.StringIO(STATES))
    real = library.prepare

    def prepare(state, questions, **kw):
        probes.append(kw["max_options"])
        return real(state, questions, **kw)

    monkeypatch.setattr(library, "prepare", prepare)
    assert cli.main(["decide", "--jsonl", "-q", BANK]) == 0
    assert probes[1:] == [library.MAX_OPTIONS] * 4
    assert len(caps) == 1


# --- 4. update --------------------------------------------------------------------------------------

pytestmark_update = pytest.mark.usefixtures("no_release_lookup")


@pytest.mark.usefixtures("no_release_lookup")
def test_a_second_ctrl_c_during_the_restore_says_how_to_reinstall(tool, monkeypatch, capsys):
    plain = tool.__call__
    seen = []

    def call(argv, cwd=None):
        if argv[:3] == ["uv", "tool", "install"]:
            tool.calls.append(argv)
            seen.append(argv[-1])
            if argv[-1].endswith("@v0.10.0"):
                return 1, "error: boom"
            raise KeyboardInterrupt  # the restore
        return plain(argv, cwd)

    monkeypatch.setattr(update, "_run", call)
    assert cli.main(["update"]) == 130
    err = capsys.readouterr().err
    assert "uv tool install" in err and "missing" in err


@pytest.mark.usefixtures("no_release_lookup")
def test_the_restore_message_does_not_claim_the_update_did_not_happen(
        tool, monkeypatch, capsys, scripted_installs):
    monkeypatch.setattr(update, "_run", scripted_installs(tool, (1, "error: boom"), (0, "")))
    assert cli.main(["update"]) == 2
    err = capsys.readouterr().err
    assert "did not happen" not in err and "reinstall" in err


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_released_skips_entries_that_are_not_objects(monkeypatch):
    import urllib.request

    body = json.dumps([1, "x", None, {"tag_name": "v0.3.0"}]).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(body))
    assert update._released("https://github.com/iksnerd/verdict.git") == {(0, 3, 0)}


@pytest.mark.parametrize("body,up", [(b'{"ready": false}', False), (b'{"ready": true}', True),
                                     (b'{"status": "ok"}', True), (b"ok", True)])
def test_server_up_reads_ready_from_healthz(monkeypatch, body, up):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(body))
    assert update._server_up() is up


# --- 5. setup ---------------------------------------------------------------------------------------

def _ipv6_loopback():
    try:
        s = socket.socket(socket.AF_INET6)
        s.bind(("::1", 0))
        s.close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ipv6_loopback(), reason="no IPv6 loopback")
def test_port_free_understands_an_ipv6_host():
    s = socket.socket(socket.AF_INET6)
    s.bind(("::1", 0))
    s.listen()
    port = s.getsockname()[1]
    try:
        assert setup._port_free("::1", port) is False
    finally:
        s.close()
    assert setup._port_free("::1", port) is True


def test_a_server_that_stopped_for_another_reason_does_not_blame_the_port(capsys):
    def run(*a, **k):
        raise SystemExit(3)  # a lifespan failure

    args = types.SimpleNamespace(host="127.0.0.1", port=0, log_level="warning")
    assert setup._run(types.SimpleNamespace(run=run), object(), args) == 2
    assert "taken" not in capsys.readouterr().err


def _init(monkeypatch, init_env, tmp_path, free):
    init_env(free)
    monkeypatch.setattr(setup, "_verdict_listens", lambda url: False)
    target = tmp_path / "c.toml"
    assert cli.main(["init", "--yes", "--out", str(target)]) == 0
    return target


def test_init_does_not_call_a_taken_port_free_when_none_is(monkeypatch, init_env, tmp_path, capsys):
    _init(monkeypatch, init_env, tmp_path, lambda *a: False)
    out = capsys.readouterr().out
    assert "is free" not in out and "every port up to" in out


def test_init_does_not_freeze_the_environment_into_a_new_file(monkeypatch, init_env, tmp_path):
    monkeypatch.setenv("VERDICT_URL", "http://127.0.0.1:5999")
    monkeypatch.setenv("VERDICT_BITS", "8")
    target = _init(monkeypatch, init_env, tmp_path, lambda *a: True)
    text = target.read_text()
    assert "5999" not in text and "bits = 8" not in text


# --- 6. inference -----------------------------------------------------------------------------------

def test_write_atomic_keeps_a_symlink_and_the_file_mode(tmp_path):
    real = tmp_path / "dotfiles" / "config.toml"
    real.parent.mkdir()
    real.write_text("old")
    real.chmod(0o600)
    link = tmp_path / "config.toml"
    link.symlink_to(real)
    inference.write_atomic(link, "new")
    assert link.is_symlink() and real.read_text() == "new"
    assert stat.S_IMODE(real.stat().st_mode) == 0o600


def test_a_pinned_in_process_run_with_no_engine_loads_instead_of_a_key_error(monkeypatch, local_engine):
    monkeypatch.setattr(client, "decide", _down)
    local_engine()
    got = inference._answer({"kind": "a"}, {"q": {"type": "noul", "instructions": "Is `kind` bad?"}},
                            "http://127.0.0.1:1", "m", pin=("in-process", 2))
    assert got["source"] == "in-process"


# --- 7. a 207 batch ---------------------------------------------------------------------------------

def test_a_batch_item_with_an_error_is_a_server_failure(monkeypatch, capsys):
    from verdict.cli import routing

    monkeypatch.setattr(client, "route_batch", lambda prompts, url=None: [
        {"index": 0, "branch": "small", "reason": "r"}, {"index": 1, "error": "model fell over"}])
    with pytest.raises(client.ServerError) as exc:
        routing.decide_many(["a", "b"])
    assert "1" in str(exc.value) and "model fell over" in str(exc.value)
