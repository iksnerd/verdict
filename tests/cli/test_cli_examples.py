"""`verdict examples`: runs examples/ banks against a backend and prints each row's answers --
the same real inputs examples/README.md's own numbers came from, generalized to any backend so a
new one (--systemone URL) can be checked against them instead of one-off questions."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from verdict import bench, cli, client
from verdict.cli import examples_cmd, inference
from conftest import typed

ROOT = Path(__file__).resolve().parents[2]


def fake_server(monkeypatch):
    calls = []

    def fake_decide(state, questions, url=None, model=None):
        calls.append((state, questions))
        return {"model": "fake", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": 0.5, "confidence": 0.5})}

    monkeypatch.setattr(client, "decide", fake_decide)
    return calls


def jev_transport(calls):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        body = json.loads(request.content)
        answers = {qid: {"type": "noul", "noul": 0.42, "confidence": 0.1} if q["type"] == "noul"
                   else {"type": "choice", "choice": next(iter(q["criteria"])),
                        "probabilities": {k: 1 / len(q["criteria"]) for k in q["criteria"]},
                        "confidence": 0.5} if q["type"] == "choice"
                   else {"type": "score", "score": 0.0,
                        "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                        "probabilities": {str(i): 1 / len(q["criteria"]) for i in range(len(q["criteria"]))},
                        "confidence": 0.5}
                   for qid, q in body["questions"].items()}
        return httpx.Response(200, json={
            "model": "jev-1.13", "usage": {"input_tokens": 9, "output_tokens": 0},
            "answers": answers})
    return httpx.MockTransport(handler)


def test_runs_every_example_by_default(monkeypatch, capsys):
    fake_server(monkeypatch)
    assert cli.main(["examples"]) == 0
    out = capsys.readouterr().out
    for name in ("room-triage", "secret-commands", "citation-check", "checklist-check"):
        assert f"=== {name} ===" in out


def test_a_name_selects_only_that_example(monkeypatch, capsys):
    fake_server(monkeypatch)
    assert cli.main(["examples", "citation-check"]) == 0
    out = capsys.readouterr().out
    assert "=== citation-check ===" in out
    assert "=== room-triage ===" not in out


def test_an_unknown_name_fails_cleanly(monkeypatch, capsys):
    fake_server(monkeypatch)
    assert cli.main(["examples", "not-a-real-example"]) == 2
    assert "no example" in capsys.readouterr().err


def test_systemone_bypasses_local_inference_entirely(monkeypatch, capsys):
    def boom(*a, **k):
        pytest.fail("must not load a local model when --systemone is given")

    monkeypatch.setattr(inference, "_Asker", boom)
    calls = []
    monkeypatch.setattr(bench, "_default_transport", lambda: jev_transport(calls))
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert cli.main(["examples", "citation-check", "--systemone", "https://api.example"]) == 0
    assert calls, "the fake systemone server was never called"
    out = capsys.readouterr().out
    assert "=== citation-check ===" in out
    assert "relation=" in out


def dead_transport():
    import httpx

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated: server never answered")
    return httpx.MockTransport(handler)


def test_a_persistently_unreachable_systemone_server_fails_cleanly(monkeypatch, capsys):
    """A server that never answers -- crashed, or never started -- must not surface as an
    unhandled traceback: `verdict examples` is a CLI, and a Python stack trace is not a usage
    error message."""
    monkeypatch.setattr(bench, "_default_transport", dead_transport)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    code = cli.main(["examples", "citation-check", "--systemone", "https://api.example",
                     "--pause", "0"])
    assert code == 2
    err = capsys.readouterr().err
    assert "loading" in err
    assert "Traceback" not in err


def _toy_example(tmp_path, rows):
    ex_dir = tmp_path / "toy"
    ex_dir.mkdir()
    (ex_dir / "bank.json").write_text(json.dumps({"q": {"type": "noul", "instructions": "?"}}))
    (ex_dir / "items.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_scores_against_expected_labels_when_present(monkeypatch, tmp_path, capsys):
    root = _toy_example(tmp_path, [
        {"state": {"text": "a"}, "expected": {"q": True}},
        {"state": {"text": "b"}, "expected": {"q": False}},
    ])
    monkeypatch.setattr(examples_cmd, "_examples_root", lambda: root)
    monkeypatch.setattr(client, "decide", lambda state, questions, url=None, model=None: {
        "model": "fake", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": 0.9, "confidence": 0.5})})
    assert cli.main(["examples", "toy"]) == 0
    assert "toy: 1/2 match the recorded expectation" in capsys.readouterr().out


def test_no_score_line_when_nothing_carries_an_expectation(monkeypatch, tmp_path, capsys):
    root = _toy_example(tmp_path, [{"state": {"text": "a"}}])
    monkeypatch.setattr(examples_cmd, "_examples_root", lambda: root)
    monkeypatch.setattr(client, "decide", lambda state, questions, url=None, model=None: {
        "model": "fake", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": 0.9, "confidence": 0.5})})
    assert cli.main(["examples", "toy"]) == 0
    assert "match the recorded expectation" not in capsys.readouterr().out


def test_reports_latency_per_example(monkeypatch, capsys):
    fake_server(monkeypatch)
    assert cli.main(["examples", "citation-check"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"citation-check: \d+ ms/call over \d+ calls", out)


def test_pause_between_calls_is_not_counted_as_latency(monkeypatch, tmp_path, capsys):
    """`--pause` is a deliberate delay between calls (be gentle on a laptop GPU), not part of
    what a call itself costs; counting it would make every backend look equally slow."""
    root = _toy_example(tmp_path, [{"state": {"text": "a"}}, {"state": {"text": "b"}}])
    monkeypatch.setattr(examples_cmd, "_examples_root", lambda: root)
    monkeypatch.setattr(client, "decide", lambda state, questions, url=None, model=None: {
        "model": "fake", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": 0.5, "confidence": 0.5})})
    assert cli.main(["examples", "toy", "--pause", "0.2"]) == 0
    out = capsys.readouterr().out
    ms = int(re.search(r"toy: (\d+) ms/call", out).group(1))
    assert ms < 100, f"pause leaked into the reported latency: {ms} ms/call"


def test_examples_is_packed_into_the_wheel():
    """A directory read in a checkout but missing from the wheel works in tests and fails on
    every install (docs.py's own topics have the same guard, tests/test_docs_cmd.py)."""
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packed = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert packed.get("examples") == "verdict/_examples"
