"""The CLI applies laya's rewrite and option cap only when laya is behind the server.

It used to run `library.prepare` for laya on every call, so hosted Jev behind `verdict serve`
received a yes/no rewritten as a no/yes choice and a 25-option bank was refused. A server says what
it fronts at `/v1/capabilities`; the CLI asks once per command. The wording refusals stay for every
backend (§47).
"""
from __future__ import annotations

import json

import pytest

from verdict import cli, client


@pytest.fixture
def server(monkeypatch):
    """A fake server. `family` is what /v1/capabilities reports; None means no such endpoint."""
    box = {"family": "laya", "sent": [], "probes": 0}

    def capabilities(url=None, model=None, timeout=None):
        box["probes"] += 1
        if box["family"] is None:
            raise client.NoServer("no capability discovery")
        return {"family": box["family"], "model": "m"}

    def decide(state, questions, url=None, model=None, **flags):
        box["sent"].append((questions, flags))
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.8}
            else:
                keys = list(q["criteria"])
                p = {k: (0.9 if i == 0 else 0.1 / (len(keys) - 1)) for i, k in enumerate(keys)}
                answers[qid] = {"type": "choice", "choice": keys[0], "probabilities": p,
                                "confidence": 0.5}
        return {"model": "served", "answers": answers}

    monkeypatch.setattr(client, "capabilities", capabilities)
    monkeypatch.setattr(client, "decide", decide)
    return box


YESNO = json.dumps({"q": {"type": "noul", "instructions": "Is `message` about money?"}})


def many(n):
    return json.dumps({"q": {"type": "choice", "instructions": "Which?",
                             "criteria": {f"o{i}": "" for i in range(n)}}})


def run(argv, capsys):
    code = cli.main(["decide", '{"message": "hi"}', "--server-only", *argv])
    return code, capsys.readouterr()


@pytest.mark.parametrize("family", ["jev", "systemone"])
def test_a_proxied_backend_gets_the_yes_no_as_asked(server, capsys, family):
    server["family"] = family
    code, out = run(["--questions", YESNO], capsys)
    assert code == 0
    assert server["sent"][0][0]["q"]["type"] == "noul"
    assert json.loads(out.out)["answers"]["q"]["noul"] == 0.8


def test_a_proxied_backend_takes_more_than_twenty_options(server, capsys):
    server["family"] = "jev"
    code, _ = run(["--questions", many(40)], capsys)
    assert code == 0 and len(server["sent"][0][0]["q"]["criteria"]) == 40


def test_a_proxied_backend_still_refuses_what_measured_at_chance(server, capsys):
    server["family"] = "jev"
    bank = json.dumps({"q": {"type": "noul", "instructions": "How dangerous is `message`?"}})
    code, out = run(["--questions", bank], capsys)
    assert code == 2 and server["sent"] == []
    assert json.loads(out.out)["error"]["code"] == "refused"


def test_laya_behind_the_server_keeps_the_rewrite_and_the_cap(server, capsys):
    code, _ = run(["--questions", YESNO], capsys)
    assert code == 0 and server["sent"][0][0]["q"]["type"] == "choice"
    code, out = run(["--questions", many(25)], capsys)
    assert code == 2 and "20" in json.loads(out.out)["error"]["message"]


def test_a_server_without_capability_discovery_is_treated_as_laya(server, capsys):
    server["family"] = None
    code, _ = run(["--questions", YESNO], capsys)
    assert code == 0 and server["sent"][0][0]["q"]["type"] == "choice"


def test_the_server_is_asked_once_per_command_not_once_per_row(server, monkeypatch, capsys):
    server["family"] = "jev"
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO('{"message": "a"}\n{"message": "b"}\n{"message": "c"}\n'))
    code = cli.main(["decide", "--jsonl", "-q", YESNO, "--server-only", "--pause", "0"])
    assert code == 0 and len(server["sent"]) == 3 and server["probes"] == 1


def test_a_server_that_dies_mid_run_does_not_leave_its_rules_on_the_local_fallback(server, monkeypatch):
    """Jev's rules (no rewrite, no cap) must not outlive the server on rows a local laya answers."""
    import argparse
    from verdict.cli import inference

    server["family"] = "jev"
    sources = iter(["server http://127.0.0.1:8799", "in-process", "in-process"])

    def answer(state, questions, *a, **k):
        return {"model": "m", "source": next(sources),
                "answers": {"q": {"type": "noul", "noul": 0.8, "confidence": 0.6}}}

    monkeypatch.setattr(inference, "_answer", answer)
    asker = inference._Asker(argparse.Namespace(url=None, model=None, lang="en", server_only=False))
    asker({"message": "a"}, json.loads(YESNO))
    assert asker._family == "jev"
    asker({"message": "b"}, json.loads(YESNO))   # the server is gone: answered in process
    assert asker._family is inference._UNASKED, "the next row must ask again, not reuse Jev's rules"


def test_a_baseline_message_names_the_models_it_did_find(tmp_path, capsys):
    from conftest import typed  # noqa: F401  (keeps the import style of this package)
    (tmp_path / "v0.6.0.json").write_text(json.dumps(
        {"model": "verdict-v1-mlx", "suites": {"toy": {"metric": "auc", "value": 0.8, "ci": [0.7, 0.9]}}}))
    (tmp_path / "v0.7.0.json").write_text(json.dumps(
        {"model": "laya-mlx", "suites": {"toy": {"metric": "auc", "value": 0.5, "ci": [0.4, 0.6]}}}))
    assert cli.main(["bench", "--verify", str(tmp_path / "v0.7.0.json")]) == 0
    err = capsys.readouterr().err
    assert "laya-mlx" in err and "verdict-v1-mlx" in err and "baseline" in err
