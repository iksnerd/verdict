"""Second audit of the decision commands: fit identity, a checkpoint that changes mid-run, the
proxy probe, empty example banks and atomic output. Servers are faked; no model loads."""
from __future__ import annotations

import json

import pytest

from verdict import cli, client
from verdict.cli import evaluation, inference
from conftest import typed
from fakes import noul

BANK = json.dumps({"risky": {"type": "noul", "instructions": "Is `kind` marked bad?"}})
CALL = ["decide", '{"kind": "bad"}', "--questions", BANK]


def serve(monkeypatch, model="fake"):
    def decide(state, questions, url=None, model_=None, model=None, **flags):
        return {"model": served[0], "answers": typed(questions, lambda k: noul(0.5))}

    served = [model]
    monkeypatch.setattr(client, "decide", lambda s, q, url=None, model=None, **f: {
        "model": served[0], "answers": typed(q, lambda k: noul(0.5))})
    return served


def fit_file(tmp_path, doc):
    f = tmp_path / "fit.json"
    f.write_text(json.dumps(doc))
    return str(f)


FIT = {"risky": {"type": "noul", "cut": 0.4}}


# 1. fit identity

def test_decide_warns_when_the_fit_was_made_on_another_model(monkeypatch, tmp_path, capsys):
    serve(monkeypatch, "model-b")
    f = fit_file(tmp_path, {"model": "model-a", "questions": FIT})
    assert cli.main([*CALL, "--calibration", f]) == 0
    assert "fitted on model-a, but model-b answered" in capsys.readouterr().err


def test_decide_is_quiet_when_the_model_matches(monkeypatch, tmp_path, capsys):
    serve(monkeypatch, "model-a")
    f = fit_file(tmp_path, {"model": "model-a", "questions": FIT})
    assert cli.main([*CALL, "--calibration", f]) == 0
    assert "fitted" not in capsys.readouterr().err


def test_decide_warns_on_a_budget_mismatch(monkeypatch, tmp_path, capsys):
    serve(monkeypatch, "model-a")
    f = fit_file(tmp_path, {"model": "model-a", "budget": 1, "questions": FIT})
    assert cli.main([*CALL, "--calibration", f]) == 0
    assert "fitted at budget 1" in capsys.readouterr().err


def test_ask_cut_file_warns_on_another_model(monkeypatch, tmp_path, capsys):
    serve(monkeypatch, "model-b")
    f = fit_file(tmp_path, {"model": "model-a", "questions": FIT})
    cli.main(["ask", "x", "Is it bad?", "--cut", f])
    assert "fitted on model-a" in capsys.readouterr().err


def test_an_empty_fits_file_warns_instead_of_silently_skipping(monkeypatch, tmp_path, capsys):
    serve(monkeypatch)
    f = fit_file(tmp_path, {"questions": {}})
    assert cli.main([*CALL, "--calibration", f]) == 0
    assert "holds no fits" in capsys.readouterr().err


def test_a_bank_as_calibration_is_a_usage_error(monkeypatch, tmp_path, capsys):
    serve(monkeypatch)
    f = fit_file(tmp_path, json.loads(BANK))
    assert cli.main([*CALL, "--calibration", f]) == 2
    assert "looks like a question bank" in capsys.readouterr().err


def test_calibrate_records_bits_and_budget(monkeypatch, tmp_path):
    serve(monkeypatch)
    rows = [{"state": {"kind": str(i)}, "label": i % 2 == 0} for i in range(20)]
    data = tmp_path / "l.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "fit.json"
    # noul 0.5 for every row gives no spread; the fit may still succeed or fail, so use varying scores
    monkeypatch.setattr(client, "decide", lambda s, q, url=None, model=None, **f: {
        "model": "m", "answers": typed(q, lambda k: noul(0.9 if s["kind"].isdigit() and int(s["kind"]) % 2 == 0 else 0.1))})
    assert cli.main(["calibrate", str(data), "-q", BANK, "--out", str(out), "--pause", "0"]) == 0
    doc = json.loads(out.read_text())
    assert doc["model"] == "m" and "bits" in doc and "budget" in doc


# 2. the checkpoint changing mid-run

def test_a_server_that_disappears_mid_run_is_flagged(monkeypatch, capsys):
    calls = []

    def decide(state, questions, url=None, model=None, **flags):
        calls.append(state)
        if len(calls) > 1:
            raise client.NoServer("gone")
        return {"model": "served", "answers": typed(questions, lambda k: noul(0.5))}

    class Eng:
        name = "local"

        def clip_state(self, state, budget):
            return state

        def predict(self, state, questions):
            return typed(questions, lambda k: noul(0.5))

    monkeypatch.setattr(client, "decide", decide)
    monkeypatch.setattr(inference, "load", lambda *a, **k: Eng())
    monkeypatch.setattr(inference, "_ENGINES", {})
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO('{"kind": "a"}\n{"kind": "b"}\n'))
    assert cli.main(["decide", "--jsonl", "-q", BANK]) == 0
    assert "answers switched from server (served) to in-process (local)" in capsys.readouterr().err


def test_calibrate_fails_on_mixed_models_before_printing_fits(monkeypatch, tmp_path, capsys):
    n = []

    def decide(state, questions, url=None, model=None, **flags):
        n.append(1)
        return {"model": "a" if len(n) < 10 else "b",
                "answers": typed(questions, lambda k: noul(0.9 if int(state["kind"]) % 2 == 0 else 0.1))}

    monkeypatch.setattr(client, "decide", decide)
    data = tmp_path / "l.jsonl"
    data.write_text("\n".join(json.dumps({"state": {"kind": str(i)}, "label": i % 2 == 0})
                              for i in range(20)))
    assert cli.main(["calibrate", str(data), "-q", BANK, "--pause", "0"]) == 2
    out = capsys.readouterr()
    assert "more than one model" in out.err
    assert "held-out" not in out.out


# 3. the proxy probe after a flap

def test_a_proxy_that_appears_after_a_flap_is_recognised(monkeypatch):
    import argparse

    probes = iter([None, None, "jev"])   # down, down (re-ask), then up once the server returned
    state = {"i": 0}

    def capabilities(url=None, model=None, timeout=None):
        fam = next(probes)
        if fam is None:
            raise client.NoServer("down")
        return {"family": fam}

    seq = ["server", "server", "local", "server", "server"]

    class Eng:
        name = "local"

        def clip_state(self, s, b):
            return s

        def predict(self, s, q):
            return typed(q, lambda k: noul(0.5))

    def decide(s, q, url=None, model=None, **f):
        kind = seq[state["i"]]
        if kind == "local":
            raise client.NoServer("gone")
        return {"model": "m", "answers": typed(q, lambda k: noul(0.5))}

    monkeypatch.setattr(client, "capabilities", capabilities)
    monkeypatch.setattr(client, "decide", decide)
    monkeypatch.setattr(inference, "load", lambda *a, **k: Eng())
    monkeypatch.setattr(inference, "_ENGINES", {})
    ns = argparse.Namespace(url=None, model=None, lang=None, server_only=False,
                            allow_unmeasured=True, yesno=False, warn_clipped=False)
    ask = inference._Asker(ns)
    bank = json.loads(BANK)
    for i in range(5):
        state["i"] = i
        ask({"kind": "x"}, bank)
    assert ask._family == "jev"


# 5. examples with only labelled files

def test_examples_with_only_labelled_rows_does_not_crash(monkeypatch, tmp_path, capsys):
    from verdict.cli import examples_cmd

    d = tmp_path / "toy"
    d.mkdir()
    (d / "bank.json").write_text(BANK)
    (d / "labelled.jsonl").write_text(json.dumps({"state": {"kind": "a"}, "label": True}) + "\n")
    monkeypatch.setattr(examples_cmd, "_examples_root", lambda: tmp_path)
    assert cli.main(["examples", "toy"]) == 0
    assert "no rows to ask" in capsys.readouterr().out


# 6. atomic writes

def test_write_atomic_leaves_the_old_file_when_the_write_fails(tmp_path, monkeypatch):
    target = tmp_path / "card.json"
    target.write_text("old")
    monkeypatch.setattr("os.replace", lambda *a: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        inference.write_atomic(target, "new")
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["card.json"]


def test_calibrate_out_goes_through_the_atomic_writer(monkeypatch, tmp_path):
    serve(monkeypatch)
    monkeypatch.setattr(client, "decide", lambda s, q, url=None, model=None, **f: {
        "model": "m", "answers": typed(q, lambda k: noul(0.9 if int(s["kind"]) % 2 == 0 else 0.1))})
    data = tmp_path / "l.jsonl"
    data.write_text("\n".join(json.dumps({"state": {"kind": str(i)}, "label": i % 2 == 0})
                              for i in range(20)))
    seen = []
    real = inference.write_atomic
    monkeypatch.setattr(inference, "write_atomic", lambda p, t: (seen.append(str(p)), real(p, t)))
    out = tmp_path / "fit.json"
    assert cli.main(["calibrate", str(data), "-q", BANK, "--out", str(out), "--pause", "0"]) == 0
    assert seen == [str(out)]
