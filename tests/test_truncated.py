"""A state longer than the token budget is read only in part, and the answer says so: `truncated:
true` on each answer of that call. The field is absent when
nothing was cut, so an unclipped call's output is what it always was, and `/v1/systemone`, whose
answer models are strict, never carries it. A fake tokenizer, one token per word; no model."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from verdict import cli, client
from verdict.api import create_app
from verdict.backend_mlx import MlxBackend
from verdict.cli import inference
from verdict.engine import Engine

from test_clip_state import Agent, WordTokenizer

BANK = {"q": {"type": "noul", "instructions": "?"}}
LONG = "x " * 500


def server(budget=10):
    backend = MlxBackend(agent=Agent(), budget=budget)
    backend.engine._tokenizer = WordTokenizer()
    return TestClient(create_app(backend))


def test_a_clipped_state_marks_every_answer_truncated():
    body = {"state": LONG, "questions": {**BANK, "r": {"type": "noul", "instructions": "??"}}}
    answers = server().post("/v1/decide", json=body).json()["answers"]
    assert {qid: a["truncated"] for qid, a in answers.items()} == {"q": True, "r": True}


def test_a_state_within_the_budget_has_no_truncated_key():
    answers = server().post("/v1/decide", json={"state": "short", "questions": BANK}).json()["answers"]
    assert "truncated" not in answers["q"]


def test_a_rewritten_yes_no_keeps_the_mark():
    # `noul` is asked as a no/yes choice unless ?yesno=true, then folded back into a noul answer.
    response = server().post("/v1/decide", json={"state": LONG, "questions": BANK})
    assert response.json()["answers"]["q"]["truncated"] is True


def test_systemone_never_carries_truncated():
    # A choice question: a yes/no is rewritten to a choice and back, which would drop the mark
    # before the endpoint ever saw it and the test would pass for the wrong reason.
    pick = {"c": {"type": "choice", "instructions": "?", "criteria": {"a": "one", "b": "two"}}}
    body = {"state": LONG, "model": "jev-latest", "questions": pick}
    decided = server().post("/v1/decide", json={"state": LONG, "questions": pick}).json()
    assert decided["answers"]["c"]["truncated"] is True
    answers = server().post("/v1/systemone", json=body).json()["answers"]
    assert "truncated" not in answers["c"]


def test_decide_in_this_process_marks_a_clipped_state(monkeypatch, capsys):
    def no_server(*args, **kwargs):
        raise client.NoServer("no server")

    engine = Engine("fake", agent=Agent())
    engine._tokenizer = WordTokenizer()
    monkeypatch.setattr(client, "decide", no_server)
    monkeypatch.setattr(inference, "load", lambda *args, **kwargs: engine)
    monkeypatch.setattr(inference, "_ENGINES", {})
    cli.main(["decide", json.dumps(LONG), "--questions", json.dumps(BANK), "--allow-unmeasured"])
    answers = json.loads(capsys.readouterr().out)["answers"]
    assert answers["q"]["truncated"] is True
    assert answers["q"]["noul"] == 0.5
