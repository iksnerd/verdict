"""A run asks the server or loads the model here once, on its first row, and keeps that source:
a server that flaps mid-run neither switches checkpoint nor hides behind a fallback."""
from __future__ import annotations

import io
import json

from verdict import cli, client
from verdict.cli import inference
from conftest import typed
from fakes import noul

BANK = json.dumps({"risky": {"type": "noul", "instructions": "Is `kind` marked bad?"}})
STATES = "".join(json.dumps({"kind": k}) + "\n" for k in "abcde")


class Eng:
    name = "local"

    def clip_state(self, state, budget):
        return state

    def predict(self, state, questions):
        return typed(questions, lambda k: noul(0.5))


def flap(monkeypatch, up):
    """`up(n)` says whether the server answers the n-th call (1-based)."""
    calls = []

    def decide(state, questions, url=None, model=None, **flags):
        calls.append(state)
        if not up(len(calls)):
            raise client.NoServer("flap")
        return {"model": "served", "answers": typed(questions, lambda k: noul(0.5))}

    monkeypatch.setattr(client, "decide", decide)
    monkeypatch.setattr(inference, "load", lambda *a, **k: Eng())
    monkeypatch.setattr(inference, "_ENGINES", {})
    monkeypatch.setattr("sys.stdin", io.StringIO(STATES))


def rows(out):
    return [r for r in map(json.loads, filter(str.strip, out.splitlines())) if "answers" in r]


def test_a_server_that_dies_at_row_3_fails_that_row_and_keeps_the_printed_ones(monkeypatch, capsys):
    flap(monkeypatch, lambda n: n < 3)
    assert cli.main(["decide", "--jsonl", "-q", BANK]) == 2
    out = capsys.readouterr()
    got = rows(out.out)
    assert [r["index"] for r in got] == [0, 1] and {r["model"] for r in got} == {"served"}
    assert "call 3" in out.err and "pinned to the server" in out.err


def test_a_server_that_appears_at_row_3_is_ignored_with_one_note(monkeypatch, capsys):
    flap(monkeypatch, lambda n: n >= 3)
    assert cli.main(["decide", "--jsonl", "-q", BANK]) == 0
    out = capsys.readouterr()
    got = rows(out.out)
    assert len(got) == 5 and {r["model"] for r in got} == {"local"}
    assert out.err.count("pinned to in-process") == 1
