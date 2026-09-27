"""The multilingual checkpoint, asked for by name the way laya's `Router.predict(model=...)` does.

Nothing loads it silently: the English checkpoint collapses off English while staying confident,
so the CLI says so, but a second model on this machine is an explicit choice (CLAUDE.md)."""
from __future__ import annotations

import io
import json
import sys
import types

import pytest
from fastapi.testclient import TestClient

from verdict import cli, client
from conftest import typed
from verdict.api import create_app
from verdict.backend_mlx import MlxBackend

Q = {"urgent": {"type": "noul", "instructions": "?"}}


class Agent:
    def __init__(self, tag):
        self.tag, self.calls = tag, 0

    def predict(self, state, questions):
        self.calls += 1
        return {"answers": typed(
            questions, lambda k: {"type": "noul", "noul": 0.6, "confidence": 0.6})}


@pytest.fixture
def fake_laya(monkeypatch):
    loaded = {}
    fake = types.ModuleType("laya_mlx")
    fake.load = lambda model_id, **kw: loaded.setdefault(model_id, Agent(model_id))
    monkeypatch.setitem(sys.modules, "laya_mlx", fake)
    return loaded


def test_default_request_uses_the_main_checkpoint_and_never_loads_the_other(fake_laya):
    backend = MlxBackend("main", multilingual_id="multi")
    r = TestClient(create_app(backend)).post("/v1/decide", json={"state": "hi", "questions": Q})
    assert r.status_code == 200
    assert r.json()["model"] == "laya-mlx:main"
    assert set(fake_laya) == {"main"}


def test_model_multilingual_loads_it_once_and_answers_from_it(fake_laya):
    app = TestClient(create_app(MlxBackend("main", multilingual_id="multi")))
    for _ in range(2):
        r = app.post("/v1/decide", json={"state": "hola", "questions": Q, "model": "multilingual"})
        assert r.json()["model"] == "laya-mlx:multi"
    assert fake_laya["multi"].calls == 2


def test_multilingual_without_a_configured_checkpoint_is_a_400(fake_laya):
    r = TestClient(create_app(MlxBackend("main"))).post(
        "/v1/decide", json={"state": "hola", "questions": Q, "model": "multilingual"})
    assert r.status_code == 400


def test_unknown_model_name_is_rejected():
    """`model` is a plain string now (config decides the valid set, not a fixed Literal), so a
    typo like this is refused by the backend (400, UnknownModel) rather than by pydantic (422)."""
    r = TestClient(create_app(MlxBackend("main"))).post(
        "/v1/decide", json={"state": "x", "questions": Q, "model": "klingon"})
    assert r.status_code == 400


def fake_server(monkeypatch):
    sent = []

    def fake_decide(state, questions, url=None, model=None):
        sent.append(model)
        return {"model": "fake", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": 0.5, "confidence": 0.5})}

    monkeypatch.setattr(client, "decide", fake_decide)
    return sent


def test_lang_multi_asks_for_the_multilingual_checkpoint(monkeypatch):
    sent = fake_server(monkeypatch)
    cli.main(["ask", "Der Kunde wurde zweimal belastet", "urgent?", "--lang", "multi"])
    assert sent == ["multilingual"]


def test_auto_warns_on_non_english_but_does_not_switch(monkeypatch, capsys):
    pytest.importorskip("laya_mlx")
    sent = fake_server(monkeypatch)
    cli.main(["ask", "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।", "refund?"])
    assert sent == [None]
    assert "--lang multi" in capsys.readouterr().err


def test_auto_is_silent_on_english(monkeypatch, capsys):
    pytest.importorskip("laya_mlx")
    fake_server(monkeypatch)
    cli.main(["ask", "I was charged twice, please refund", "refund?"])
    assert "--lang" not in capsys.readouterr().err


@pytest.mark.parametrize("command", [
    'ls -la .config/ 2>/dev/null && echo "--- plugins ---" && ls -la .config',
    'ls -la src/ && echo "--- tests ---" && ls -la tests',
    'ls -la build/ 2>/dev/null && echo "--- assets ---" && ls -la build',
])
def test_shell_commands_do_not_trip_the_language_warning(monkeypatch, capsys, command):
    """Shell that laya's word-frequency guesser reads as French, the shape real commands took (41 of
    3,000 were flagged, §29). The first assert keeps the test honest: if the guesser stops misreading
    them, there is nothing left for the filter to catch and this would pass for the wrong reason."""
    pytest.importorskip("laya_mlx")
    from verdict import inputs

    inputs.language("warm up")
    assert not inputs._lang.analyse(command)["is_english"], "the guesser no longer misreads this"
    fake_server(monkeypatch)
    # A question about what the command shows: a consequence-shaped one ("risky?") is refused
    # before the language check runs, which is how this test once passed without testing anything.
    assert cli.main(["ask", json.dumps({"command": command}),
                     "Does `command` list files?"]) == 0
    assert "--lang" not in capsys.readouterr().err


@pytest.mark.parametrize("text", [
    "Der Kunde wurde zweimal belastet, bitte erstatten Sie das Geld (Rechnung #4411).",
    '{"message": "Der Kunde wurde zweimal belastet, bitte erstatten"}',
    "Le client a été facturé deux fois : remboursez-le aujourd'hui !",
])
def test_non_english_prose_still_warns(monkeypatch, capsys, text):
    pytest.importorskip("laya_mlx")
    fake_server(monkeypatch)
    cli.main(["ask", text, "refund?"])
    assert "--lang multi" in capsys.readouterr().err
