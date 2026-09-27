"""Further named laya checkpoints (config.py's [model.extra]), asked for by name the same way
`multilingual` already is (test_multilingual.py). Unlike multilingual, there is no dedicated
config key or lang-detection behaviour behind these: they are plain alternates a request selects
by name, loaded on first use and cached, never at startup."""
from __future__ import annotations

import sys
import types

import pytest
from fastapi.testclient import TestClient

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


def test_default_request_never_loads_an_extra_checkpoint(fake_laya):
    backend = MlxBackend("main", extra_checkpoints={"support": "support-model"})
    r = TestClient(create_app(backend)).post("/v1/decide", json={"state": "hi", "questions": Q})
    assert r.status_code == 200
    assert set(fake_laya) == {"main"}


def test_a_named_extra_checkpoint_loads_once_and_answers_from_it(fake_laya):
    backend = MlxBackend("main", extra_checkpoints={"support": "support-model"})
    app = TestClient(create_app(backend))
    for _ in range(2):
        r = app.post("/v1/decide", json={"state": "hi", "questions": Q, "model": "support"})
        assert r.json()["model"] == "laya-mlx:support-model"
    assert fake_laya["support-model"].calls == 2


def test_two_extra_checkpoints_load_and_cache_independently(fake_laya):
    backend = MlxBackend("main", extra_checkpoints={"support": "support-model", "legal": "legal-model"})
    app = TestClient(create_app(backend))
    app.post("/v1/decide", json={"state": "hi", "questions": Q, "model": "support"})
    app.post("/v1/decide", json={"state": "hi", "questions": Q, "model": "legal"})
    assert set(fake_laya) == {"support-model", "legal-model"}, "neither request named the primary"


def test_an_unconfigured_extra_name_is_a_400(fake_laya):
    backend = MlxBackend("main")
    r = TestClient(create_app(backend)).post(
        "/v1/decide", json={"state": "hi", "questions": Q, "model": "support"})
    assert r.status_code == 400


def test_models_endpoint_lists_every_configured_extra_checkpoint(fake_laya):
    backend = MlxBackend("main", extra_checkpoints={"support": "support-model", "legal": "legal-model"})
    body = TestClient(create_app(backend)).get("/v1/models").json()
    names = {m["name"] for m in body["models"]}
    assert {"support", "legal"} <= names


def test_systemone_can_select_an_advertised_extra(fake_laya):
    backend = MlxBackend("main", extra_checkpoints={"support": "support-model"})
    app = TestClient(create_app(backend))
    r = app.post("/v1/systemone", json={"state": "hi", "questions": Q, "model": "support"})
    assert r.status_code == 200
    assert r.json()["model"] == "laya-mlx:support-model"
    assert set(fake_laya) == {"support-model"}


def test_systemone_refuses_misspelled_model(fake_laya):
    app = TestClient(create_app(MlxBackend("main")))
    r = app.post("/v1/systemone", json={"state": "hi", "questions": Q, "model": "suport"})
    assert r.status_code == 400
    assert not fake_laya
