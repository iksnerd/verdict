"""The `Backend` contract: `name` and `decide` are all a backend needs, `proxy` says what kind it is,
and everything else (`models`, `capabilities`, `known_models`, `engine`) is optional with one
documented default. A new backend that implements only `decide` must work behind every front door."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from verdict import Verdict
from verdict.api import create_app
from verdict.backends.base import UniformBackend
from verdict.backends.jev import JevBackend
from verdict.backends.mlx import MlxBackend
from verdict.backends.systemone import SystemOneBackend
from verdict.schema import DecideResponse, NoulAnswer, Usage

BODY = {"state": {"message": "I was charged twice"}, "model": "jev-latest",
        "questions": {"spam": {"type": "noul", "instructions": "Is this spam?"}}}


class DecideOnly:
    """Everything the Protocol requires, nothing else. Remembers the model each request named."""
    name = "decide-only"

    def __init__(self):
        self.models_seen = []

    def decide(self, request):
        self.models_seen.append(request.model)
        answers = {qid: NoulAnswer(noul=0.5, confidence=0.5) for qid in request.questions}
        return DecideResponse(model=self.name, answers=answers, usage=Usage(input_tokens=0))


class Forwarder(DecideOnly):
    name = "forwarder"
    proxy = True


@pytest.mark.parametrize("backend, is_proxy", [
    (UniformBackend, False), (MlxBackend, False), (JevBackend, True), (SystemOneBackend, True)])
def test_every_shipped_backend_declares_whether_it_is_a_proxy(backend, is_proxy):
    assert backend.proxy is is_proxy


def test_a_backend_with_only_decide_answers_over_every_endpoint():
    app = TestClient(create_app(DecideOnly()))
    assert app.post("/v1/decide", json={"state": "x", "questions": BODY["questions"]}).status_code == 200
    assert app.post("/v1/systemone", json=BODY).status_code == 200
    assert app.post("/v1/route", json={"prompt": "what does chmod 755 mean"}).status_code == 200
    names = [card["name"] for card in app.get("/v1/models").json()["models"]]
    assert "english" in names
    caps = app.get("/v1/capabilities").json()
    assert (caps["family"], caps["model"]) == ("unknown", "decide-only")


def test_a_backend_with_only_decide_works_in_process():
    with Verdict(backend=DecideOnly()) as v:
        assert v.models() == [{"name": "decide-only"}]
        caps = v.capabilities()
        assert (caps.family, caps.model) == ("unknown", "decide-only")


def test_a_local_backend_drops_a_jev_alias_and_a_proxy_forwards_it():
    local, forwarder = DecideOnly(), Forwarder()
    TestClient(create_app(local)).post("/v1/systemone", json=BODY)
    TestClient(create_app(forwarder)).post("/v1/systemone", json=BODY)
    assert local.models_seen == [None]
    assert forwarder.models_seen == ["jev-latest"]


def test_a_proxy_refuses_routing_because_the_thresholds_are_laya_s():
    r = TestClient(create_app(Forwarder())).post("/v1/route", json={"prompt": "hello"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "refused"
