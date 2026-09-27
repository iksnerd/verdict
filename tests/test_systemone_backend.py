"""SystemOneBackend: any /v1/systemone-compatible server as a Backend. Kev, a real Jev, or
another verdict all answer this shape, so this needs no translation layer beyond typing the
raw JSON -- reuses bench.py's already-tested systemone_asker for the HTTP itself (auth, retries)."""
from __future__ import annotations

import json

import httpx

from fastapi.testclient import TestClient

from verdict.api import create_app
from verdict.backend_systemone import SystemOneBackend
from verdict.schema import DecideRequest

Q = {
    "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "refunds", "tech": "bugs"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
    "spam": {"type": "noul", "instructions": "Is this spam?"},
}


def kev_transport(answers, model="kev-0.8b", usage=None):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/systemone"
        return httpx.Response(200, json={
            "model": model, "usage": usage or {"input_tokens": 12, "output_tokens": 0},
            "answers": answers,
        })
    return httpx.MockTransport(handler)


def request():
    return DecideRequest(state={"message": "I was charged twice"}, questions=Q)


def test_choice_and_score_pass_through_confidence_unchanged():
    backend = SystemOneBackend("https://kev.example", "kev-latest", "k", transport=kev_transport({
        "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1}, "confidence": 0.8},
        "urgency": {"type": "score", "score": 1.2, "legend": {"0": "low", "1": "mid", "2": "high"},
                   "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}, "confidence": 0.5},
        "spam": {"type": "noul", "noul": 0.2, "confidence": 0.6},
    }))
    r = backend.decide(request())
    assert r.answers["dept"].choice == "billing" and r.answers["dept"].confidence == 0.8
    assert r.answers["urgency"].score == 1.2 and r.answers["urgency"].confidence == 0.5
    assert r.answers["spam"].confidence == 0.6


def test_noul_confidence_stays_absent_when_the_server_omits_it():
    """A statistic borrowed from another provider is not a reported confidence score."""
    backend = SystemOneBackend("https://kev.example", "kev-latest", "k", transport=kev_transport({
        "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0, "tech": 0.0}, "confidence": 1.0},
        "urgency": {"type": "score", "score": 0.0, "legend": {"0": "low", "1": "mid", "2": "high"},
                   "probabilities": {"0": 1.0, "1": 0.0, "2": 0.0}, "confidence": 1.0},
        "spam": {"type": "noul", "noul": 0.9},
    }))
    r = backend.decide(request())
    assert r.answers["spam"].confidence is None


def test_the_reported_model_is_the_upstream_checkpoint_not_our_own_name():
    backend = SystemOneBackend("https://kev.example", "kev-latest", "k", transport=kev_transport({
        "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0, "tech": 0.0}, "confidence": 1.0},
        "urgency": {"type": "score", "score": 0.0, "legend": {"0": "low", "1": "mid", "2": "high"},
                   "probabilities": {"0": 1.0, "1": 0.0, "2": 0.0}, "confidence": 1.0},
        "spam": {"type": "noul", "noul": 0.9, "confidence": 0.8},
    }, model="kev-0.8b@r15"))
    r = backend.decide(request())
    assert r.model == "kev-0.8b@r15"


def test_usage_passes_through():
    backend = SystemOneBackend("https://kev.example", "kev-latest", "k", transport=kev_transport({
        "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 1.0, "tech": 0.0}, "confidence": 1.0},
        "urgency": {"type": "score", "score": 0.0, "legend": {"0": "low", "1": "mid", "2": "high"},
                   "probabilities": {"0": 1.0, "1": 0.0, "2": 0.0}, "confidence": 1.0},
        "spam": {"type": "noul", "noul": 0.9, "confidence": 0.8},
    }, usage={"input_tokens": 47, "output_tokens": 0}))
    r = backend.decide(request())
    assert r.usage.input_tokens == 47


def test_the_questions_sent_upstream_are_plain_wire_dicts():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={
            "model": "kev-0.8b", "usage": {"input_tokens": 1, "output_tokens": 0},
            "answers": {"spam": {"type": "noul", "noul": 0.5, "confidence": 0.0}}})

    backend = SystemOneBackend("https://kev.example", "kev-latest", "k",
                               transport=httpx.MockTransport(handler))
    backend.decide(DecideRequest(state="x", questions={"spam": {"type": "noul", "instructions": "?"}}))
    assert calls[0]["questions"] == {"spam": {"type": "noul", "instructions": "?"}}


def upstream_models_transport(cards, models_path="/v1/models"):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == models_path and req.method == "GET"
        return httpx.Response(200, json={"models": cards})
    return httpx.MockTransport(handler)


CARDS = [
    {"name": "kev-latest", "description": "the served checkpoint", "release_date": "2026-09-24"},
    {"name": "kev-0.8b", "description": "Kev-0.8B", "release_date": "2026-09-24"},
]


def test_models_forwards_the_upstream_list():
    backend = SystemOneBackend("https://kev.example", "kev-latest", "k",
                               transport=upstream_models_transport(CARDS))
    assert backend.models() == CARDS


def test_v1_models_route_lists_the_upstream_checkpoints_not_our_own_defaults():
    """The hardcoded jev-latest/english/multilingual cards are laya/Jev-specific; a proxied
    backend has its own real model list and should report that instead."""
    backend = SystemOneBackend("https://kev.example", "kev-latest", "k",
                               transport=upstream_models_transport(CARDS))
    body = TestClient(create_app(backend)).get("/v1/models").json()
    assert body["models"] == CARDS


def test_a_persistently_unreachable_upstream_is_a_502_not_a_500(monkeypatch):
    """`decide()` raises the plain ValueError bench.systemone_asker raises (its own tests cover
    the retry-and-give-up path); the HTTP seam here is only about not letting that surface as an
    unhandled 500 with a traceback when verdict serve is fronting this backend."""
    monkeypatch.setattr("time.sleep", lambda s: None)

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated: server never answered")

    backend = SystemOneBackend("https://kev.example", "kev-latest", "k",
                               transport=httpx.MockTransport(handler))
    r = TestClient(create_app(backend)).post("/v1/decide", json={"state": "x", "questions": Q})
    assert r.status_code == 502
    assert "loading" in r.json()["detail"]


def test_models_degrades_to_empty_when_the_upstream_is_unreachable(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = SystemOneBackend("https://kev.example", "kev-latest", "k",
                               transport=httpx.MockTransport(handler))
    assert backend.models() == []
    body = TestClient(create_app(backend)).get("/v1/models").json()
    assert body["models"] == []


def test_route_endpoints_turn_an_upstream_failure_into_a_502(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    backend = SystemOneBackend("https://kev.example", "kev-latest", "k",
                               transport=httpx.MockTransport(lambda req: httpx.Response(503)))
    app = TestClient(create_app(backend), raise_server_exceptions=False)
    for path, body in (("/v1/route", {"prompt": "x"}), ("/v1/route/batch", {"prompts": ["x"]})):
        r = app.post(path, json=body)
        assert r.status_code == 502, (path, r.status_code, r.text)
        assert "detail" in r.json()
