"""Audit 4: batch 207 on any exception, an honest /healthz, /v1/models dedupe and fallback header."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from verdict.api import create_app
from verdict.schema import DecideResponse, NoulAnswer, ScoreAnswer
from verdict.backends.base import UniformBackend


class Flaky:
    name = "flaky"
    proxy = False

    def __init__(self, exc, fail_at=2):
        self.calls, self.exc, self.fail_at = 0, exc, fail_at

    def decide(self, request):
        self.calls += 1
        if self.calls - 1 == self.fail_at:
            raise self.exc
        answers = {}
        for qid, q in request.questions.items():
            if q.type == "score":
                keys = q.option_keys()
                answers[qid] = ScoreAnswer(score=0.0, legend={k: k for k in keys},
                                           probabilities={k: 1 / len(keys) for k in keys},
                                           confidence=0.5)
            else:
                answers[qid] = NoulAnswer(noul=0.5, confidence=0.5)
        return DecideResponse(model=self.name, answers=answers)


def _post(backend, n=4):
    return TestClient(create_app(backend), raise_server_exceptions=False).post(
        "/v1/route/batch", json={"prompts": [f"p{i}" for i in range(n)]})


def test_a_runtime_error_mid_batch_is_a_207_keeping_earlier_results():
    r = _post(Flaky(RuntimeError("out of memory")))
    assert r.status_code == 207
    results = r.json()["results"]
    assert "branch" in results[0] and "branch" in results[1]
    assert "out of memory" in results[2]["error"] and "branch" not in results[2]
    assert "not attempted" in results[3]["error"]


def test_a_runtime_error_on_the_first_prompt_is_not_a_207():
    r = _post(Flaky(RuntimeError("boom"), fail_at=0))
    assert r.status_code == 500


def test_keyboard_interrupt_is_not_swallowed():
    with pytest.raises(KeyboardInterrupt):
        TestClient(create_app(Flaky(KeyboardInterrupt(), fail_at=1))).post(
            "/v1/route/batch", json={"prompts": ["a", "b", "c"]})


def test_healthz_not_ready_after_close():
    class Closed(Flaky):
        _closed = True
    body = TestClient(create_app(Closed(None, fail_at=99))).get("/healthz").json()
    assert body["ready"] is False and "closed" in body["reason"] and body["status"] == "closed"


def test_healthz_ready_has_no_reason():
    body = TestClient(create_app(Flaky(None, fail_at=99))).get("/healthz").json()
    assert body == {"status": "ok", "backend": "flaky", "ready": True}


def test_healthz_uniform_says_why():
    body = TestClient(create_app(UniformBackend())).get("/healthz").json()
    assert body["ready"] is False and body["reason"]


class Extra(Flaky):
    extra_checkpoints = {"english": "org/other", "jev-latest": "org/jev", "mine": "org/mine"}
    multilingual_id = "org/multi"

    def known_models(self):
        return {"english", "jev-latest", "mine"}


def test_models_has_no_duplicate_card_names():
    names = [m["name"] for m in TestClient(create_app(Extra(None))).get("/v1/models").json()["models"]]
    assert len(names) == len(set(names))


def test_models_fallback_is_flagged_for_a_proxy_whose_upstream_failed():
    class Down(Flaky):
        proxy = True

        def models(self):
            return None
    r = TestClient(create_app(Down(None))).get("/v1/models")
    assert r.headers["X-Verdict-Models"] == "fallback"


def test_models_of_a_local_backend_are_not_flagged():
    r = TestClient(create_app(Flaky(None))).get("/v1/models")
    assert "X-Verdict-Models" not in r.headers
