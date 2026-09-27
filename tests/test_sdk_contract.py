"""Cross-backend contracts using transport fakes; never sends data to a hosted model."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from verdict.api import create_app
from verdict.backend_mlx import MlxBackend
from verdict.backend_systemone import SystemOneBackend
from verdict.schema import DecideRequest

Q = {"q": {"type": "noul", "instructions": "Is it urgent?"}}


def request(model=None):
    return DecideRequest(state="hello", questions=Q, model=model)


def test_proxy_honors_model_and_does_not_invent_confidence_or_usage():
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body)
        return httpx.Response(200, json={"model": body["model"], "answers": {
            "q": {"type": "noul", "noul": .9}}})

    with SystemOneBackend("https://api.typesafe.ai", transport=httpx.MockTransport(handler)) as backend:
        c = TestClient(create_app(backend))
        for path in ("/v1/decide", "/v1/systemone"):
            r = c.post(path, json={"state": "hello", "questions": Q, "model": "jev-1.13.0"})
            assert r.status_code == 200, r.text
            assert r.json()["model"] == "jev-1.13.0"
            assert r.json()["answers"]["q"] == {"type": "noul", "noul": .9}
            if path.endswith("systemone"):
                assert r.json()["usage"] == {}
        assert all(b["model"] == "jev-1.13.0" and b["questions"] == Q for b in calls)


def test_capabilities_distinguish_limits_without_loading_models():
    local = TestClient(create_app(MlxBackend("unloaded", budget=128)))
    info = local.get("/v1/capabilities").json()
    assert info["family"] == "laya"
    assert info["state_token_budget"] == 128
    assert info["recommended_choice_options"] == 20
    assert info["max_choice_options"] is None
    assert info["max_request_tokens"] is None
    assert local.get("/v1/capabilities?model=jev-latest").json()["family"] == "laya"
    assert local.get("/v1/capabilities?model=typo").status_code == 400
    with SystemOneBackend("https://api.typesafe.ai") as backend:
        info = TestClient(create_app(backend)).get("/v1/capabilities").json()
        assert info["family"] == "jev"
        assert info["max_choice_options"] == 255
        assert info["max_score_levels"] == 10
        assert info["noul_confidence"] == "absent"
    with SystemOneBackend("http://localhost:8000") as backend:
        assert backend.capabilities().family == "systemone"
        assert backend.capabilities().max_choice_options is None


def test_jev_limit_is_not_layas_recommendation():
    calls = []

    def handler(req):
        calls.append(req)
        q = json.loads(req.content)["questions"]["q"]
        keys = list(q["criteria"])
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"q": {
            "type": "choice", "choice": keys[0], "confidence": 0.,
            "probabilities": {k: 1 / len(keys) for k in keys}}}})

    with SystemOneBackend("https://api.typesafe.ai", transport=httpx.MockTransport(handler)) as backend:
        for n in (21, 255):
            backend.decide(DecideRequest(state="x", questions={"q": {
                "type": "choice", "instructions": "Choose", "criteria": {str(i): None for i in range(n)}}}))
        with pytest.raises(ValueError, match="255"):
            backend.decide(DecideRequest(state="x", questions={"q": {
                "type": "choice", "instructions": "Choose", "criteria": {str(i): None for i in range(256)}}}))
        assert len(calls) == 2


@pytest.mark.parametrize("answers", [{}, {"q": {"type": "noul", "noul": 1.2}},
    {"q": {"type": "noul", "noul": float("nan")}},
    {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 1.}, "confidence": 1.}},
    {"q": {"type": "noul", "noul": .7}, "extra": {"type": "noul", "noul": .2}}])
def test_invalid_upstream_answers_are_bad_gateway(answers):
    def handler(req):
        return httpx.Response(200, content=json.dumps({"model": "jev-1.13.0", "answers": answers}))
    with SystemOneBackend("https://api.typesafe.ai", transport=httpx.MockTransport(handler)) as backend:
        r = TestClient(create_app(backend)).post("/v1/decide", json={"state": "x", "questions": Q})
        assert r.status_code == 502


def test_retry_after_and_connection_lifecycle(monkeypatch):
    waits, calls = [], []
    monkeypatch.setattr("time.sleep", waits.append)

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "3"})
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": .9}}})

    with SystemOneBackend("https://api.typesafe.ai", transport=httpx.MockTransport(handler)) as backend:
        backend.decide(request())
        assert waits == [3.0]
    with pytest.raises(RuntimeError, match="closed"):
        backend.decide(request())


@pytest.mark.parametrize("status", [401, 403, 422])
def test_permanent_errors_are_not_retried(status):
    calls = []
    def handler(req):
        calls.append(req)
        return httpx.Response(status, text="private state must not leak")
    with SystemOneBackend("https://api.typesafe.ai", transport=httpx.MockTransport(handler)) as backend:
        with pytest.raises(ValueError, match=str(status)) as exc:
            backend.decide(request())
        assert "private" not in str(exc.value)
    assert len(calls) == 1


def test_score_limit_is_rejected_before_transport():
    def handler(req):
        pytest.fail("invalid question must not reach upstream")
    with SystemOneBackend("https://api.typesafe.ai", transport=httpx.MockTransport(handler)) as backend:
        c = TestClient(create_app(backend))
        r = c.post("/v1/decide", json={"state": "x", "questions": {"q": {
            "type": "score", "instructions": "Rate", "criteria": [str(i) for i in range(11)]}}})
        assert r.status_code == 422


@pytest.mark.parametrize("value,expected", [("4", 4.), ("invalid", None), ("NaN", None), ("-1", 0.)])
def test_retry_after_values(value, expected):
    from verdict.systemone import _retry_after
    assert _retry_after(value) == expected


def test_retry_after_http_date():
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime
    from verdict.systemone import _retry_after

    value = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=20))
    assert 18 <= _retry_after(value) <= 20


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"model":"jev","answers":{"q":{"noul":0.9}}}'])
def test_malformed_response_is_a_clean_error(body):
    with SystemOneBackend("https://api.typesafe.ai", transport=httpx.MockTransport(
            lambda req: httpx.Response(200, content=body))) as backend:
        with pytest.raises(ValueError, match="upstream"):
            backend.decide(request())


@pytest.mark.parametrize("patch", [
    {"probabilities": {"a": .5}},
    {"probabilities": {"a": .8, "b": .8}},
    {"probabilities": {"a": -.1, "b": 1.1}},
    {"choice": "unknown"}, {"choice": "b"}, {"confidence": float("inf")},
])
def test_inconsistent_choice_is_rejected(patch):
    answer = {"type": "choice", "choice": "a", "probabilities": {"a": .7, "b": .3},
              "confidence": .1, **patch}
    def handler(req):
        return httpx.Response(200, content=json.dumps({"model": "model", "answers": {"q": answer}}))
    with SystemOneBackend("http://localhost:8000", transport=httpx.MockTransport(handler)) as backend:
        with pytest.raises(ValueError, match="invalid SystemOne"):
            backend.decide(DecideRequest(state="x", questions={"q": {
                "type": "choice", "instructions": "Pick", "criteria": {"a": None, "b": None}}}))


@pytest.mark.parametrize("patch", [
    {"score": 3.}, {"score": .3}, {"legend": {"0": "wrong", "1": "high"}},
])
def test_inconsistent_score_is_rejected(patch):
    answer = {"type": "score", "score": .8, "probabilities": {"0": .2, "1": .8},
              "legend": {"0": "low", "1": "high"}, "confidence": .2, **patch}
    def handler(req):
        return httpx.Response(200, json={"model": "model", "answers": {"q": answer}})
    with SystemOneBackend("http://localhost:8000", transport=httpx.MockTransport(handler)) as backend:
        with pytest.raises(ValueError, match="invalid SystemOne"):
            backend.decide(DecideRequest(state="x", questions={"q": {
                "type": "score", "instructions": "Rate", "criteria": ["low", "high"]}}))


def test_laya_context_is_per_question_not_a_shared_request_budget():
    class Agent:
        cfg = {"max_len": 1024}
    caps = MlxBackend("model", agent=Agent(), budget=128).capabilities()
    assert caps.max_request_tokens is None
    assert caps.max_state_and_question_tokens == 1024


def test_jev_preserves_long_structured_state_and_noul_question():
    state = {"text": "hello " * 2000, "nested": {"k": [1, 2, 3]}}
    def handler(req):
        body = json.loads(req.content)
        assert body["state"] == state
        assert body["questions"] == Q
        return httpx.Response(200, json={"model": "jev-1.13.0", "answers": {"q": {"type": "noul", "noul": .9}},
                                         "usage": {"input_tokens": 2048, "output_tokens": 20}})
    with SystemOneBackend("https://api.typesafe.ai", transport=httpx.MockTransport(handler)) as backend:
        result = backend.decide(DecideRequest(state=state, questions=Q))
        assert result.usage.output_tokens == 20
