"""JevBackend: hosted Jev through the official `typesafe-sdk`.

The seam is the Backend's `decide()` and the HTTP API over it, with the SDK's own mock transport
standing in for api.typesafe.ai. Answer shapes follow the SDK's wire models (the same source as
tests/test_systemone.py); no network is touched.
"""
from __future__ import annotations

import json

import pytest

httpx2 = pytest.importorskip("httpx2")
typesafe_sdk = pytest.importorskip("typesafe_sdk")

from fastapi.testclient import TestClient  # noqa: E402

from verdict.api import create_app  # noqa: E402
from verdict.backends.jev import JevBackend  # noqa: E402
from verdict.errors import QuestionError  # noqa: E402
from verdict.schema import DecideRequest  # noqa: E402
from verdict.systemone import UpstreamError  # noqa: E402

Q = {
    "dept": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "refunds", "tech": "bugs"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
    "spam": {"type": "noul", "instructions": "Is this spam?"},
}

ANSWERS = {
    "dept": {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.9, "tech": 0.1},
             "confidence": 0.8},
    "urgency": {"type": "score", "score": 1.2, "legend": {"0": "low", "1": "mid", "2": "high"},
                "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}, "confidence": 0.5},
    "spam": {"type": "noul", "noul": 0.2},
}


def fast():
    """No sleeping between retries."""
    return typesafe_sdk.RetryPolicy(max_retries=2, backoff_initial=0, backoff_max=0)


def backend(handler, **kw):
    kw.setdefault("retry", fast())
    return JevBackend("jev-latest", api_key="sk-test", base_url="https://jev.example",
                      transport=httpx2.MockTransport(handler), **kw)


def ok(answers=ANSWERS, model="jev-1.13.0", seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx2.Response(200, json={"model": model, "answers": answers,
                                          "usage": {"input_tokens": 12, "output_tokens": 0}})
    return handler


def req(model=None, questions=Q):
    return DecideRequest(state={"message": "I was charged twice"}, questions=questions, model=model)


def test_answers_come_back_in_verdicts_shape():
    r = backend(ok()).decide(req())
    assert r.model == "jev-1.13.0"
    assert r.answers["dept"].choice == "billing" and r.answers["dept"].confidence == 0.8
    assert r.answers["urgency"].score == 1.2
    assert r.answers["urgency"].legend == {"0": "low", "1": "mid", "2": "high"}
    assert r.answers["spam"].noul == 0.2 and r.answers["spam"].confidence is None
    assert r.usage.input_tokens == 12


def test_the_request_is_the_sdks_typed_wire_form():
    seen = []
    backend(ok(seen=seen)).decide(req())
    sent = seen[0]
    assert sent.url.path == "/v1/systemone"
    assert sent.headers["authorization"] == "Bearer sk-test"
    body = json.loads(sent.content)
    assert body["model"] == "jev-latest" and body["state"] == {"message": "I was charged twice"}
    assert body["questions"]["spam"] == {"type": "noul", "instructions": "Is this spam?"}
    assert body["questions"]["urgency"]["criteria"] == ["low", "mid", "high"]


def test_a_requests_model_wins_including_a_pinned_version():
    seen = []
    backend(ok(seen=seen)).decide(req(model="jev-1.13.0"))
    assert json.loads(seen[0].content)["model"] == "jev-1.13.0"


def test_a_bad_key_is_named_without_echoing_the_body_or_the_key():
    def handler(request):
        return httpx2.Response(401, json={"error": "invalid key sk-test, state was: hello"})
    with pytest.raises(UpstreamError) as info:
        backend(handler).decide(req())
    text = str(info.value)
    assert "401" in text and "sk-test" not in text and "hello" not in text


def test_an_upstream_question_rejection_is_the_callers_to_fix_and_names_the_request():
    def handler(request):
        return httpx2.Response(422, json={"detail": "bad question about hello"},
                               headers={"x-typesafe-request-id": "req_123"})
    with pytest.raises(QuestionError) as info:
        backend(handler).decide(req())
    text = str(info.value)
    assert "422" in text and "req_123" in text and "hello" not in text


def test_overload_is_retried_then_reported_as_upstream_failure():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(529, json={"error": "overloaded"})
    with pytest.raises(UpstreamError, match="529"):
        backend(handler).decide(req())
    assert len(calls) == 3, "one try and two retries, as the SDK's policy says"


def test_a_malformed_answer_is_refused_not_passed_on():
    bad = {**ANSWERS, "dept": {**ANSWERS["dept"], "choice": "tech"}}  # not the highest probability
    with pytest.raises(UpstreamError, match="invalid"):
        backend(ok(bad)).decide(req())


def test_jevs_own_limits_are_checked_before_any_request():
    called = []
    many = {"q": {"type": "choice", "instructions": "?",
                  "criteria": {f"o{i}": "" for i in range(256)}}}

    def handler(request):
        called.append(1)
        return httpx2.Response(500)
    with pytest.raises(QuestionError, match="255"):
        backend(handler).decide(req(questions=many))
    assert called == []


def test_models_lists_what_the_service_lists():
    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx2.Response(200, json={"models": [{"name": "jev-latest", "description": "x",
                                                      "release_date": "2026-09-01"}]})
    assert backend(handler).models()[0]["name"] == "jev-latest"


def test_the_http_api_over_it_answers_and_maps_upstream_failure_to_502():
    app = TestClient(create_app(backend(ok({"spam": ANSWERS["spam"]}))))
    body = {"state": {"message": "hi"}, "questions": {"spam": Q["spam"]}}
    assert app.post("/v1/decide", json=body).status_code == 200

    def down(request):
        return httpx2.Response(503, json={})
    r = TestClient(create_app(backend(down))).post("/v1/decide", json=body)
    assert r.status_code == 502


def test_the_name_says_it_is_hosted_jev():
    assert backend(ok()).name == "jev:jev-latest@https://jev.example"


# bench and examples reach hosted Jev through the SDK too (bench.systemone_asker).

def hosted_asker(handler, **kw):
    from verdict.evaluation import bench
    return bench.systemone_asker("https://api.typesafe.ai", "jev-latest", "sk-test", backoff=0,
                                 transport=httpx2.MockTransport(handler), **kw)


def test_the_bench_asker_for_hosted_jev_is_the_sdk_backend():
    from verdict.backends.jev import JevAsker
    asker = hosted_asker(ok({"spam": ANSWERS["spam"]}))
    assert isinstance(asker, JevAsker)
    raw = asker({"text": "win a prize"}, {"spam": Q["spam"]})
    assert raw["answers"]["spam"]["noul"] == 0.2 and raw["usage"]["input_tokens"] == 12
    asker.close()


def test_the_bench_asker_retries_overload_through_the_sdk_then_names_the_status():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(529, json={})
    asker = hosted_asker(handler, attempts=3)
    with pytest.raises(UpstreamError, match="529"):
        asker({"text": "x"}, {"spam": Q["spam"]})
    assert len(calls) == 3


def test_another_server_keeps_the_dependency_free_client():
    from verdict.evaluation import bench
    from verdict.systemone import SystemOneClient
    asker = bench.systemone_asker("https://kev.example", "kev-latest", "k")
    assert isinstance(asker, SystemOneClient)
    asker.close()


def test_without_the_sdk_hosted_jev_falls_back_to_the_plain_client(monkeypatch):
    import sys
    from verdict.evaluation import bench
    from verdict.systemone import SystemOneClient
    monkeypatch.setitem(sys.modules, "verdict.backends.jev", None)  # import raises ImportError
    asker = bench.systemone_asker("https://api.typesafe.ai", "jev-latest", "k")
    assert isinstance(asker, SystemOneClient)
    asker.close()


# The name is what logs, /healthz and scorecards say; it must be where the states actually go.

def test_the_name_follows_the_sdks_own_base_url_resolution(monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://127.0.0.1:8799/")
    b = JevBackend("jev-latest", api_key="sk-test", transport=httpx2.MockTransport(ok()))
    assert b.base_url == "http://127.0.0.1:8799" and b.name == "jev:jev-latest@http://127.0.0.1:8799"


def test_an_explicit_base_url_beats_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://127.0.0.1:8799")
    b = JevBackend("jev-latest", api_key="sk-test", base_url="https://jev.example",
                   transport=httpx2.MockTransport(ok()))
    assert b.base_url == "https://jev.example"


def test_an_empty_environment_value_is_ignored_as_the_sdk_ignores_it(monkeypatch):
    monkeypatch.setenv("TYPESAFE_BASE_URL", "  ")
    b = JevBackend("jev-latest", api_key="sk-test", transport=httpx2.MockTransport(ok()))
    assert b.base_url == "https://api.typesafe.ai"
