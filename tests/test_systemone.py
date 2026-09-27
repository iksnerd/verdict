"""`POST /v1/systemone` and `GET /v1/models`: TypeSafe's Jev wire protocol, served locally.

The shapes are read from the official `typesafe-sdk` 0.7.1 wire models (generated from
api.typesafe.ai/openapi.json), so the official SDKs, LiteLLM's passthrough and anything else
written for Jev can point at `verdict serve` with `TYPESAFE_BASE_URL`. No model loads here."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from verdict import inputs, library
from verdict.api import create_app, to_decide_request
from verdict.backend import UniformBackend
from verdict.backend_mlx import MlxBackend

from test_mlx_backend import BODY as MLX_BODY, FakeAgent

QUESTIONS = {
    "intent": {"type": "choice", "instructions": "What do they want?",
               "criteria": {"refund": "money back", "info": None}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
    "spam": {"type": "noul", "instructions": "Is this spam?"},
}
JEV = {"state": {"message": "I was charged twice"}, "model": "jev-latest", "questions": QUESTIONS}


def uniform():
    return TestClient(create_app(UniformBackend()))


def test_answers_in_the_jev_shape():
    r = uniform().post("/v1/systemone", json=JEV)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"model", "answers", "usage"}
    assert set(body["answers"]) == set(QUESTIONS)
    assert {a["type"] for a in body["answers"].values()} == {"choice", "score", "noul"}
    assert isinstance(body["usage"]["input_tokens"], int)
    assert body["usage"]["output_tokens"] == 0


def test_the_sdk_strict_types_hold():
    """The SDK's answer models are `strict=True`: a float field fed an int fails validation."""
    a = uniform().post("/v1/systemone", json=JEV).json()["answers"]
    assert isinstance(a["spam"]["noul"], float)
    assert isinstance(a["urgency"]["score"], float)
    assert all(isinstance(p, float) for p in a["intent"]["probabilities"].values())


def test_model_is_required_as_jev_requires_it():
    body = {k: v for k, v in JEV.items() if k != "model"}
    assert uniform().post("/v1/systemone", json=body).status_code == 422


def test_answers_match_v1_decide():
    c = uniform()
    jev = c.post("/v1/systemone", json=JEV).json()["answers"]
    ours = c.post("/v1/decide", json={"state": JEV["state"], "questions": QUESTIONS}).json()["answers"]
    assert jev == ours


def test_v1_decide_output_is_unchanged():
    """Two consumers parse `decide`'s output (CLAUDE.md); `usage` belongs to the Jev route only."""
    body = uniform().post("/v1/decide", json={"state": "x", "questions": QUESTIONS}).json()
    assert set(body) == {"model", "answers"}


@pytest.mark.parametrize("q", [
    {"type": "noul"},                                                     # instructions optional
    {"type": "noul", "instructions": {"task": "Identify advertising."}},  # structured
    {"type": "choice", "instructions": ["pick", "one"], "criteria": {"a": None, "b": {"k": 1}}},
    {"type": "noul", "instructions": "?", "criteria": {"true": None, "false": "legit"}},
])
def test_accepts_every_question_form_the_sdk_sends(q):
    r = uniform().post("/v1/systemone", json={**JEV, "questions": {"q": q}})
    assert r.status_code == 200, r.text


def test_a_bearer_key_is_accepted_and_ignored():
    """The SDK will not run without a key. The server is localhost-only, so any key does."""
    r = uniform().post("/v1/systemone", json=JEV, headers={"Authorization": "Bearer local"})
    assert r.status_code == 200


@pytest.mark.parametrize("name,expected", [
    ("jev-latest", None), ("jev-preview", None), ("jev-1.13", None), ("anything", "anything"),
    ("english", "english"), ("multilingual", "multilingual"),
])
def test_model_names_map_to_our_checkpoints(name, expected):
    assert to_decide_request({**JEV, "model": name}).model == expected


def test_models_lists_the_jev_alias():
    body = uniform().get("/v1/models").json()
    names = [m["name"] for m in body["models"]]
    assert "jev-latest" in names
    for m in body["models"]:
        assert set(m) == {"name", "description", "release_date"}


def test_noul_answers_keep_confidence_though_jev_omits_it():
    """typesafe-sdk 0.7.1's own `NoulAnswer` wire model has only `type` and `noul` (verified
    against the cached wheel): Jev doesn't score noul confidence separately. verdict's routing
    needs it, so `NoulAnswer` (schema.py) keeps the field on purpose. The real SDK's default
    `extra="ignore"` drops it silently rather than failing, so this stays additive, not a break."""
    body = uniform().post("/v1/systemone", json=JEV).json()["answers"]
    assert set(body["spam"]) == {"type", "noul", "confidence"}


def test_mlx_backend_reports_the_tokens_the_model_read():
    agent = FakeAgent()
    c = TestClient(create_app(MlxBackend(agent=agent)))
    body = c.post("/v1/systemone", json={**MLX_BODY, "model": "jev-latest"}).json()
    assert body["usage"] == {"input_tokens": 40, "output_tokens": 0}


def test_missing_instructions_reach_laya_as_an_empty_string():
    """laya-mlx raises on a question with no `instructions` key; Jev treats it as optional."""
    agent = FakeAgent()
    c = TestClient(create_app(MlxBackend(agent=agent)))
    c.post("/v1/systemone", json={**MLX_BODY, "model": "jev-latest",
                                  "questions": {"decided": {"type": "noul"}}})
    assert agent.seen[1]["decided"]["instructions"] == ""


def test_structured_instructions_do_not_break_the_warnings():
    q = {"q": {"type": "noul", "instructions": {"task": "Is `field` there?"}}}
    assert library.lint(q) == []
    assert inputs.missing_fields({"other": 1}, q)
