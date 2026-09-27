from fastapi.testclient import TestClient

from verdict.api import create_app
from verdict.backend_mlx import MlxBackend

BODY = {
    "state": {"message": "shipped it"},
    "questions": {
        "kind": {"type": "choice", "instructions": "?", "criteria": {"action": "", "note": ""}},
        "level": {"type": "score", "instructions": "?", "criteria": ["low", "high"]},
        "decided": {"type": "noul", "instructions": "?"},
    },
}


class FakeAgent:
    """Shaped exactly like laya's Agent.predict output (read from laya/agent.py)."""

    def __init__(self):
        self.seen = None

    def predict(self, state, questions):
        self.seen = (state, questions)
        return {
            "model": "laya-rl-agent",
            "answers": {
                "kind": {
                    "type": "choice",
                    "choice": "action",
                    "probabilities": {"action": 0.7, "note": 0.3},
                    "confidence": 0.12,
                    "action": {"act_probability": 0.9},
                },
                "level": {
                    "type": "score",
                    "score": 0.6,
                    "legend": {"0": "low", "1": "high"},
                    "probabilities": {"0": 0.4, "1": 0.6},
                    "confidence": 0.03,
                    "action": {"act_probability": 0.5},
                },
                "decided": {"type": "noul", "noul": 0.81, "confidence": 0.81, "action": {"act_probability": 0.5}},
            },
            "usage": {"input_tokens": 40, "output_tokens": 0},
        }


def client(agent=None):
    return TestClient(create_app(MlxBackend(agent=agent or FakeAgent())))


def test_questions_are_passed_through_in_laya_shape():
    agent = FakeAgent()
    client(agent).post("/v1/decide", json=BODY)
    state, questions = agent.seen
    assert state == {"message": "shipped it"}
    assert questions["kind"] == {"type": "choice", "instructions": "?", "criteria": {"action": "", "note": ""}}
    # POST /v1/decide asks a new yes/no as a no/yes choice, as the CLI does (FINDINGS §38).
    assert questions["decided"] == {"type": "choice", "instructions": "?",
                                    "criteria": {"no": "", "yes": ""}}


def test_answers_are_mapped_back():
    r = client().post("/v1/decide", json=BODY).json()
    assert r["model"].startswith("laya-mlx")
    a = r["answers"]
    assert a["kind"]["choice"] == "action" and a["kind"]["probabilities"]["action"] == 0.7
    assert a["level"]["score"] == 0.6
    assert a["decided"]["noul"] == 0.81


def test_healthz_names_the_real_backend():
    assert client().get("/healthz").json()["backend"].startswith("laya-mlx")


def test_list_criteria_survive_the_round_trip():
    agent = FakeAgent()
    body = {"state": "x", "questions": {"kind": {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}}}
    client(agent).post("/v1/decide", json=body)
    assert agent.seen[1]["kind"]["criteria"] == ["a", "b"]


def test_load_options_are_passed_through_to_laya_mlx(monkeypatch):
    import sys
    import types

    seen = {}
    fake = types.ModuleType("laya_mlx")

    def load(model_id, **kw):
        seen["id"], seen["kw"] = model_id, kw
        return FakeAgent()

    fake.load = load
    monkeypatch.setitem(sys.modules, "laya_mlx", fake)
    b = MlxBackend("some/model", compile=True, pad_to_multiple=16, batch_size=64)
    b.engine.agent  # loading is lazy; touching the property is what loads
    assert seen["id"] == "some/model"
    assert seen["kw"] == {"compile": True, "pad_to_multiple": 16, "batch_size": 64}


def test_defaults_pass_no_options(monkeypatch):
    import sys
    import types

    seen = {}
    fake = types.ModuleType("laya_mlx")
    fake.load = lambda model_id, **kw: seen.update(kw=kw) or FakeAgent()
    monkeypatch.setitem(sys.modules, "laya_mlx", fake)
    MlxBackend("m").engine.agent
    assert seen["kw"] == {}


def test_structured_score_levels_come_back_in_the_legend():
    """Jev's Score contract allows object or list level descriptions, and laya returns them as
    given in the legend. A str-only legend turned that valid request into a 500."""
    levels = [{"label": "low", "means": "no action"}, ["high", "act now"]]

    class StructuredAgent(FakeAgent):
        def predict(self, state, questions):
            out = super().predict(state, questions)
            out["answers"]["level"]["legend"] = {str(i): v for i, v in enumerate(levels)}
            return out

    body = {**BODY, "questions": {**BODY["questions"],
                                  "level": {"type": "score", "criteria": levels}}}
    r = client(StructuredAgent()).post("/v1/decide", json=body)
    assert r.status_code == 200
    assert r.json()["answers"]["level"]["legend"] == {"0": levels[0], "1": levels[1]}
