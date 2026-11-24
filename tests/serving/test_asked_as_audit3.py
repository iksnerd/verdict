"""Audit 3: a yes/no asked as a choice keeps its `asked_as` marker on every path, and the Jev path
leaves it out. No model."""
from __future__ import annotations

from fastapi.testclient import TestClient

from verdict import library
from verdict.api import create_app
from verdict.schema import ChoiceAnswer, DecideRequest, DecideResponse, NoulAnswer, Usage

PLAIN = {"type": "noul", "instructions": "Is `message` a refund request?"}
BODY = {"state": {"message": "charged twice"}, "questions": {"refund": PLAIN}}


class Recording:
    name = "recording"

    def decide(self, request):
        answers = {}
        for qid, q in request.questions.items():
            if q.type == "noul":
                answers[qid] = NoulAnswer(noul=0.5, confidence=0.5)
                continue
            keys = q.option_keys()
            answers[qid] = ChoiceAnswer(choice=keys[0], probabilities={k: 1 / len(keys) for k in keys},
                                        confidence=0.0)
        return DecideResponse(model=self.name, answers=answers, usage=Usage(input_tokens=0))


def test_library_ask_keeps_asked_as():
    out = library.ask(Recording(), DecideRequest(state={"message": "x"}, questions={"refund": PLAIN}))
    assert out.answers["refund"].asked_as == "choice"


def test_http_decide_carries_asked_as():
    r = TestClient(create_app(Recording())).post("/v1/decide", json=BODY)
    assert r.json()["answers"]["refund"]["asked_as"] == "choice"


def test_http_decide_with_yesno_true_has_no_marker():
    r = TestClient(create_app(Recording())).post("/v1/decide?yesno=true", json=BODY)
    assert "asked_as" not in r.json()["answers"]["refund"]


def test_jev_path_drops_asked_as():
    r = TestClient(create_app(Recording())).post(
        "/v1/systemone", json={**BODY, "model": "jev-latest"})
    assert r.status_code == 200
    assert set(r.json()["answers"]["refund"]) == {"type", "noul", "confidence"}


def test_unset_asked_as_is_not_serialized():
    assert "asked_as" not in NoulAnswer(noul=0.5).model_dump(exclude_none=True)
