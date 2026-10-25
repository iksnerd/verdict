"""A yes/no may name its two sides: `{"labels": {"false": "calm", "true": "urgent"}}`, as laya's
`noul` takes since laya-mlx 0.3.0 (the port in mizorewww/laya-mlx#17 was closed because upstream
shipped it). verdict passes it through as written. It is not measured: describing the sides of a
yes/no hurt (§22), so the docs say so and nothing here recommends it."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from verdict import Verdict, cli, client
from verdict.api import create_app
from verdict.backends.capabilities import jev_capabilities
from verdict.cli import inference
from verdict.errors import QuestionError
from verdict.schema import (
    ChoiceAnswer,
    DecideRequest,
    DecideResponse,
    NoulAnswer,
    Usage,
    laya_questions,
)
from verdict.sdk import Noul
from fakes import FakeEngine

LABELS = {"false": "calm", "true": "urgent"}
ASKED = {"type": "noul", "instructions": "Is `message` urgent?", "labels": LABELS}
PLAIN = {"type": "noul", "instructions": "Is `message` a refund request?"}
LAYA_REFUSAL = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"


class Recording:
    """A local backend that records the questions it was asked and answers every one 0.5."""
    name = "recording"

    def __init__(self):
        self.asked = []

    def decide(self, request):
        self.asked.append(request.questions)
        answers = {qid: NoulAnswer(noul=0.5, confidence=0.5)
                   for qid, q in request.questions.items() if q.type == "noul"}
        for qid, q in request.questions.items():
            if q.type != "noul":
                keys = q.option_keys()
                answers[qid] = ChoiceAnswer(choice=keys[0], probabilities={k: 1 / len(keys) for k in keys},
                                            confidence=0.0)
        return DecideResponse(model=self.name, answers=answers, usage=Usage(input_tokens=0))


def test_a_yes_no_with_labels_is_passed_to_laya_as_written():
    assert laya_questions({"q": ASKED})["q"]["labels"] == LABELS


@pytest.mark.parametrize("labels", [
    {"true": "urgent"},
    {"false": "calm", "true": "calm"},
    {"false": "", "true": "urgent"},
    {"false": "calm", "true": "urgent", "maybe": "unsure"},
    ["calm", "urgent"],
])
def test_labels_that_laya_would_refuse_are_refused_here_with_its_wording(labels):
    with pytest.raises(QuestionError, match=LAYA_REFUSAL):
        laya_questions({"q": {**ASKED, "labels": labels}})


def test_over_http_a_labelled_yes_no_is_asked_as_written_and_a_plain_one_is_still_a_choice():
    backend = Recording()
    r = TestClient(create_app(backend)).post(
        "/v1/decide", json={"state": {"message": "charged twice"},
                            "questions": {"urgent": ASKED, "refund": PLAIN}})
    assert r.status_code == 200
    asked = backend.asked[0]
    assert (asked["urgent"].type, asked["urgent"].labels) == ("noul", LABELS)
    assert asked["refund"].type == "choice"
    assert r.json()["answers"]["urgent"]["type"] == "noul"
    assert r.json()["answers"]["refund"]["type"] == "noul"


def test_over_http_bad_labels_are_a_refusal_not_a_server_error():
    bad = {**ASKED, "labels": {"false": "calm", "true": "calm"}}
    r = TestClient(create_app(Recording())).post(
        "/v1/decide", json={"state": {"message": "x"}, "questions": {"q": bad}})
    assert r.status_code == 422
    assert LAYA_REFUSAL in r.text


def test_the_python_client_takes_labels_on_a_noul():
    backend = Recording()
    with Verdict(backend=backend) as v:
        result = v.decide({"message": "charged twice"},
                          {"urgent": Noul(instructions="Is `message` urgent?", labels=LABELS)})
    assert result.nouls["urgent"].noul == 0.5
    assert backend.asked[0]["urgent"].labels == LABELS


def test_the_cli_passes_labels_to_a_local_engine_and_does_not_rewrite_the_question(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(client, "decide", lambda *a, **k: (_ for _ in ()).throw(client.NoServer("none")))
    monkeypatch.setattr(inference, "_ENGINES", {})
    monkeypatch.setattr(inference, "load", lambda *a, **k: engine)
    bank = '{"urgent": {"type": "noul", "instructions": "Is `message` urgent?", ' \
           '"labels": {"false": "calm", "true": "urgent"}}}'
    assert cli.main(["decide", '{"message": "charged twice"}', "--questions", bank]) == 0
    sent = engine.seen[0]["urgent"]
    assert (sent["type"], sent["labels"]) == ("noul", LABELS)


def test_jev_is_refused_labels_rather_than_sent_a_field_its_sdk_has_no_name_for():
    request = DecideRequest(state={"message": "x"}, questions={"urgent": ASKED})
    with pytest.raises(QuestionError, match="Jev.*labels"):
        jev_capabilities("jev-latest").validate_request(request)
