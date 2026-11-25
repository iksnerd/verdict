"""`POST /v1/decide?abstain_below=P`: the opt-in abstention mark over HTTP."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from verdict.api import create_app
from verdict.schema import DecideResponse, NoulAnswer
from fakes import uniform_client

BODY = {"state": {"message": "hello"},
        "questions": {"urgent": {"type": "noul", "instructions": "Is `message` urgent?"}}}


class Stub:
    """A backend that is not `uniform`, answering every yes/no 0.6."""
    name = "stub"
    proxy = False

    def decide(self, request):
        return DecideResponse(model="stub", answers={q: NoulAnswer(noul=0.6, confidence=0.6)
                                                     for q in request.questions})


def test_the_query_parameter_marks_each_answer():
    c = TestClient(create_app(Stub()))
    a = c.post("/v1/decide?abstain_below=0.8&allow_unmeasured=true", json=BODY).json()["answers"]["urgent"]
    assert (a["abstention"], a["abstention_threshold"], a["noul"]) == ("abstained", 0.8, 0.6)
    a = c.post("/v1/decide?abstain_below=0.5&allow_unmeasured=true", json=BODY).json()["answers"]["urgent"]
    assert a["abstention"] == "passed"


def test_without_it_the_output_has_no_new_keys():
    a = TestClient(create_app(Stub())).post("/v1/decide?allow_unmeasured=true",
                                            json=BODY).json()["answers"]["urgent"]
    assert not [k for k in a if k.startswith("abstention")]


def test_a_uniform_backend_is_unevaluated():
    a = uniform_client().post("/v1/decide?abstain_below=0.8&allow_unmeasured=true",
                              json=BODY).json()["answers"]["urgent"]
    assert a["abstention"] == "unevaluated"


@pytest.mark.parametrize("bad", ["0", "1.5", "-1"])
def test_an_out_of_range_value_is_a_422_in_the_refusal_shape(bad):
    r = uniform_client().post(f"/v1/decide?abstain_below={bad}", json=BODY)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "refused"
    assert r.json()["detail"] == r.json()["error"]["message"]


@pytest.mark.parametrize("bad", ["x", "true"])
def test_a_non_number_is_a_422(bad):
    assert uniform_client().post(f"/v1/decide?abstain_below={bad}", json=BODY).status_code == 422


def test_systemone_never_carries_the_mark():
    jev = {"model": "jev-latest", "state": {"message": "hi"},
           "questions": {"spam": {"type": "noul", "instructions": "?"}}}
    a = TestClient(create_app(Stub())).post("/v1/systemone?abstain_below=0.9", json=jev).json()
    assert "abstention" not in a["answers"]["spam"]
