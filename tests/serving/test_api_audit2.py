"""Audit 2: which failures are a 502 and which are ours. No model."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from verdict.api import create_app, UNDATED, _release_date
from verdict.backends.base import UnknownModel
from verdict.config import ConfigError
from verdict.errors import QuestionError
from verdict.systemone import UpstreamError

BODY = {"state": {"m": "hi"}, "questions": {"q": {"type": "noul", "instructions": "Is `m` odd?"}}}


class Raising:
    name = "raising"
    proxy = False

    def __init__(self, exc):
        self.exc = exc

    def decide(self, request):
        raise self.exc


def _status(exc):
    client = TestClient(create_app(Raising(exc)), raise_server_exceptions=False)
    return client.post("/v1/decide", json=BODY).status_code


def _validation_error():
    class M(BaseModel):
        x: int
    try:
        M(x="no")
    except ValidationError as e:
        return e


@pytest.mark.parametrize("exc, status", [
    (UpstreamError("down"), 502),
    (UnknownModel("nope"), 400),
    (QuestionError("refused"), 422),
    (ValueError("a local bug"), 500),
    (ConfigError("bad config"), 500),
    (_validation_error(), 500),
])
def test_each_failure_maps_to_its_status(exc, status):
    assert _status(exc) == status


def test_the_release_date_is_fixed_without_a_model_path():
    from verdict.backends.base import UniformBackend

    assert _release_date(UniformBackend()) == UNDATED == _release_date(UniformBackend())


def test_a_hub_id_is_undated_too():
    class Hub:
        name = "x"
        model_id = "owner/not-on-disk-anywhere"

    assert _release_date(Hub()) == UNDATED


def test_healthz_says_a_uniform_backend_is_not_ready():
    from fakes import uniform_client

    r = uniform_client().get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "uniform" and body["ready"] is False and body["status"] != "ok"


def test_a_uniform_models_list_is_marked_for_the_client():
    from fakes import uniform_client

    r = uniform_client().get("/v1/models")
    assert r.headers["X-Verdict-Backend"] == "uniform"
