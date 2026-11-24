"""Audit 2: models() tells 'down' from 'none'; a closed remote backend says so; uniform is detected by type."""
from __future__ import annotations

import httpx
import pytest

from verdict.backends.base import UniformBackend, capabilities_of, listed_models
from verdict.backends.systemone import SystemOneBackend
from verdict.schema import DecideRequest
from verdict.systemone import UpstreamError

Q = {"q": {"type": "noul", "instructions": "?"}}


def backend(handler):
    return SystemOneBackend("http://x.example", "m", "k", transport=httpx.MockTransport(handler),
                            attempts=1)


def test_models_none_when_the_service_is_down():
    b = backend(lambda r: httpx.Response(500))
    assert b.models() is None
    assert listed_models(b) is None


def test_models_empty_list_when_the_service_has_none():
    b = backend(lambda r: httpx.Response(200, json={"models": []}))
    assert b.models() == []
    assert listed_models(b) == []


def test_decide_after_close_says_closed():
    b = backend(lambda r: httpx.Response(200, json={}))
    b.close()
    with pytest.raises(RuntimeError, match="closed"):
        b.decide(DecideRequest(state="x", questions=Q))


def test_close_twice_is_fine():
    b = backend(lambda r: httpx.Response(200, json={}))
    b.close()
    b.close()


def test_a_subclass_of_uniform_is_still_uniform_family():
    class Mine(UniformBackend):
        name = "mine"
    assert capabilities_of(Mine()).family == "uniform"
