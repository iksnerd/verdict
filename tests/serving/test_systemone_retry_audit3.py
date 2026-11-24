"""Audit 3: a POST is re-sent only where the upstream provably did not process it. No network."""
from __future__ import annotations

import httpx
import pytest

from fakes import SYSTEMONE_OK as OK, SYSTEMONE_Q as Q, retry_run
from verdict import systemone
from verdict.systemone import SystemOneClient


def client(handler, attempts=3):
    return SystemOneClient("http://x.example", "m", "k", attempts=attempts, backoff=0.,
                           transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("first", [
    httpx.ConnectError("refused"), httpx.ConnectTimeout("slow connect"),
    httpx.Response(429), httpx.Response(429, headers={"Retry-After": "2"}),
    httpx.Response(503), httpx.Response(503, headers={"Retry-After": "2"}),
])
def test_a_post_the_upstream_did_not_process_is_retried(first, monkeypatch):
    calls, exc = retry_run(first, monkeypatch)
    assert exc is None and len(calls) == 2


@pytest.mark.parametrize("first", [
    httpx.ReadTimeout("no answer"), httpx.Response(500), httpx.Response(502), httpx.Response(504),
])
def test_a_post_that_may_have_been_processed_is_not_retried(first, monkeypatch):
    calls, exc = retry_run(first, monkeypatch)
    assert len(calls) == 1
    assert "may have been processed" in str(exc)


def test_a_get_is_still_retried_on_a_read_timeout(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json={"models": []})

    assert client(handler).models() == []
    assert len(calls) == 2


def test_the_wait_cap_still_holds(monkeypatch):
    waits = []
    monkeypatch.setattr("time.sleep", waits.append)
    n = []

    def handler(req):
        n.append(1)
        return httpx.Response(200, json=OK) if len(n) > 1 else httpx.Response(
            503, headers={"Retry-After": "86400"})

    client(handler)("x", Q)
    assert waits == [systemone.MAX_RETRY_WAIT]
