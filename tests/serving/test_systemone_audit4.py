"""Audit 4: retry_unsafe re-sends a POST after a timeout only, never after an answer; honest log."""
from __future__ import annotations

import httpx
import pytest

from verdict.systemone import SystemOneClient, UpstreamError

OK = {"model": "m", "answers": {"q": {"type": "noul", "noul": .9}}}
Q = {"q": {"type": "noul", "instructions": "?"}}


def run(first, monkeypatch, **kw):
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            if isinstance(first, Exception):
                raise first
            return first
        return httpx.Response(200, json=OK)

    c = SystemOneClient("http://x.example", "m", "k", attempts=3, backoff=0.,
                        transport=httpx.MockTransport(handler), **kw)
    try:
        c("x", Q)
        return calls, None
    except UpstreamError as exc:
        return calls, exc


@pytest.mark.parametrize("status", [500, 502, 504])
def test_retry_unsafe_never_resends_after_an_answer(status, monkeypatch):
    calls, exc = run(httpx.Response(status), monkeypatch, retry_unsafe=True)
    assert len(calls) == 1 and "not re-sent" in str(exc)


def test_retry_unsafe_resends_after_a_read_timeout(monkeypatch):
    calls, exc = run(httpx.ReadTimeout("loading"), monkeypatch, retry_unsafe=True)
    assert len(calls) == 2 and exc is None


def test_retry_unsafe_does_not_resend_after_another_transport_error(monkeypatch):
    calls, exc = run(httpx.RemoteProtocolError("cut off"), monkeypatch, retry_unsafe=True)
    assert len(calls) == 1 and exc is not None


def test_the_retry_line_says_the_real_reason(monkeypatch, capsys):
    run(httpx.Response(429), monkeypatch)
    err = capsys.readouterr().err
    assert "answered 429" in err and "may still be loading" not in err


def test_the_retry_line_says_loading_for_a_timeout(monkeypatch, capsys):
    run(httpx.ReadTimeout("t"), monkeypatch, retry_unsafe=True)
    assert "may still be loading" in capsys.readouterr().err
