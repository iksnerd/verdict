"""Audit 4: retry_unsafe re-sends a POST after a timeout only, never after an answer; honest log."""
from __future__ import annotations

import httpx
import pytest

from fakes import retry_run


@pytest.mark.parametrize("status", [500, 502, 504])
def test_retry_unsafe_never_resends_after_an_answer(status, monkeypatch):
    calls, exc = retry_run(httpx.Response(status), monkeypatch, retry_unsafe=True)
    assert len(calls) == 1 and "not re-sent" in str(exc)


def test_retry_unsafe_resends_after_a_read_timeout(monkeypatch):
    calls, exc = retry_run(httpx.ReadTimeout("loading"), monkeypatch, retry_unsafe=True)
    assert len(calls) == 2 and exc is None


def test_retry_unsafe_does_not_resend_after_another_transport_error(monkeypatch):
    calls, exc = retry_run(httpx.RemoteProtocolError("cut off"), monkeypatch, retry_unsafe=True)
    assert len(calls) == 1 and exc is not None


def test_the_retry_line_says_the_real_reason(monkeypatch, capsys):
    retry_run(httpx.Response(429), monkeypatch)
    err = capsys.readouterr().err
    assert "answered 429" in err and "may still be loading" not in err


def test_the_retry_line_says_loading_for_a_timeout(monkeypatch, capsys):
    retry_run(httpx.ReadTimeout("t"), monkeypatch, retry_unsafe=True)
    assert "may still be loading" in capsys.readouterr().err
