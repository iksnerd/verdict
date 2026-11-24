"""Audit 4: the SDK raises one error type, also for a malformed answer and a bad config."""
from __future__ import annotations

import pytest

from verdict import sdk
from verdict.config import ConfigError


Q = {"q": {"type": "noul", "instructions": "is it?"}}


def test_a_malformed_served_answer_is_a_verdict_error(monkeypatch):
    monkeypatch.setattr(sdk._http, "decide", lambda *a, **k: {"unexpected": "shape"})
    with pytest.raises(sdk.VerdictError) as err:
        sdk.Verdict(url="http://127.0.0.1:1").decide("x", Q)
    assert err.value.code == "upstream" and err.value.status == 502


def test_a_config_error_is_a_verdict_error(monkeypatch):
    def bad(*a, **k):
        raise ConfigError("[server].url is 'x'")
    monkeypatch.setattr(sdk._http, "decide", bad)
    with pytest.raises(sdk.VerdictError) as err:
        sdk.Verdict().decide("x", Q)
    assert err.value.code == "config" and "[server].url" in str(err.value)
