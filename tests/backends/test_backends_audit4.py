"""Audit 4: one fixed context limit on the Engine, truncation checked whenever our own clip did not
mark it, a tokenizer that will not load is unknown (not a 500), closed backends are an UpstreamError."""
from __future__ import annotations

import logging

import httpx
import pytest

from verdict.backends import capabilities as caps
from verdict.backends.engine import LAYA_MAX_LEN, Engine
from verdict.backends.mlx import MlxBackend
from verdict.backends.systemone import SystemOneBackend
from verdict.schema import DecideRequest
from verdict.systemone import UpstreamError

from fakes import NOUL_BANK as BANK, Agent, fake_transformers


class Cfg(Agent):
    def __init__(self, max_len):
        super().__init__()
        self.cfg = {"max_len": max_len}


def test_engine_limit_is_fixed_at_first_use_and_never_moves_after(monkeypatch):
    """Defined once, on first use (configured max_len, else the loaded agent's cfg, else 512), then
    frozen: an earlier audit fixed a limit that moved mid-run."""
    fake_transformers(monkeypatch, [])
    e = Engine("main")
    e._agent = Cfg(1024)   # what a lazy load does, before anything asked for the limit
    text = "x " * 600
    assert e.exceeds_context(text) is False
    assert e.max_len == 1024
    e._agent.cfg["max_len"] = 2048   # nothing that happens later changes it
    assert e.max_len == 1024


def test_engine_limit_is_512_when_the_agent_reports_none(monkeypatch):
    fake_transformers(monkeypatch, [])
    e = Engine("main", agent=Agent())
    assert e.max_len == LAYA_MAX_LEN
    assert e.exceeds_context("x " * 600) is True


def test_engine_limit_comes_from_an_injected_agent(monkeypatch):
    fake_transformers(monkeypatch, [])
    e = Engine("main", agent=Cfg(1024))
    assert e.max_len == 1024
    assert e.exceeds_context("x " * 600) is False


def test_budget_larger_than_the_context_still_marks_truncation(monkeypatch):
    fake_transformers(monkeypatch, [])
    backend = MlxBackend("main", agent=Agent(), budget=5000)
    resp = backend.decide(DecideRequest(state="x " * 600, questions=BANK))
    assert resp.answers["q"].truncated is True


def test_tokenizer_load_failure_is_unknown_not_an_error_and_warns_once(monkeypatch, caplog):
    calls = []
    fake_transformers(monkeypatch, calls, fail=True)
    e = Engine("main", agent=Agent())
    with caplog.at_level(logging.WARNING):
        assert e.exceeds_context("x " * 600) is False
        assert e.exceeds_context("x " * 600) is False
    assert len(calls) == 1
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_tokenizer_property_still_raises(monkeypatch):
    fake_transformers(monkeypatch, [], fail=True)
    with pytest.raises(OSError):
        Engine("main", agent=Agent()).tokenizer


def test_decide_survives_a_broken_tokenizer(monkeypatch):
    fake_transformers(monkeypatch, [], fail=True)
    backend = MlxBackend("main", agent=Agent(), budget=None)
    resp = backend.decide(DecideRequest(state="x " * 600, questions=BANK))
    assert resp.answers["q"].truncated is None


def so_backend():
    return SystemOneBackend("http://x.example", "m", "k",
                            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})), attempts=1)


def test_closed_error_is_an_upstream_error_and_a_runtime_error():
    b = so_backend()
    b.close()
    with pytest.raises(UpstreamError):
        b.decide(DecideRequest(state="x", questions=BANK))
    with pytest.raises(RuntimeError, match="closed"):
        b.decide(DecideRequest(state="x", questions=BANK))


def test_models_on_a_closed_backend_is_none():
    b = so_backend()
    b.close()
    assert b.models() is None


def test_a_failing_close_leaves_the_backend_open():
    b = so_backend()

    def boom():
        raise OSError("close failed")

    b._ask.close = boom
    with pytest.raises(OSError):
        b.close()
    assert b._closed is False


def test_one_jev_precision_constant():
    assert hasattr(caps, "JEV_PRECISION")
    assert caps.jev_capabilities("m").precision == caps.JEV_PRECISION
    pytest.importorskip("httpx2")
    pytest.importorskip("typesafe_sdk")
    from verdict.backends import jev
    assert jev.JEV_PRECISION is caps.JEV_PRECISION
