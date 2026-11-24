"""Audit 3: with no budget, a state laya cuts at its own context is marked, loading the tokenizer
lazily (once, only when the state could be long enough to matter)."""
from __future__ import annotations

import sys
import types

from verdict.backends.mlx import MlxBackend
from verdict.schema import DecideRequest

from fakes import NOUL_BANK as BANK, Agent, WordTokenizer


def fake_transformers(monkeypatch, calls):
    def from_pretrained(*a, **k):
        calls.append(a)
        return WordTokenizer()

    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=from_pretrained)))


def test_no_budget_loads_the_tokenizer_lazily_to_mark_a_long_state(monkeypatch):
    calls = []
    fake_transformers(monkeypatch, calls)
    backend = MlxBackend("main", agent=Agent(), budget=None)
    resp = backend.decide(DecideRequest(state="x " * 600, questions=BANK))
    assert resp.answers["q"].truncated is True
    backend.decide(DecideRequest(state="x " * 600, questions=BANK))
    assert len(calls) == 1


def test_no_budget_short_state_never_loads_a_tokenizer(monkeypatch):
    calls = []
    fake_transformers(monkeypatch, calls)
    backend = MlxBackend("main", agent=Agent(), budget=None)
    resp = backend.decide(DecideRequest(state="x " * 5, questions=BANK))
    assert resp.answers["q"].truncated is None
    assert calls == []
