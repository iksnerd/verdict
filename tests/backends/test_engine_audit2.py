"""Audit 2: lazy caches load once under concurrency, capabilities() is stable and cache-free,
`truncated` also marks a state laya cuts at its own context length."""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from verdict.backends import mlx as mlx_mod
from verdict.backends.engine import Engine
from verdict.backends.mlx import MlxBackend
from verdict.schema import DecideRequest

from fakes import NOUL_BANK as BANK, Agent, WordTokenizer, noul


def run_threads(fn, n=2):
    barrier = threading.Barrier(n)
    out = []

    def work():
        barrier.wait()
        out.append(fn())

    threads = [threading.Thread(target=work) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def test_agent_loads_once_under_two_threads(monkeypatch):
    loads = []

    def slow_load(model_id, **options):
        loads.append(model_id)
        time.sleep(0.05)
        return object()

    monkeypatch.setitem(sys.modules, "laya_mlx", types.SimpleNamespace(load=slow_load))
    engine = Engine("m")
    agents = run_threads(lambda: engine.agent)
    assert len(loads) == 1 and agents[0] is agents[1]


def test_failed_quantize_is_not_cached_and_does_not_leave_a_half_state(monkeypatch):
    monkeypatch.setitem(sys.modules, "laya_mlx", types.SimpleNamespace(load=lambda *a, **k: object()))
    from verdict.backends import engine as engine_mod

    def boom(agent, bits):
        raise RuntimeError("no")

    monkeypatch.setattr(engine_mod, "quantize", boom)
    engine = Engine("m", bits=8)
    with pytest.raises(RuntimeError):
        engine.agent
    assert engine._agent is None


def test_tokenizer_loads_once_under_two_threads(monkeypatch):
    loads = []

    class Auto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            loads.append(args)
            time.sleep(0.05)
            return WordTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Auto))
    engine = Engine("org/name", agent=object())
    run_threads(lambda: engine.tokenizer)
    assert len(loads) == 1


def test_extra_checkpoint_loads_once_under_two_threads(monkeypatch):
    loads = []

    def slow_load(model_id, **options):
        loads.append(model_id)
        time.sleep(0.05)
        return Engine(model_id, agent=object())

    monkeypatch.setattr(mlx_mod, "load", slow_load)
    backend = MlxBackend("main", agent=object(), multilingual_id="multi")
    engines = run_threads(lambda: backend.engine_for("multilingual"))
    assert loads == ["multi"] and engines[0] is engines[1]


def test_capabilities_do_not_depend_on_load_order_and_do_not_touch_the_cache():
    backend = MlxBackend("main", agent=Agent(), multilingual_id="multi")
    before = backend.capabilities("multilingual")
    assert backend._loaded == {}
    backend.decide(DecideRequest(state="x", questions=BANK, model="multilingual"))
    assert backend.capabilities("multilingual") == before
    assert backend.capabilities().max_state_and_question_tokens == \
        backend.capabilities().max_state_and_question_tokens


def test_capabilities_of_a_lazy_agent_do_not_change_once_it_has_a_cfg(monkeypatch):
    backend = MlxBackend("main", agent=Agent())
    before = backend.capabilities()
    backend.engine._agent = types.SimpleNamespace(cfg={"max_len": 77}, predict=Agent().predict)
    assert backend.capabilities() == before


def test_no_budget_still_marks_a_state_past_the_model_context():
    backend = MlxBackend("main", agent=Agent(), budget=None)
    backend.engine._tokenizer = WordTokenizer()
    long = DecideRequest(state="x " * 600, questions=BANK)
    short = DecideRequest(state="x " * 5, questions=BANK)
    assert backend.decide(long).answers["q"].truncated is True
    assert backend.decide(short).answers["q"].truncated is None
