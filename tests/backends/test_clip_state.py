"""`decide` clips long states the way `route` always has (FINDINGS §11, §29): 380 ms a call on
600-character commands against 65 ms on short ones. A fake tokenizer, one token per word."""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from verdict.api import create_app
from verdict.backends.mlx import MlxBackend
from verdict.backends.engine import Engine
from fakes import noul


class WordTokenizer:
    def __call__(self, text, add_special_tokens=True):
        ids = list(range(len(text.split())))
        return {"input_ids": [-1] + ids + [-2] if add_special_tokens else ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


def engine():
    e = Engine("fake", agent=object())
    e._tokenizer = WordTokenizer()
    return e


def test_a_short_state_is_returned_untouched_even_when_structured():
    state = {"command": "ls -la"}
    assert engine().clip_state(state, 128) is state


def test_a_long_string_is_clipped_to_the_budget():
    assert engine().clip_state("a " * 500, 10).split() == [f"w{i}" for i in range(10)]


def test_a_long_dict_is_serialized_the_way_laya_does_then_clipped():
    state = {"command": "x " * 500}
    clipped = engine().clip_state(state, 10)
    assert isinstance(clipped, str) and len(clipped.split()) == 10


def test_no_budget_means_no_clipping():
    state = {"command": "x " * 500}
    assert engine().clip_state(state, None) is state
    assert engine().clip_state(state, 0) is state


class Agent:
    def __init__(self):
        self.seen = None

    def predict(self, state, questions):
        self.seen = state
        return {"answers": {k: noul(0.5) for k in questions}}


def test_the_server_clips_decide_states_to_its_budget():
    agent = Agent()
    backend = MlxBackend(agent=agent, budget=10)
    backend.engine._tokenizer = WordTokenizer()
    body = {"state": {"command": "x " * 500}, "questions": {"q": {"type": "noul", "instructions": "?"}}}
    assert TestClient(create_app(backend)).post("/v1/decide", json=body).status_code == 200
    assert isinstance(agent.seen, str) and len(agent.seen.split()) == 10


def test_the_server_passes_short_states_through_as_structured():
    agent = Agent()
    backend = MlxBackend(agent=agent, budget=10)
    backend.engine._tokenizer = WordTokenizer()
    body = {"state": {"command": "ls"}, "questions": {"q": {"type": "noul", "instructions": "?"}}}
    TestClient(create_app(backend)).post("/v1/decide", json=body)
    assert agent.seen == {"command": "ls"}
