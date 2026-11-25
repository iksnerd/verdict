"""Test doubles that more than one test file needs. Fixtures built on them live in conftest.py."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from verdict.api import create_app
from verdict.backends.base import UniformBackend
from verdict.schema import ChoiceAnswer, DecideResponse, NoulAnswer, ScoreAnswer, Usage
from verdict.systemone import SystemOneClient, UpstreamError

REPO_ROOT = Path(__file__).resolve().parents[1]


def project_version():
    """The package version, as pyproject.toml says it."""
    import tomllib

    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]


def typed(questions, noul_answer):
    """A fake server's answers, typed as a real server types them. `noul_answer(qid)` is the
    yes/no answer the fake would give; a question the CLI rewrote as a no/yes choice gets the
    same number back as a choice, so fakes written for yes/no keep their meaning."""
    out = {}
    for qid, q in questions.items():
        a = noul_answer(qid)
        if q.get("type") == "choice" and list(q.get("criteria") or {}) == ["no", "yes"]:
            p = a["noul"]
            a = {"type": "choice", "choice": "yes" if p >= 0.5 else "no", "confidence": a["confidence"],
                 "probabilities": {"no": 1 - p, "yes": p}}
        out[qid] = a
    return out


def noul(p, confidence=None):
    """A yes/no answer as a server types it. `confidence` defaults to `p`."""
    return {"type": "noul", "noul": p, "confidence": p if confidence is None else confidence}


def sure_noul(p):
    """A yes/no answer as sure as `p` is far from 0.5: confidence is the likelier side."""
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}


def uniform_client():
    """The decision API over the `uniform` backend: every route answers, no model is loaded."""
    return TestClient(create_app(UniformBackend()))


class FakeLayaAgent:
    """What `laya_mlx.load` returns in tests: answers every question 0.6, counts the calls."""

    def __init__(self, tag):
        self.tag, self.calls = tag, 0

    def predict(self, state, questions):
        self.calls += 1
        return {"answers": typed(questions, lambda k: noul(0.6))}


class FakeEngine:
    """A local engine for CLI tests: answers every yes/no `p`, remembers the questions it saw.
    `strict` raises like laya-mlx does on a question with no `instructions` key."""
    name = "fake-local"

    def __init__(self, p=0.5, strict=False, name=None):
        self.p, self.strict, self.seen = p, strict, []
        if name is not None:
            self.name = name

    def clip_state(self, state, budget):
        return state

    def predict(self, state, questions):
        if self.strict and any("instructions" not in q for q in questions.values()):
            raise ValueError("Question is missing instructions")
        self.seen.append(questions)
        return typed(questions, lambda k: noul(self.p))


NOUL_BANK = {"q": {"type": "noul", "instructions": "?"}}
URGENT_BANK = {"urgent": {"type": "noul", "instructions": "?"}}
SPAM_BANK = {"spam": {"type": "noul", "instructions": "Is this spam?"}}

#: A one-question yes/no bank as the JSON a `--questions` flag takes, and five states for stdin.
RISKY_BANK = json.dumps({"risky": {"type": "noul", "instructions": "Is `kind` marked bad?"}})
KIND_STATES = "".join(json.dumps({"kind": k}) + "\n" for k in "abcde")


class WordTokenizer:
    """A fake tokenizer, one token per word."""

    def __call__(self, text, add_special_tokens=True):
        ids = list(range(len(text.split())))
        return {"input_ids": [-1] + ids + [-2] if add_special_tokens else ids}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)


class Agent:
    """A laya agent double: answers every question 0.5 and keeps the state it saw."""

    def __init__(self):
        self.seen = None

    def predict(self, state, questions):
        self.seen = state
        return {"answers": {k: noul(0.5) for k in questions}}


def router_answers(difficulty_probs, difficulty, sensitive, sens_conf=0.9):
    return {
        "difficulty": {"type": "score", "score": difficulty, "legend": {}, "probabilities": difficulty_probs,
                       "confidence": 0.0},
        "is_sensitive": {"type": "noul", "noul": sensitive, "confidence": sens_conf},
    }


class Recording:
    """A local backend that records the questions it was asked and answers every one 0.5."""
    name = "recording"

    def __init__(self):
        self.asked = []

    def decide(self, request):
        self.asked.append(request.questions)
        answers = {}
        for qid, q in request.questions.items():
            if q.type == "noul":
                answers[qid] = NoulAnswer(noul=0.5, confidence=0.5)
                continue
            keys = q.option_keys()
            answers[qid] = ChoiceAnswer(choice=keys[0], probabilities={k: 1 / len(keys) for k in keys},
                                        confidence=0.0)
        return DecideResponse(model=self.name, answers=answers, usage=Usage(input_tokens=0))


GOOD_LABELS = {
    "kind": {"action": 0.7, "synthesis": 0.1, "decision": 0.05, "thought": 0.05, "draft": 0.05, "note": 0.03, "other": 0.02},
    "records_decision": {"true": 0.1, "false": 0.9},
    "reports_shipped_work": {"true": 0.9, "false": 0.1},
    "leaves_open_question": {"true": 0.1, "false": 0.9},
    "is_proposal": {"true": 0.1, "false": 0.9},
}


class PaidGarbage:
    """Answers every call, bills for it, and never returns a parsable label."""

    def __init__(self):
        self.calls = 0

    def __call__(self, model, prompt):
        self.calls += 1
        return "not json", {"promptTokenCount": 1000, "candidatesTokenCount": 500}


class FailsAt:
    """A backend that answers every call (0.5 yes/no, uniform scores) until call number `fail_at`
    (counted from 0), which raises `exc` (an `UpstreamError` unless given)."""
    proxy = False

    def __init__(self, exc=None, fail_at=2, name="flaky"):
        self.calls, self.exc, self.fail_at, self.name = 0, exc or UpstreamError("upstream went away"), fail_at, name

    def decide(self, request):
        self.calls += 1
        if self.calls - 1 == self.fail_at:
            raise self.exc
        answers = {}
        for qid, q in request.questions.items():
            if q.type == "score":
                keys = q.option_keys()
                answers[qid] = ScoreAnswer(score=0.0, legend={k: k for k in keys},
                                           probabilities={k: 1 / len(keys) for k in keys},
                                           confidence=0.5)
            else:
                answers[qid] = NoulAnswer(noul=0.5, confidence=0.5)
        return DecideResponse(model=self.name, answers=answers)


#: A `/v1/systemone` answer to `SYSTEMONE_Q`, and the question it answers.
SYSTEMONE_OK = {"model": "m", "answers": {"q": {"type": "noul", "noul": .9}}}
SYSTEMONE_Q = {"q": {"type": "noul", "instructions": "?"}}


def retry_run(first, monkeypatch, **client_kw):
    """What happens when the first attempt fails with `first` (a response or an exception):
    returns `(calls, exc)`, `exc` being the `UpstreamError` if the call gave up. Later attempts
    answer 200. No sleeping, no network."""
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = []

    def handler(req):
        calls.append(req)
        if len(calls) == 1:
            if isinstance(first, Exception):
                raise first
            return first
        return httpx.Response(200, json=SYSTEMONE_OK)

    c = SystemOneClient("http://x.example", "m", "k", attempts=3, backoff=0.,
                        transport=httpx.MockTransport(handler), **client_kw)
    try:
        c("x", SYSTEMONE_Q)
        return calls, None
    except UpstreamError as exc:
        return calls, exc


def fake_mlx_serve(monkeypatch, run=None):
    """Stubs what `verdict serve` loads and starts: `backends.mlx.MlxBackend` becomes a class named
    `fake:<model_id>` with no model behind it, and `uvicorn.run` calls `run` (default: nothing).
    Returns the list of `(model_id, kwargs)` the backend was built with."""
    import uvicorn

    import verdict.backends.mlx as backend_mlx

    built = []

    class FakeMlxBackend:
        def __init__(self, model_id, **kw):
            built.append((model_id, kw))
            self.name = f"fake:{model_id}"
            self.engine = types.SimpleNamespace(agent=None, tokenizer=None)

    monkeypatch.setattr(backend_mlx, "MlxBackend", FakeMlxBackend)
    monkeypatch.setattr(uvicorn, "run", run or (lambda *a, **k: None))
    return built


def install_laya(monkeypatch, load):
    """Puts a fake `laya_mlx` module in `sys.modules` whose `load` is `load`; returns the module."""
    module = types.ModuleType("laya_mlx")
    module.load = load
    monkeypatch.setitem(sys.modules, "laya_mlx", module)
    return module


def fake_transformers(monkeypatch, calls, fail=False, from_pretrained=None):
    """A fake `transformers` whose `AutoTokenizer.from_pretrained` records its args in `calls` and
    returns a `WordTokenizer` (or raises OSError when `fail`). `from_pretrained` replaces that
    function wholesale, for a test that needs a slow or special loader."""
    def default(*a, **k):
        calls.append(a)
        if fail:
            raise OSError("no tokenizer")
        return WordTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=from_pretrained or default)))
