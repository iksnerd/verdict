"""Test doubles that more than one test file needs. Fixtures built on them live in conftest.py."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from verdict.api import create_app
from verdict.backends.base import UniformBackend

REPO_ROOT = Path(__file__).resolve().parents[1]

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

    def __init__(self, p=0.5, strict=False):
        self.p, self.strict, self.seen = p, strict, []

    def clip_state(self, state, budget):
        return state

    def predict(self, state, questions):
        if self.strict and any("instructions" not in q for q in questions.values()):
            raise ValueError("Question is missing instructions")
        self.seen.append(questions)
        return typed(questions, lambda k: noul(self.p))
