"""Test doubles that more than one test file needs. Fixtures built on them live in conftest.py."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from verdict.api import create_app
from verdict.backends.base import UniformBackend
from verdict.schema import ChoiceAnswer, DecideResponse, NoulAnswer, Usage

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


NOUL_BANK = {"q": {"type": "noul", "instructions": "?"}}


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
