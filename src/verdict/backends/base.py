"""Backends answer typed questions. Anything with `name` and `decide` is one."""
from __future__ import annotations

from typing import Protocol

from ..schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecideRequest,
    DecideResponse,
    NoulAnswer,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)


class UnknownModel(ValueError):
    """A request named a checkpoint (`DecideRequest.model`) this backend cannot serve."""


class Backend(Protocol):
    name: str

    def decide(self, request: DecideRequest) -> DecideResponse: ...


class UniformBackend:
    """Answers every question with maximum uncertainty. It exists so the API, the
    schema and the gating logic are testable before any model is trained, and so a
    caller can tell 'no model behind this' apart from 'the model is unsure'."""

    name = "uniform"

    def decide(self, request: DecideRequest) -> DecideResponse:
        answers = {}
        for qid, q in request.questions.items():
            if isinstance(q, ChoiceQuestion):
                keys = q.option_keys()
                p = round(1.0 / len(keys), 4)
                answers[qid] = ChoiceAnswer(
                    choice=keys[0], probabilities={k: p for k in keys}, confidence=0.0
                )
            elif isinstance(q, ScoreQuestion):
                keys = q.option_keys()
                p = round(1.0 / len(keys), 4)
                answers[qid] = ScoreAnswer(
                    score=(len(keys) - 1) / 2,
                    legend={k: c for k, c in zip(keys, q.criteria)},
                    probabilities={k: p for k in keys},
                    confidence=0.0,
                )
            else:
                answers[qid] = NoulAnswer(noul=0.5, confidence=0.5)
        return DecideResponse(model=self.name, answers=answers, usage=Usage(input_tokens=0))
