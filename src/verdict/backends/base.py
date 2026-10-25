"""Backends answer typed questions. Anything with `name` and `decide` is one.

Everything else is optional, and read through the helpers at the bottom so the default lives in one
place: `proxy` (True when the backend forwards to another model server, which then keeps the
caller's model names and gets the refusals but none of laya's local handling), `models()` (its own
model list), `known_models()` (extra names a request may select), `capabilities(model)`, `engine`
(a local engine that can clip a prompt) and `model_id`. A new backend sets `proxy` explicitly.
"""
from __future__ import annotations

from typing import Any, Protocol

from .capabilities import Capabilities

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
    proxy = False

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


def is_proxy(backend: Backend) -> bool:
    """True when `backend` forwards to another model server (`proxy = True`)."""
    return bool(getattr(backend, "proxy", False))


def known_models(backend: Backend) -> set[str]:
    """Names besides the served checkpoint that a request may select; none by default."""
    return getattr(backend, "known_models", lambda: set())()


def listed_models(backend: Backend) -> list[dict[str, Any]] | None:
    """The backend's own model list, or None when it has none and the caller describes it."""
    return backend.models() if hasattr(backend, "models") else None


def capabilities_of(backend: Backend, model: str | None = None) -> Capabilities:
    """The backend's limits, or an `unknown` family (`uniform` for the uniform backend)."""
    if hasattr(backend, "capabilities"):
        return backend.capabilities(model)
    return Capabilities(family="uniform" if backend.name == "uniform" else "unknown",
                        model=backend.name)


def engine_of(backend: Backend) -> Any:
    """The local engine behind `backend`, or None (uniform, a proxy, a decide-only backend)."""
    return getattr(backend, "engine", None)


def model_id_of(backend: Backend, default: Any = None) -> Any:
    """The checkpoint a backend serves (a path or Hub id), or `default`."""
    return getattr(backend, "model_id", default)
