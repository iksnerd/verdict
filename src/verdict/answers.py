"""One place that knows how to read an answer.

An answer arrives in two shapes that both have to work: a pydantic model from `schema.py` when it
came through the API, and a plain dict when it came straight off `laya_mlx.predict`. And a yes/no
answer carries a single `noul` probability where a `choice` or `score` answer carries a
distribution, so anything that treats questions uniformly has to bridge that.

The `noul` to `{"true", "false"}` conversion and the dict-or-attribute access live here, once, so
the pipeline, training and the router all agree and changing the convention is one edit.

The `"true"` / `"false"` keys are not arbitrary: they are the option names the Gemini teachers are
told to use (`label/teacher.py`) and the names Laya's training format expects, so they are wire
vocabulary rather than an internal choice.
"""
from __future__ import annotations

from typing import Any, Mapping

TRUE = "true"
FALSE = "false"


def field(answer: Any, name: str) -> Any:
    """Read `name` off an answer, whether it is a Mapping or a pydantic model."""
    return answer[name] if isinstance(answer, Mapping) else getattr(answer, name)


def kind(answer: Any) -> str:
    """The question type this answer came from: choice, score or noul."""
    return str(field(answer, "type"))


def noul_probs(p: float) -> dict[str, float]:
    """A yes/no probability as the two-option distribution the training format expects."""
    return {TRUE: float(p), FALSE: 1.0 - float(p)}


def probs(answer: Any) -> dict[str, float]:
    """Every answer as a distribution, so callers do not branch on the question type."""
    if kind(answer) == "noul":
        return noul_probs(field(answer, "noul"))
    return {str(k): float(v) for k, v in field(answer, "probabilities").items()}


def top(answer: Any) -> str:
    """The winning option key. For yes/no this is the side of 0.5 the probability falls on."""
    if kind(answer) == "noul":
        return TRUE if float(field(answer, "noul")) >= 0.5 else FALSE
    p = probs(answer)
    return max(p, key=p.__getitem__)


def max_probability(answer: Any) -> float:
    """Probability of the most likely outcome, including max(p, 1-p) for a Noul.

    An explicitly derived statistic with the same definition across providers; it is not the
    provider's confidence field or a promise of empirical accuracy. Calibrate before gating.
    """
    return max(probs(answer).values())


def point(answer: Any) -> float | dict[str, float]:
    """The shape the teachers label in: a bare probability for yes/no, a distribution otherwise.

    This is what the scorers compare against gold, and it is deliberately *not* `probs`: the
    teacher labels a yes/no question with one number, so scoring it as a two-option distribution
    would double-count the same decision.
    """
    if kind(answer) == "noul":
        return float(field(answer, "noul"))
    return probs(answer)


def gold_point(gold_entry: Mapping[str, Any]) -> float | dict[str, float]:
    """The same shape as `point`, read back out of an exported `gold[qid]` entry."""
    p = gold_entry["probabilities"]
    return float(p[TRUE]) if TRUE in p else {str(k): float(v) for k, v in p.items()}


def is_flat(answer: Any) -> bool:
    """True when this answer carries no information, i.e. nothing is behind it.

    Checked on the distribution rather than on `confidence`, because for a yes/no answer Laya
    reports `confidence == max(p, 1 - p)`, which makes a confidence gate the same test as a
    probability threshold, and because `UniformBackend` reports confidence 0.0 for a score question
    but 0.5 for a yes/no one, so no single confidence cutoff catches both.
    """
    if kind(answer) == "noul":
        return abs(float(field(answer, "noul")) - 0.5) < 1e-9
    values = list(probs(answer).values())
    return bool(values) and max(values) - min(values) < 1e-9
