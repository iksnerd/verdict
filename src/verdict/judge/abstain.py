"""An opt-in abstention mark on answers: "the model was not sure enough to act on this".

It reports and never changes an answer. The raw argmax, the probabilities and `confidence` stay as
the backend gave them; the caller reads `abstention` and decides. The mark is computed here, from
each answer's own probabilities, so it works on every backend (laya's own `min_confidence` gate
exists only on one).

A scalar threshold on base Laya is a threshold, not a calibrated probability: the model is
over-confident. Pick it with `verdict calibrate` data (docs/usage/guide.md).
"""
from __future__ import annotations

import math
from typing import Any

from ..answers import max_probability
from ..errors import QuestionError

PASSED, ABSTAINED, UNEVALUATED = "passed", "abstained", "unevaluated"


def validate_threshold(value: Any) -> float:
    """The threshold as a float, or a `QuestionError` (exit 2 on the CLI, 422 over HTTP).
    It must be a real number in (0, 1]: a bool, a string, NaN, 0 and anything above 1 are refused."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value <= 1):
        raise QuestionError(f"abstain threshold {value!r} must be a number above 0 and at most 1 "
                            "(the lowest max-probability an answer may have and still pass)")
    return float(value)


def parse_threshold(text: str) -> float:
    """`validate_threshold` for a command-line string."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        raise QuestionError(f"abstain threshold {text!r} is not a number; give a probability "
                            "above 0 and at most 1") from None
    return validate_threshold(value)


def gate(answers: dict[str, Any], threshold: float, *, uniform: bool) -> dict[str, dict[str, Any]]:
    """Each answer (a dict) with `abstention` and `abstention_threshold` added; the input is untouched.

    Confidence is the largest probability: for a choice or a score, the max of `probabilities`;
    for a yes/no, max(p, 1 - p). `abstained` when it is below `threshold`, else `passed`.

    `uniform` is a backend with no model behind it. Its answer means "no model", not "unsure", so
    it is never gated: every answer reads `unevaluated`.
    """
    threshold = validate_threshold(threshold)
    out = {}
    for qid, a in answers.items():
        if uniform:
            mark = UNEVALUATED
        else:
            mark = PASSED if max_probability(a) >= threshold else ABSTAINED
        out[qid] = {**a, "abstention": mark, "abstention_threshold": threshold}
    return out
