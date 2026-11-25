"""`judge.abstain.gate`: the opt-in abstention mark. It reports; it never changes an answer."""
from __future__ import annotations

import pytest

from verdict.errors import QuestionError
from verdict.judge import abstain

ANSWERS = {
    "c": {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.4},
    "s": {"type": "score", "score": 1.0, "legend": {"lo": "", "hi": ""},
          "probabilities": {"lo": 0.55, "hi": 0.45}, "confidence": 0.1},
    "n": {"type": "noul", "noul": 0.2, "confidence": 0.8},
}


def test_marks_passed_and_abstained_for_each_type():
    out = abstain.gate(ANSWERS, 0.6, uniform=False)
    assert out["c"]["abstention"] == "passed"      # max p 0.7
    assert out["s"]["abstention"] == "abstained"   # max p 0.55
    assert out["n"]["abstention"] == "passed"      # max(0.2, 0.8) = 0.8
    assert {a["abstention_threshold"] for a in out.values()} == {0.6}


def test_noul_confidence_is_max_of_p_and_one_minus_p():
    low = {"n": {"type": "noul", "noul": 0.55}}
    assert abstain.gate(low, 0.6, uniform=False)["n"]["abstention"] == "abstained"
    high = {"n": {"type": "noul", "noul": 0.05}}   # a confident no
    assert abstain.gate(high, 0.9, uniform=False)["n"]["abstention"] == "passed"


def test_the_cut_itself_passes():
    out = abstain.gate({"n": {"type": "noul", "noul": 0.75}}, 0.75, uniform=False)
    assert out["n"]["abstention"] == "passed"


def test_uniform_backend_is_never_passed_or_abstained():
    out = abstain.gate(ANSWERS, 0.6, uniform=True)
    assert {a["abstention"] for a in out.values()} == {"unevaluated"}


def test_answers_are_not_changed_and_the_input_is_not_mutated():
    out = abstain.gate(ANSWERS, 0.6, uniform=False)
    assert "abstention" not in ANSWERS["c"]
    assert {k: v for k, v in out["c"].items() if not k.startswith("abstention")} == ANSWERS["c"]


@pytest.mark.parametrize("bad", [0, 0.0, 1.5, -1, "x", True, False, None, float("nan"), float("inf")])
def test_invalid_thresholds_are_refused(bad):
    with pytest.raises(QuestionError, match="abstain"):
        abstain.validate_threshold(bad)


@pytest.mark.parametrize("ok", [0.01, 0.5, 1, 1.0])
def test_valid_thresholds_pass(ok):
    assert abstain.validate_threshold(ok) == float(ok)
