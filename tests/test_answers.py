"""`answers.py` replaced five hand-rolled noul conversions and two accessors. These pin the
behaviour all of them have to share, including against the real pydantic models."""
from __future__ import annotations


def test_max_probability_does_not_depend_on_provider_confidence():
    from verdict.answers import max_probability
    from verdict.schema import NoulAnswer

    assert max_probability(NoulAnswer(noul=.9)) == .9
    assert max_probability({"type": "choice", "probabilities": {"a": .7, "b": .3},
                            "confidence": .1}) == .7

import pytest

from verdict.answers import (
    FALSE,
    TRUE,
    field,
    gold_point,
    is_flat,
    kind,
    noul_probs,
    point,
    probs,
    top,
)
from verdict.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer

CHOICE = {"type": "choice", "choice": "b", "probabilities": {"a": 0.3, "b": 0.7}, "confidence": 0.7}
SCORE = {"type": "score", "score": 1.4, "legend": {}, "probabilities": {"0": 0.1, "1": 0.6, "2": 0.3}, "confidence": 0.6}
NOUL = {"type": "noul", "noul": 0.8, "confidence": 0.8}


def test_field_reads_dicts_and_models_alike():
    assert field(NOUL, "noul") == 0.8
    assert field(NoulAnswer(noul=0.8, confidence=0.8), "noul") == 0.8


def test_kind_reports_the_question_type():
    assert kind(CHOICE) == "choice" and kind(SCORE) == "score" and kind(NOUL) == "noul"


def test_noul_becomes_a_two_option_distribution():
    assert noul_probs(0.8) == {TRUE: 0.8, FALSE: pytest.approx(0.2)}
    assert probs(NOUL) == {TRUE: 0.8, FALSE: pytest.approx(0.2)}


def test_probs_leaves_a_real_distribution_alone():
    assert probs(CHOICE) == {"a": 0.3, "b": 0.7}
    assert sum(probs(SCORE).values()) == pytest.approx(1.0)


def test_top_is_the_argmax_and_the_side_of_one_half():
    assert top(CHOICE) == "b"
    assert top(SCORE) == "1"
    assert top(NOUL) == TRUE
    assert top({"type": "noul", "noul": 0.2, "confidence": 0.8}) == FALSE


def test_top_treats_exactly_one_half_as_true():
    """Pinning the boundary, because `parity.py` used `>= 0.5` and a scorer could have differed."""
    assert top({"type": "noul", "noul": 0.5, "confidence": 0.5}) == TRUE


def test_point_is_the_shape_the_teachers_label_in():
    """A yes/no question is one number, not two, or scoring double-counts the same decision."""
    assert point(NOUL) == 0.8
    assert point(CHOICE) == {"a": 0.3, "b": 0.7}


def test_gold_point_round_trips_what_export_writes():
    assert gold_point({"probabilities": noul_probs(0.8)}) == 0.8
    assert gold_point({"probabilities": {"a": 0.3, "b": 0.7}}) == {"a": 0.3, "b": 0.7}


def test_point_and_gold_point_agree_so_a_scorer_can_compare_them():
    assert point(NOUL) == gold_point({"probabilities": probs(NOUL)})


def test_is_flat_detects_an_answer_with_nothing_behind_it():
    assert is_flat({"type": "noul", "noul": 0.5, "confidence": 0.5})
    assert is_flat({"type": "choice", "choice": "a", "probabilities": {"a": 0.5, "b": 0.5}, "confidence": 0.0})
    assert not is_flat(NOUL)
    assert not is_flat(CHOICE)


def test_is_flat_works_on_the_pydantic_models_too():
    assert is_flat(NoulAnswer(noul=0.5, confidence=0.5))
    assert is_flat(ScoreAnswer(score=1.5, legend={}, probabilities={"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}, confidence=0.0))
    assert not is_flat(ChoiceAnswer(choice="b", probabilities={"a": 0.3, "b": 0.7}, confidence=0.7))


# --- the schema mirrors laya's format, so it must not be stricter than it ------------------

def test_noul_accepts_the_criteria_laya_renders():
    """`extra: forbid` with no `criteria` field rejected input laya accepts, and `criteria` is the
    only context channel a yes/no question has: without it the model reads literally
    "true: yes, the statement holds"."""
    from verdict.schema import DecideRequest

    req = DecideRequest(state="x", questions={"q": {
        "type": "noul", "instructions": "?",
        "criteria": {"true": "money or safety", "false": "a lookup"}}})
    assert req.questions["q"].criteria == {"true": "money or safety", "false": "a lookup"}


def test_noul_criteria_are_still_bounded_to_the_two_sides():
    from verdict.schema import DecideRequest

    with pytest.raises(Exception, match="true"):
        DecideRequest(state="x", questions={"q": {
            "type": "noul", "instructions": "?", "criteria": {"maybe": "x"}}})


def test_noul_criteria_stay_optional():
    from verdict.schema import DecideRequest

    req = DecideRequest(state="x", questions={"q": {"type": "noul", "instructions": "?"}})
    assert req.questions["q"].criteria is None


@pytest.mark.parametrize(
    "question",
    [
        {"type": "choice", "instructions": "?", "criteria": {"a": {"means": "x"}, "b": "y"}},
        {"type": "score", "instructions": "?", "criteria": [{"lo": 1}, "hi"]},
    ],
    ids=["choice", "score"],
)
def test_structured_criteria_are_accepted(question):
    """laya renders a dict or list criterion as compact JSON, so a rubric arrives as JSON."""
    from verdict.schema import DecideRequest

    DecideRequest(state="x", questions={"q": question})
