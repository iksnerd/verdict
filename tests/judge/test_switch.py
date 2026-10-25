"""The Switch is pure, so it is tested without a model, like every other test here."""
from __future__ import annotations

import pytest

from verdict.backends.base import UniformBackend
from verdict.schema import DecideRequest
from verdict.judge.switch import Branch, Switch, choice_switch

CASES = {"a": "the first thing", "b": "the second thing"}


def const(name: str | None):
    return lambda answers: None if name is None else Branch(name, "because")


def make(select, default="b"):
    return Switch(name="t", cases=CASES, default=default,
                  questions={"q": {"type": "noul", "instructions": "?"}}, select=select)


def test_the_mechanism_picks_the_branch():
    assert make(const("a")).decide({"q": {}}).name == "a"


def test_an_undecided_mechanism_falls_back_to_default():
    b = make(const(None)).decide({"q": {}})
    assert b.name == "b" and "default" in b.reason


def test_default_must_be_one_of_the_cases():
    with pytest.raises(ValueError, match="not one of the cases"):
        Switch(name="t", cases=CASES, default="z",
               questions={"q": {"type": "noul", "instructions": "?"}}, select=const("a"))


def test_a_switch_needs_at_least_two_cases():
    with pytest.raises(ValueError, match="at least 2 cases"):
        Switch(name="t", cases={"only": "one"}, default="only",
               questions={"q": {"type": "noul", "instructions": "?"}}, select=const("only"))


def test_a_switch_needs_a_question():
    with pytest.raises(ValueError, match="at least one question"):
        Switch(name="t", cases=CASES, default="b", questions={}, select=const("a"))


def test_a_mechanism_cannot_invent_a_branch():
    """The return value is a key of `cases`, checked. That is what closes the branch set."""
    with pytest.raises(ValueError, match="not one of"):
        make(const("nonexistent")).decide({"q": {}})


def test_a_missing_answer_is_an_error_not_a_silent_default():
    with pytest.raises(KeyError, match="'q'"):
        make(const("a")).decide({})


# --- choice_switch, the one-line mechanism ------------------------------------------------

def test_choice_switch_returns_the_chosen_case():
    sw = choice_switch("t", "which?", CASES, default="b")
    answers = {"t_case": {"type": "choice", "choice": "a", "probabilities": {"a": 0.8, "b": 0.2}, "confidence": 0.8}}
    assert sw.decide(answers).name == "a"


def test_choice_switch_falls_back_when_the_distribution_is_flat():
    """A flat distribution means no model, which must never pick a branch on its own."""
    sw = choice_switch("t", "which?", CASES, default="b")
    answers = {"t_case": {"type": "choice", "choice": "a", "probabilities": {"a": 0.5, "b": 0.5}, "confidence": 0.5}}
    assert sw.decide(answers).name == "b"


def test_choice_switch_against_the_uniform_backend():
    sw = choice_switch("t", "which?", CASES, default="b")
    resp = UniformBackend().decide(DecideRequest(state="x", questions=sw.questions))
    assert sw.decide(resp.answers).name == "b"


def test_choice_switch_min_confidence_is_opt_in():
    low = {"t_case": {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.4}, "confidence": 0.6}}
    assert choice_switch("t", "?", CASES, default="b").decide(low).name == "a"
    assert choice_switch("t", "?", CASES, default="b", min_confidence=0.9).decide(low).name == "b"


def test_choice_switch_builds_the_question_from_the_cases():
    sw = choice_switch("t", "which?", CASES, default="b")
    q = sw.questions["t_case"]
    assert q["type"] == "choice" and q["criteria"] == CASES


def test_branch_serialises():
    d = Branch("a", "because", {"x": 1.23456}).as_dict()
    assert d == {"branch": "a", "reason": "because", "scores": {"x": 1.2346}}
