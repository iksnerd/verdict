"""`validate_response` checks a provider's answer for internal consistency. It must accept the
precision the provider really reports: Laya rounds to four decimals, hosted Jev to two, so a valid
Jev choice can sum to 0.99. Found by the first live Jev bench run (emotion, 6 options)."""
from __future__ import annotations

import pytest

from verdict.schema import DecideRequest
from verdict.systemone import UpstreamError, validate_response

OPTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
Q = {"e": {"type": "choice", "instructions": "?", "criteria": {o: o for o in OPTIONS}}}


def check(probs, choice, q=Q, type_="choice", **extra):
    raw = {"model": "m", "answers": {"e": {"type": type_, "choice": choice, "probabilities": probs,
                                           "confidence": 0.5, **extra}}}
    return validate_response(raw, DecideRequest(state="x", questions=q))


def test_a_two_decimal_distribution_that_sums_to_099_is_accepted():
    p = {"sadness": 0.28, "joy": 0.0, "love": 0.0, "anger": 0.0, "fear": 0.66, "surprise": 0.05}
    assert check(p, "fear").answers["e"].choice == "fear"


def test_a_two_decimal_distribution_that_sums_to_101_is_accepted():
    p = {"sadness": 0.34, "joy": 0.0, "love": 0.0, "anger": 0.0, "fear": 0.34, "surprise": 0.33}
    assert check(p, "sadness").answers["e"].choice == "sadness"


def test_a_two_decimal_distribution_that_is_genuinely_off_is_still_refused():
    p = {"sadness": 0.28, "joy": 0.0, "love": 0.0, "anger": 0.0, "fear": 0.5, "surprise": 0.05}
    with pytest.raises(UpstreamError):
        check(p, "fear")  # 0.83


def test_a_four_decimal_distribution_keeps_its_tight_tolerance():
    """Laya's precision: 0.99 is wrong at four decimals, not rounding."""
    p = {"sadness": 0.2812, "joy": 0.0, "love": 0.0, "anger": 0.0, "fear": 0.6712, "surprise": 0.0476}
    check(p, "fear")  # sums to 1.0000
    bad = {**p, "fear": 0.6612}  # 0.9900 at four decimals
    with pytest.raises(UpstreamError):
        check(bad, "fear")


def test_a_score_computed_from_unrounded_probabilities_is_accepted_at_two_decimals():
    q = {"s": {"type": "score", "instructions": "?", "criteria": ["a", "b", "c", "d", "e"]}}
    probs = {"0": 0.07, "1": 0.13, "2": 0.4, "3": 0.28, "4": 0.12}   # sums to 1.00
    exact = sum(int(k) * v for k, v in probs.items())               # 2.45
    raw = {"model": "m", "answers": {"s": {
        "type": "score", "score": exact + 0.04, "confidence": 0.4, "probabilities": probs,
        "legend": {str(i): c for i, c in enumerate("abcde")}}}}
    assert validate_response(raw, DecideRequest(state="x", questions=q)).answers["s"].score > 2


def test_a_choice_one_rounding_step_below_the_displayed_maximum_is_accepted_at_two_decimals():
    """Jev nudges its rounded values to sum to 1, so the stated choice (fear) can sit 0.01 under
    the displayed maximum (sadness). Seen live on dair-ai/emotion."""
    p = {"sadness": 0.47000000000000003, "joy": 0.01, "love": 0.0, "anger": 0.0, "fear": 0.46,
         "surprise": 0.06}
    assert check(p, "fear").answers["e"].choice == "fear"


def test_a_choice_two_steps_below_the_maximum_is_still_refused_at_two_decimals():
    p = {"sadness": 0.48, "joy": 0.01, "love": 0.0, "anger": 0.0, "fear": 0.45, "surprise": 0.06}
    with pytest.raises(UpstreamError):
        check(p, "fear")


def test_at_four_decimals_a_choice_below_the_maximum_stays_refused():
    p = {"sadness": 0.4712, "joy": 0.0112, "love": 0.0, "anger": 0.0, "fear": 0.4612, "surprise": 0.0564}
    with pytest.raises(UpstreamError):
        check(p, "fear")


# The precision is the backend's to state; it is inferred from the values only when unstated.

def test_a_stated_two_decimal_precision_is_used_even_for_round_looking_values():
    p = {"a": 0.3, "b": 0.69}  # 0.99
    q = {"e": {"type": "choice", "instructions": "?", "criteria": {"a": "", "b": ""}}}
    raw = {"model": "m", "answers": {"e": {"type": "choice", "choice": "b", "probabilities": p,
                                           "confidence": 0.5}}}
    request = DecideRequest(state="x", questions=q)
    assert validate_response(raw, request, precision=0.01).answers["e"].choice == "b"


def test_a_stated_four_decimal_precision_refuses_a_distribution_that_looks_two_decimal():
    p = {"a": 0.3, "b": 0.69}  # sums to 0.99, which four decimals cannot explain
    q = {"e": {"type": "choice", "instructions": "?", "criteria": {"a": "", "b": ""}}}
    raw = {"model": "m", "answers": {"e": {"type": "choice", "choice": "b", "probabilities": p,
                                           "confidence": 0.5}}}
    with pytest.raises(UpstreamError):
        validate_response(raw, DecideRequest(state="x", questions=q), precision=0.0001)
