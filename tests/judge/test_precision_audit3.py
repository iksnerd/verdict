"""Audit 3: a backend that rounds to 0.01 is a real model, so a flat-looking answer from it is
not 'no model'; is_flat takes the precision."""
from __future__ import annotations

from verdict.answers import is_flat
from verdict.backends.capabilities import jev_capabilities
from verdict.judge.router import BIG, SMALL, is_uniform, route


def answers(difficulty_probs, difficulty, sensitive):
    return {
        "difficulty": {"type": "score", "score": difficulty, "legend": {}, "probabilities": difficulty_probs,
                       "confidence": 0.0},
        "is_sensitive": {"type": "noul", "noul": sensitive, "confidence": 0.5},
    }


NEAR = {"0": .26, "1": .25, "2": .25, "3": .24}   # 0.01-rounded, genuinely unsure


def test_is_flat_precision_default_is_unchanged():
    assert not is_flat({"type": "noul", "noul": 0.505, "confidence": 0.5})
    assert is_flat({"type": "noul", "noul": 0.5, "confidence": 0.5})


def test_is_flat_with_precision_counts_one_step_as_flat():
    near = {"type": "score", "score": 1.5, "legend": {}, "probabilities": NEAR, "confidence": 0.0}
    assert not is_flat(near)
    assert is_flat(near, precision=0.02)
    assert not is_flat({"type": "noul", "noul": 0.52, "confidence": 0.5}, precision=0.01)


def test_jev_capabilities_declare_their_precision():
    assert jev_capabilities("m").precision == 0.01


def test_rounding_backend_noul_of_exactly_half_is_not_no_model():
    a = answers({"0": .1, "1": .2, "2": .3, "3": .4}, 2.0, 0.5)
    assert is_uniform(a)
    assert not is_uniform(a, precision=0.01)


def test_rounding_backend_flat_difficulty_is_not_no_model_and_is_routed_on_its_scores():
    a = answers({"0": .25, "1": .25, "2": .25, "3": .25}, 1.5, 0.05)
    assert route(a).name == BIG
    assert route(a, precision=0.01).name == SMALL
