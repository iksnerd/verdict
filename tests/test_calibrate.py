"""The fits a gate is built on. Pure arithmetic, so it is checked against hand-worked cases."""
from __future__ import annotations

import math

import pytest

from verdict import calibrate as cal


def test_auc_perfect_inverted_and_tied():
    assert cal.auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert cal.auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert cal.auc([0.5, 0.5], [False, True]) == 0.5
    assert cal.auc([0.1, 0.2], [True, True]) is None


def test_fit_cut_separates_a_separable_set():
    cut = cal.fit_cut([0.30, 0.35, 0.40, 0.55, 0.60], [False, False, False, True, True])
    assert 0.40 < cut < 0.55


def test_fit_cut_is_balanced_not_majority():
    """91/9 skew: plain accuracy would put the cut above everything and never say yes."""
    scores = [0.3] * 91 + [0.6] * 9
    labels = [False] * 91 + [True] * 9
    assert cal.fit_cut(scores, labels) < 0.6


def test_temper_keeps_argmax_and_sharpens_below_one():
    p = {"a": 0.6, "b": 0.3, "c": 0.1}
    sharp, flat = cal.temper(p, 0.5), cal.temper(p, 2.0)
    assert max(sharp, key=sharp.get) == max(flat, key=flat.get) == "a"
    assert sharp["a"] > 0.6 > flat["a"]
    assert math.isclose(sum(sharp.values()), 1.0)


def test_fit_temperature_flattens_an_overconfident_model():
    """Right 60% of the time at 0.95 confidence: the fitted temperature must exceed 1."""
    rows = [{"a": 0.95, "b": 0.05}] * 10
    labels = ["a"] * 6 + ["b"] * 4
    assert cal.fit_temperature(rows, labels) > 1


def test_split_is_reproducible_and_disjoint():
    a, b = cal.split(10, 0.3, seed=1)
    assert (a, b) == cal.split(10, 0.3, seed=1)
    assert not set(a) & set(b) and len(a) + len(b) == 10


@pytest.mark.parametrize("label,expected", [("yes", True), ("No", False), (1, True), (False, False)])
def test_as_bool(label, expected):
    assert cal.as_bool(label) is expected


def test_as_bool_rejects_nonsense():
    with pytest.raises(ValueError):
        cal.as_bool("maybe")


def test_apply_adds_without_overwriting():
    answers = {
        "urgent": {"type": "noul", "noul": 0.44, "confidence": 0.44},
        "intent": {"type": "choice", "choice": "a", "probabilities": {"a": 0.6, "b": 0.4},
                   "confidence": 0.12},
        "other": {"type": "noul", "noul": 0.9, "confidence": 0.9},
    }
    fits = {"urgent": {"type": "noul", "cut": 0.41}, "intent": {"type": "choice", "temperature": 2.0}}
    out = cal.apply(answers, fits)
    assert out["urgent"]["decision"] is True and out["urgent"]["noul"] == 0.44
    assert out["intent"]["confidence"] == 0.12
    assert out["intent"]["calibrated_confidence"] < 0.6
    assert "decision" not in out["other"]


def test_fit_choice_reports_per_option_recall_and_top_confusions():
    """One accuracy number for a 7-way choice says nothing about which options to trust
    (the commit-type run, FINDINGS §31). The argmax ignores temperature, so all rows count."""
    rows = ([{"fix": 0.8, "feat": 0.2}] * 8 + [{"fix": 0.3, "feat": 0.7}] * 2 +   # fix: 8/10
            [{"fix": 0.6, "feat": 0.4}] * 6 + [{"fix": 0.1, "feat": 0.9}] * 4)    # feat: 4/10
    labels = ["fix"] * 10 + ["feat"] * 10
    fit = cal.fit_choice(rows, labels)
    assert fit["per_option"]["fix"] == {"n": 10, "recall": 0.8}
    assert fit["per_option"]["feat"] == {"n": 10, "recall": 0.4}
    assert fit["confusions"][0] == {"true": "feat", "predicted": "fix", "n": 6}


def test_the_saved_cut_is_the_one_that_was_scored():
    """A cut rounded after it was scored can land on the wrong side of the data: negatives at
    0.5000 and positives at 0.5001 reported 100% and shipped a cut that scored 50%."""
    scores = [0.5] * 10 + [0.5001] * 10
    labels = [False] * 10 + [True] * 10
    fit = cal.fit_noul(scores, labels)
    assert fit["train"]["balanced_accuracy"] == 1.0
    assert cal.balanced_accuracy(scores, labels, fit["cut"]) == 1.0


@pytest.mark.parametrize("label", [False, True], ids=["all no", "all yes"])
def test_a_single_class_sample_is_refused(label):
    """With one class there is no cut to fit, and the one fit_cut picks is arbitrary."""
    with pytest.raises(ValueError, match="both"):
        cal.fit_noul([0.1, 0.2, 0.3, 0.4], [label] * 4)


def test_a_split_that_leaves_training_one_class_is_refused():
    """Both classes overall is not enough: the shuffle can move every positive to held-out, and a
    cut fitted on negatives alone is arbitrary."""
    labels = [False] * 10
    labels[7] = True
    train, _ = cal.split(10)
    assert not any(labels[i] for i in train)  # the case under test: the one positive is held out
    with pytest.raises(ValueError, match="training"):
        cal.fit_noul([0.1 * i for i in range(10)], labels)


def test_inverted_scores_can_choose_a_constant_classifier():
    scores, labels = [0.1, 0.9], [True, False]
    cut = cal.fit_cut(scores, labels)
    assert cal.balanced_accuracy(scores, labels, cut) == 0.5


@pytest.mark.parametrize("rows,labels", [([], []), ([{"a": .5, "b": .5}], []),
                                        ([{"a": .5, "b": .5}], ["missing"])])
def test_choice_fit_refuses_invalid_training_data(rows, labels):
    with pytest.raises(ValueError):
        cal.fit_choice(rows, labels)


@pytest.mark.parametrize("heldout", [-.1, 1, 2, float("nan")])
def test_invalid_split_fraction_is_refused(heldout):
    with pytest.raises(ValueError, match="heldout"):
        cal.fit_choice([{"a": .7, "b": .3}] * 10, ["a"] * 10, heldout=heldout)
