"""`metrics.py` replaced two bootstrap routines that disagreed on iteration count and percentile
arithmetic. These pin the properties the scorers rely on."""
from __future__ import annotations

import pytest

from verdict.evaluation.metrics import auc, bootstrap_ci, distance, macro_f1, mean_ci


def test_auc_perfect_and_inverted_and_random():
    scores, labels = [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]
    assert auc(scores, labels) == 1.0
    assert auc([-s for s in scores], labels) == 0.0
    assert auc([0.5, 0.5, 0.5, 0.5], labels) == 0.5   # all ties count half


def test_auc_is_nan_without_both_classes():
    import math
    assert math.isnan(auc([1, 2, 3], [1, 1, 1]))
    assert math.isnan(auc([1, 2, 3], [0, 0, 0]))


def test_auc_is_threshold_free():
    """The property that made it the right statistic in §12: shifting every score leaves it alone."""
    scores, labels = [0.1, 0.3, 0.6, 0.7], [0, 0, 1, 1]
    assert auc(scores, labels) == auc([s - 10 for s in scores], labels)


def test_bootstrap_is_seeded_and_therefore_reproducible():
    """A reported interval has to be repeatable, or it is decoration."""
    hits = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    assert mean_ci(hits) == mean_ci(hits)
    assert mean_ci(hits, seed=7) == mean_ci(hits, seed=7)


def test_the_seed_actually_drives_the_resampling():
    """Not asserted on `mean_ci` of a small 0/1 sample: there the statistic takes only n+1 values,
    so two seeds land on the same percentile bounds and the test would pass for the wrong reason."""
    xs = [i / 97 for i in range(97)]
    assert bootstrap_ci(xs, lambda s: sum(s) / len(s), iters=300, seed=1) != \
           bootstrap_ci(xs, lambda s: sum(s) / len(s), iters=300, seed=2)


def test_bootstrap_interval_brackets_the_observed_value():
    hits = [1] * 18 + [0] * 6            # 0.75
    lo, hi = mean_ci(hits)
    assert lo <= 0.75 <= hi and lo < hi


def test_bootstrap_on_a_unanimous_sample_is_degenerate():
    assert mean_ci([1] * 20) == (1.0, 1.0)


def test_bootstrap_of_an_empty_sample_is_nan_not_a_crash():
    import math
    lo, hi = mean_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def test_bootstrap_takes_an_arbitrary_statistic():
    lo, hi = bootstrap_ci([1, 2, 3, 4, 5], lambda xs: max(xs), iters=200)
    assert 1 <= lo <= hi <= 5


def test_macro_f1_is_unweighted_so_a_majority_guess_scores_poorly():
    """The §3 failure: predicting one class for everything. Accuracy hides it, macro-F1 does not."""
    classes = ["a", "b", "c"]
    truth = ["a"] * 6 + ["b"] * 2 + ["c"] * 2
    assert macro_f1(truth, truth, classes) == 1.0
    always_a = macro_f1(truth, ["a"] * 10, classes)
    assert always_a < 0.35          # 0.6 accuracy, but two classes score zero


def test_macro_f1_scores_a_never_predicted_class_zero():
    # class a: tp=1 fp=1 fn=0 -> 2/3.  class b: tp=0 -> 0.  macro = 1/3.
    assert macro_f1(["a", "b"], ["a", "a"], ["a", "b"]) == pytest.approx(1 / 3)


def test_distance_handles_both_shapes_point_produces():
    assert distance(0.8, 0.5) == pytest.approx(0.3)
    assert distance({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}) == pytest.approx(1.0)
    assert distance({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
