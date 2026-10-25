"""Scoring functions the scorers share.

One implementation of each, so two scorers can never report different numbers for the same
intent: the bootstrap interval takes its sample count at the call site, and every scorer shares
the same total-variation distance.
"""
from __future__ import annotations

import random
from typing import Callable, Sequence


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Probability that a positive outranks a negative, ties counting half.

    Threshold-free, which is why it is the statistic to look at first on a model that ships
    uncalibrated: FINDINGS §2 and §12 both found good AUC sitting behind useless accuracy because
    the decision threshold was wrong rather than the ranking.
    """
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def bootstrap_ci(
    items: Sequence,
    statistic: Callable[[Sequence], float],
    *,
    iters: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for `statistic` over a resample of `items`.

    Seeded, so a reported interval is reproducible rather than merely plausible.
    """
    rng = random.Random(seed)
    n = len(items)
    if n == 0:
        return (float("nan"), float("nan"))
    values = sorted(statistic([items[rng.randrange(n)] for _ in range(n)]) for _ in range(iters))
    lo = values[int((alpha / 2) * iters)]
    hi = values[min(int((1 - alpha / 2) * iters), iters - 1)]
    return lo, hi


def mean_ci(hits: Sequence[float], **kw) -> tuple[float, float]:
    """Bootstrap interval for a mean, the common case (an accuracy over 0/1 outcomes)."""
    return bootstrap_ci(hits, lambda xs: sum(xs) / len(xs), **kw)


def macro_f1(truth: Sequence[str], predicted: Sequence[str], classes: Sequence[str]) -> float:
    """Unweighted mean F1 over `classes`. A class predicted for everything still scores poorly here,
    which is the point on a task where §3 found exactly that failure."""
    out = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(truth, predicted))
        fp = sum(t != c and p == c for t, p in zip(truth, predicted))
        fn = sum(t == c and p != c for t, p in zip(truth, predicted))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(out) / len(out) if out else float("nan")


def distance(a: float | dict[str, float], b: float | dict[str, float]) -> float:
    """Total variation between two distributions, or absolute error between two probabilities.

    Takes the shapes `answers.point` produces, so a yes/no question is compared as the one number
    the teacher actually labelled rather than as a two-option distribution.
    """
    if isinstance(a, dict):
        return 0.5 * sum(abs(a[k] - b[k]) for k in a)
    return abs(a - b)
