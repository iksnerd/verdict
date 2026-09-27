"""Fit the numbers you gate on, on your own labelled examples, and apply them at answer time.

Both laya's documentation and this project's findings say the same thing from different ends.
Laya: the base checkpoint's raw ECE is 0.213 and reaches 0.081 only "after domain temperature
fitting". Here: a default threshold has twice produced chance accuracy on a signal that ranked
perfectly well (FINDINGS §2, §12). So a score is not a decision until a cut is fitted for it.

Two fits, one per question type where a fit means something:

- `noul`: a cut, the threshold that maximises balanced accuracy on the training split. Balanced
  rather than plain accuracy because real labels are skewed (91/9 in §25), and plain accuracy
  there is maximised by never saying yes.
- `choice`: a temperature, the single scalar that minimises negative log-likelihood. It leaves
  the argmax alone and makes the probabilities mean what they say.

`score` gets neither: an ordinal expectation has no single cut, and nothing here has measured
which one to fit. Every fit is reported on a held-out split it never saw.

Pure Python, no numpy: a few hundred examples, and this module is imported by the CLI.
"""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

TRUE = {True, 1, "1", "true", "yes", "y", "t"}
FALSE = {False, 0, "0", "false", "no", "n", "f"}


def as_bool(label: Any) -> bool:
    key = label.lower() if isinstance(label, str) else label
    if not isinstance(key, (str, bool, int, float)):
        raise ValueError(f"{label!r} is not a yes/no label")
    if key in TRUE:
        return True
    if key in FALSE:
        return False
    raise ValueError(f"{label!r} is not a yes/no label")


def split(n: int, heldout: float = 0.3, seed: int = 0) -> tuple[list[int], list[int]]:
    """Indices for train and held-out, shuffled with a fixed seed so a refit is reproducible."""
    if n < 1:
        raise ValueError("at least one labelled example is required")
    if not math.isfinite(heldout) or not 0 <= heldout < 1:
        raise ValueError("heldout must be at least 0 and less than 1")
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = min(n - 1, max(1, round(n * heldout))) if n > 1 and heldout else 0
    return idx[cut:], idx[:cut]


def auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Probability a random positive outscores a random negative. None with one class only."""
    ranked = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    # Rank-sum with average ranks for ties.
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def balanced_accuracy(scores: Sequence[float], labels: Sequence[bool], cut: float) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    tpr = sum(s >= cut for s in pos) / len(pos) if pos else 0.0
    tnr = sum(s < cut for s in neg) / len(neg) if neg else 0.0
    return (tpr + tnr) / 2 if pos and neg else (tpr or tnr)


def fit_cut(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Maximise balanced accuracy, including the always-yes and always-no classifiers."""
    _paired(scores, labels)
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("yes/no probabilities must be finite and between 0 and 1")
    distinct = sorted(set(scores))
    candidates = [distinct[0], math.nextafter(distinct[-1], math.inf)]
    candidates += [(a + b) / 2 for a, b in zip(distinct, distinct[1:])]
    return max(candidates, key=lambda c: (balanced_accuracy(scores, labels, c), -abs(c - 0.5)))


def temper(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    """p_i^(1/T), renormalised: the same argmax, sharper for T < 1 and flatter for T > 1."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not probabilities or any(not math.isfinite(p) or not 0 <= p <= 1
                                for p in probabilities.values()) or not sum(probabilities.values()):
        raise ValueError("probabilities must be finite, between 0 and 1, with a positive sum")
    logs = {k: math.log(max(p, 1e-12)) for k, p in probabilities.items()}
    peak = max(logs.values())
    powered = {k: math.exp((v - peak) / temperature) for k, v in logs.items()}
    total = sum(powered.values())
    return {k: v / total for k, v in powered.items()}


def nll(rows: Sequence[dict[str, float]], labels: Sequence[str], temperature: float = 1.0) -> float:
    _paired(rows, labels)
    if any(y not in row for row, y in zip(rows, labels)):
        raise ValueError("each choice label must name an option in its probability distribution")
    return -sum(math.log(max(temper(p, temperature)[y], 1e-12))
                for p, y in zip(rows, labels)) / len(rows)


def fit_temperature(rows: Sequence[dict[str, float]], labels: Sequence[str]) -> float:
    """Grid search over log-spaced temperatures from 0.1 to 10; one scalar needs nothing finer."""
    grid = [10 ** (k / 40) for k in range(-40, 41)]
    return min(grid, key=lambda t: nll(rows, labels, t))


def fit_noul(scores: list[float], labels: list[bool], heldout: float = 0.3,
             seed: int = 0) -> dict[str, Any]:
    _paired(scores, labels)
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("yes/no probabilities must be finite and between 0 and 1")
    if all(labels) or not any(labels):
        raise ValueError(f"a yes/no cut needs both yes and no labels; all {len(labels)} are "
                         f"{'yes' if labels and labels[0] else 'no'}")
    train, test = split(len(scores), heldout, seed)
    s_tr, y_tr = [scores[i] for i in train], [labels[i] for i in train]
    s_te, y_te = [scores[i] for i in test], [labels[i] for i in test]
    if all(y_tr) or not any(y_tr):
        raise ValueError(f"the training split ({len(train)} of {len(labels)}) has only "
                         f"{'yes' if y_tr and y_tr[0] else 'no'} labels; label more of the "
                         f"rarer class or lower --heldout")
    # Saved exactly as scored: a rounded cut can fall on the far side of tightly packed scores.
    cut = fit_cut(s_tr, y_tr)
    return {
        "type": "noul", "cut": cut,
        "n": len(scores), "positives": sum(labels),
        "train": {"n": len(train), "auc": _r(auc(s_tr, y_tr)),
                  "balanced_accuracy": _r(balanced_accuracy(s_tr, y_tr, cut))},
        "heldout": {"n": len(test), "auc": _r(auc(s_te, y_te)),
                    "balanced_accuracy": _r(balanced_accuracy(s_te, y_te, cut)) if test else None,
                    "accuracy": _r(sum((s >= cut) == y for s, y in zip(s_te, y_te)) / len(test))
                    if test else None},
    }


def fit_choice(rows: list[dict[str, float]], labels: list[str], heldout: float = 0.3,
               seed: int = 0) -> dict[str, Any]:
    _paired(rows, labels)
    nll(rows, labels)  # Validate every label, including those destined for the held-out split.
    train, test = split(len(rows), heldout, seed)
    r_tr, y_tr = [rows[i] for i in train], [labels[i] for i in train]
    r_te, y_te = [rows[i] for i in test], [labels[i] for i in test]
    t = fit_temperature(r_tr, y_tr)
    acc = (lambda rs, ys: sum(max(r, key=r.get) == y for r, y in zip(rs, ys)) / len(ys)
           if ys else None)
    # Per option, over every row: the argmax ignores the temperature, so the fit cannot leak into
    # these. One accuracy figure for a many-way choice hides which options are usable at all.
    from collections import Counter

    predicted = [max(r, key=r.get) for r in rows]
    per_option = {}
    for option in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == option]
        per_option[option] = {"n": len(idx),
                              "recall": _r(sum(predicted[i] == option for i in idx) / len(idx))}
    confusions = [{"true": truth, "predicted": guess, "n": n} for (truth, guess), n in
                  Counter((y, g) for y, g in zip(labels, predicted) if y != g).most_common(5)]
    return {
        "type": "choice", "temperature": round(t, 4), "n": len(rows),
        "per_option": per_option, "confusions": confusions,
        "train": {"n": len(train), "accuracy": _r(acc(r_tr, y_tr)),
                  "nll_raw": _r(nll(r_tr, y_tr)), "nll_fitted": _r(nll(r_tr, y_tr, t))},
        "heldout": {"n": len(test), "accuracy": _r(acc(r_te, y_te)),
                    "nll_raw": _r(nll(r_te, y_te)) if test else None,
                    "nll_fitted": _r(nll(r_te, y_te, t)) if test else None},
    }


def apply(answers: dict[str, Any], fits: dict[str, Any]) -> dict[str, Any]:
    """Add each fit to its answer, without overwriting what the model said.

    A `noul` gains `decision` and `cut`. A `choice` gains `calibrated` probabilities and
    `calibrated_confidence` (their maximum); laya's own `confidence` is a different quantity and
    stays as it was.
    """
    out = {}
    for qid, a in answers.items():
        fit = fits.get(qid)
        a = dict(a)
        if fit and fit["type"] == "noul" and a.get("type") == "noul":
            a["cut"] = fit["cut"]
            a["decision"] = a["noul"] >= fit["cut"]
        elif fit and fit["type"] == "choice" and a.get("type") == "choice":
            cal = temper(a["probabilities"], fit["temperature"])
            a["calibrated"] = {k: round(v, 4) for k, v in cal.items()}
            a["calibrated_confidence"] = round(max(cal.values()), 4)
        out[qid] = a
    return out


def _r(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def _paired(rows: Sequence, labels: Sequence) -> None:
    if not rows or len(rows) != len(labels):
        raise ValueError("need non-empty examples and one label per example")
