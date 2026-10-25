"""A cascade: the primary backend (Laya, local) answers everything, and the answers it is least sure
of are asked again of a second backend (Jev). FINDINGS §52: on the 12 bench suites sending the
least-sure 20% closes 56% of the gap to Jev and half closes 92%, against 22% and 52% for a random
share, and the other texts never leave the machine.

Both backends are injected as `ask(state, questions) -> {"answers": {...}}`, the shape every front
door returns, so this module knows nothing about servers or models. Confidence is the probability of
the answer given (a yes/no: max(p, 1 - p); a choice or score: its largest probability), so an
uncalibrated model still orders its own answers; a threshold is therefore read off a share of the
batch (`share`) or fixed (`below`) rather than quoted as a calibrated probability.

It is a per-task decision. On AG News the primary beats the second backend and every item sent makes
the result worse (§52); `check` measures that on labelled examples before anything is sent.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

from . import calibrate
from ..errors import QuestionError

Ask = Callable[[Any, dict], dict]


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def confidence(answer: dict, cut: float | None = None) -> float:
    """How sure an answer is, from its own probabilities.

    A yes/no with a fitted `cut` is as sure as it is far from that cut, in logit units: Laya's
    yes/no scores are shifted (fitted cuts of 0.004 to 0.84 on the bench suites, none near 0.5), so
    distance from 0.5 ranks the wrong items as unsure (FINDINGS §55). Without a cut it is
    max(p, 1 - p); a choice or score is its largest probability."""
    if answer.get("type") == "noul":
        p = answer["noul"]
        if cut is not None:
            return abs(_logit(p) - _logit(cut))
        return max(p, 1 - p)
    return max(answer["probabilities"].values())


def _cuts(bank: dict, fits: dict | None) -> dict[str, float | None]:
    """The fitted cut of each yes/no question that has one."""
    fits = fits or {}
    return {qid: (fits[qid].get("cut") if qid in fits and fits[qid].get("type") == "noul"
                  and bank[qid]["type"] == "noul" else None) for qid in bank}


def _validate(share: float | None, below: float | None) -> None:
    if (share is None) == (below is None):
        raise QuestionError("give exactly one of share (a fraction of the answers to escalate) and "
                            "below (escalate answers less sure than this)")
    if share is not None and not 0 <= share <= 1:
        raise QuestionError(f"share is {share!r}; it is a fraction from 0 to 1")
    if below is not None and not 0 <= below <= 1:
        raise QuestionError(f"below is {below!r}; it is a confidence from 0 to 1")


def select(confidences: Sequence[float], *, share: float | None = None,
           below: float | None = None) -> set[int]:
    """Positions to escalate: those under `below`, or exactly the `share` least sure (earlier first on a tie)."""
    _validate(share, below)
    if below is not None:
        return {i for i, c in enumerate(confidences) if c < below}
    k = round(share * len(confidences))
    return set(sorted(range(len(confidences)), key=lambda i: (confidences[i], i))[:k])


class Cascade:
    """The merged rows and what the run cost."""

    def __init__(self, rows: list[dict], failed: list[tuple[int, str]], answers: int, escalated: int):
        self.rows, self.failed = rows, failed
        #: how many (state, question) answers there were, and how many came from the second backend
        self.answers, self.escalated = answers, escalated

    @property
    def states_sent(self) -> int:
        return sum(1 for r in self.rows if r["escalated"])


def run(states: Sequence[Any], bank: dict, primary: Ask, secondary: Ask, *,
        share: float | None = None, below: float | None = None, fits: dict | None = None,
        errors: tuple[type[BaseException], ...] = (ValueError,)) -> Cascade:
    """Answer every state with `primary`, then re-ask the least-sure answers of `secondary`.

    Only the escalated questions of an escalated state are sent. A secondary that fails leaves that
    state's primary answers in place and the row says so; the run carries on. `fits` (the file
    `verdict calibrate` writes) gives each yes/no its cut: the primary's answer then carries a
    `decision` at that cut, and the least sure are those nearest it. `below` is a probability, so
    it cannot be combined with a cut measured in logit units."""
    _validate(share, below)
    cuts = _cuts(bank, fits)
    if below is not None and any(c is not None for c in cuts.values()):
        raise QuestionError("with a fitted cut the uncertainty is a distance in logit units, not a "
                            "probability: use share (the fraction to escalate) instead of below")
    results = [primary(state, bank) for state in states]
    if fits:
        for result in results:
            result["answers"] = calibrate.apply(result["answers"], fits)
    chosen = {qid: select([confidence(r["answers"][qid], cuts[qid]) for r in results],
                          share=share, below=below) for qid in bank}
    rows, failed, escalated = [], [], 0
    for i, (state, result) in enumerate(zip(states, results)):
        asked = sorted(qid for qid in bank if i in chosen[qid])
        answers = {qid: {**a, "tier": "primary"} for qid, a in result["answers"].items()}
        row = {**{k: v for k, v in result.items() if k != "answers"}, "answers": answers,
               "escalated": []}
        if asked:
            try:
                second = secondary(state, {qid: bank[qid] for qid in asked})["answers"]
            except errors as exc:
                row["escalation_error"] = str(exc)
                failed.append((i, str(exc)))
            else:
                for qid in asked:
                    answers[qid] = {**second[qid], "tier": "escalated"}
                    if cuts[qid] is not None and answers[qid].get("type") == "noul":
                        # So a caller reads one field whichever tier answered; the second server's
                        # own number is decided at 0.5, as `check` measures it.
                        answers[qid].update(cut=0.5, decision=answers[qid]["noul"] >= 0.5)
                row["escalated"] = asked
                escalated += len(asked)
        rows.append(row)
    return Cascade(rows, failed, len(states) * len(bank), escalated)


def _correct(answer: dict, label: Any, cut: float | None = None) -> bool | None:
    if answer["type"] == "noul":
        return (answer["noul"] >= (0.5 if cut is None else cut)) == calibrate.as_bool(label)
    if answer["type"] == "choice":
        return max(answer["probabilities"], key=answer["probabilities"].get) == label
    return None   # a score has no single right answer to count


def check(states: Sequence[Any], bank: dict, labels: Sequence[dict], primary: Ask, secondary: Ask, *,
          share: float | None = None, below: float | None = None, fits: dict | None = None,
          holdout: float | None = None, seed: int = 0,
          errors: tuple[type[BaseException], ...] = (ValueError,)) -> dict[str, dict]:
    """Accuracy of the primary, of the cascade at this setting, and of the secondary, per question, on
    labelled examples; and whether escalating helps. Asks the secondary about every example once.

    With `holdout`, each yes/no question's cut is fitted on the rest by `calibrate.fit_noul` and every
    number is reported on the held-out part, so it is not flattered by the fit. With `fits` the cuts
    are the ones given (measured on every row). With neither, the cut is 0.5 on every row.
    `primary_raw` is always the primary at 0.5; `primary` is at the cut. `helps` is true only when the
    secondary beats the primary and so does the cascade: a tie is not a reason to send text out.
    Raises whatever `secondary` raises (`errors` is accepted for symmetry)."""
    _validate(share, below)
    first = [primary(state, bank)["answers"] for state in states]
    second = [secondary(state, bank)["answers"] for state in states]
    report = {}
    for qid, q in bank.items():
        if q["type"] == "score":
            continue
        rows = [i for i, lab in enumerate(labels) if lab.get(qid) is not None]
        if not rows:
            continue
        fit = (fits or {}).get(qid)
        if holdout and q["type"] == "noul":
            fit = calibrate.fit_noul([first[i][qid]["noul"] for i in rows],
                                     [calibrate.as_bool(labels[i][qid]) for i in rows], holdout, seed)
        cut = fit["cut"] if fit and fit.get("type") == "noul" and q["type"] == "noul" else None
        if below is not None and cut is not None:
            raise QuestionError("with a fitted cut the uncertainty is a distance in logit units, not a "
                                "probability: use share instead of below")
        if holdout:
            evaluated = [rows[j] for j in sorted(calibrate.split(len(rows), holdout, seed)[1])]
        else:
            evaluated = rows
        pick = {evaluated[j] for j in select([confidence(first[i][qid], cut) for i in evaluated],
                                             share=share, below=below)}
        p = [_correct(first[i][qid], labels[i][qid], cut) for i in evaluated]
        raw = [_correct(first[i][qid], labels[i][qid]) for i in evaluated]
        s = [_correct(second[i][qid], labels[i][qid]) for i in evaluated]
        c = [s[j] if evaluated[j] in pick else p[j] for j in range(len(evaluated))]
        n = len(evaluated)
        acc = lambda xs: sum(xs) / n
        report[qid] = {"n": n, "primary": acc(p), "primary_raw": acc(raw), "cascade": acc(c),
                       "secondary": acc(s), "escalated": len(pick), "cut": cut, "fit": fit,
                       "helps": acc(s) > acc(p) and acc(c) > acc(p)}
    return report
