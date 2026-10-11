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

from typing import Any, Callable, Sequence

from . import calibrate
from .errors import QuestionError

Ask = Callable[[Any, dict], dict]


def confidence(answer: dict) -> float:
    """How sure an answer is, from its own probabilities."""
    if answer.get("type") == "noul":
        p = answer["noul"]
        return max(p, 1 - p)
    return max(answer["probabilities"].values())


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
        share: float | None = None, below: float | None = None,
        errors: tuple[type[BaseException], ...] = (ValueError,)) -> Cascade:
    """Answer every state with `primary`, then re-ask the least-sure answers of `secondary`.

    Only the escalated questions of an escalated state are sent. A secondary that fails leaves that
    state's primary answers in place and the row says so; the run carries on."""
    _validate(share, below)
    results = [primary(state, bank) for state in states]
    chosen = {qid: select([confidence(r["answers"][qid]) for r in results], share=share, below=below)
              for qid in bank}
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
                row["escalated"] = asked
                escalated += len(asked)
        rows.append(row)
    return Cascade(rows, failed, len(states) * len(bank), escalated)


def _correct(answer: dict, label: Any) -> bool | None:
    if answer["type"] == "noul":
        return (answer["noul"] >= 0.5) == calibrate.as_bool(label)
    if answer["type"] == "choice":
        return max(answer["probabilities"], key=answer["probabilities"].get) == label
    return None   # a score has no single right answer to count


def check(states: Sequence[Any], bank: dict, labels: Sequence[dict], primary: Ask, secondary: Ask, *,
          share: float | None = None, below: float | None = None,
          errors: tuple[type[BaseException], ...] = (ValueError,)) -> dict[str, dict]:
    """Accuracy of the primary, of the cascade at this setting, and of the secondary, per question, on
    labelled examples; and whether escalating helps. Asks the secondary about every example once.

    `helps` is true only when the secondary beats the primary and so does the cascade: a tie is not a
    reason to send text out. Raises whatever `secondary` raises (`errors` is accepted for symmetry)."""
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
        pick = {rows[j] for j in select([confidence(first[i][qid]) for i in rows],
                                        share=share, below=below)}
        p = [_correct(first[i][qid], labels[i][qid]) for i in rows]
        s = [_correct(second[i][qid], labels[i][qid]) for i in rows]
        c = [s[j] if rows[j] in pick else p[j] for j in range(len(rows))]
        n = len(rows)
        acc = lambda xs: sum(xs) / n
        report[qid] = {"n": n, "primary": acc(p), "cascade": acc(c), "secondary": acc(s),
                       "escalated": len(pick),
                       "helps": acc(s) > acc(p) and acc(c) > acc(p)}
    return report
