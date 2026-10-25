"""A switch statement whose branch is chosen by a model instead of by `==`.

This is the shape every problem here takes. You do not ask a model to think about your prompt;
you declare the branches you are willing to take, name the one to fall back to, and get one of
them back with numbers attached:

    BIG_SMALL = Switch(
        name="big-vs-small",
        cases={"small": "a lookup, a definition, a one-liner",
               "big":   "multi-step reasoning, code changes, anything with consequences"},
        default="big",
        questions=...,
        select=...,
    )

Three things the declaration buys, all of which a free-text answer has to re-earn every call:

- **The branches are closed.** The return value is a key of `cases`, checked. There is no parsing
  step and therefore no parse failure.
- **`default` is mandatory**, and it is the branch taken when the model is unsure or absent. Most
  routing problems have asymmetric cost, so the safe branch should be written down once, in the
  declaration, rather than recovered at each call site.
- **The mechanism is separate from the interface.** `select` decides *how* the branch is chosen.

That last point is the whole reason this is not one function. The obvious implementation of a
switch is one exclusive `choice` question listing the cases, and on this project's data that is
the *worst* available mechanism: §3 and §9 measured it collapsing onto a majority class (one
branch taken for 59% of inputs where the true share was a sixth) and shifting several points when
the options were merely reordered. `choice_switch` below builds that, because sometimes it is what
you want and it takes one line, but `Switch` does not assume it. The big-vs-small router in
`router.py` is the same interface compiled to a calibrated threshold on a `score` question, which
measured far better (§14, AUC 0.92 against 0.81).

So: declare the problem as a switch, then measure which mechanism serves it, and change the
mechanism without touching the call sites.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from ..answers import field as answer_field, is_flat, probs as answer_probs


class Branch:
    """The branch taken, why, and the numbers behind it.

    A plain class rather than a dataclass: `dataclasses` pulls in `inspect`, which costs 3.7 ms
    of the CLI's 21.8 ms of imports, and this type is three fields with no generated behaviour
    worth that. Equality and repr are written out so nothing downstream changes.
    """

    __slots__ = ("name", "reason", "scores")

    def __init__(self, name: str, reason: str, scores: dict[str, float] | None = None):
        self.name = name
        self.reason = reason
        self.scores = scores if scores is not None else {}

    def as_dict(self) -> dict[str, Any]:
        return {"branch": self.name, "reason": self.reason,
                "scores": {k: round(v, 4) for k, v in self.scores.items()}}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Branch):
            return NotImplemented
        return (self.name, self.reason, self.scores) == (other.name, other.reason, other.scores)

    def __repr__(self) -> str:
        return f"Branch(name={self.name!r}, reason={self.reason!r}, scores={self.scores!r})"


#: A mechanism: given the backend's answers, return the branch and the reason, or None to mean
#: "I cannot tell", which the Switch turns into `default`.
Select = Callable[[Mapping[str, Any]], "Branch | None"]


class Switch:
    """Cases in, one case out. See the module docstring for why `select` is a parameter."""

    def __init__(
        self,
        *,
        name: str,
        cases: dict[str, str],
        default: str,
        questions: dict[str, dict[str, Any]],
        select: Select,
    ):
        if len(cases) < 2:
            raise ValueError("a switch needs at least 2 cases")
        if default not in cases:
            raise ValueError(f"default {default!r} is not one of the cases {sorted(cases)}")
        if not questions:
            raise ValueError("a switch needs at least one question")
        self.name = name
        self.cases = dict(cases)
        self.default = default
        self.questions = dict(questions)
        self._select = select

    def decide(self, answers: Mapping[str, Any]) -> Branch:
        """Pick a branch. Anything the mechanism cannot decide becomes `default`."""
        for qid in self.questions:
            if qid not in answers:
                raise KeyError(f"switch {self.name!r} needs an answer for {qid!r}")
        branch = self._select(answers)
        if branch is None:
            return Branch(self.default, "undecided, falling back to the default branch")
        if branch.name not in self.cases:
            raise ValueError(f"mechanism returned {branch.name!r}, not one of {sorted(self.cases)}")
        return branch

    def __repr__(self) -> str:
        return f"Switch({self.name!r}, cases={sorted(self.cases)}, default={self.default!r})"


def choice_switch(
    name: str,
    instructions: str,
    cases: dict[str, str],
    default: str,
    *,
    min_confidence: float = 0.0,
) -> Switch:
    """The one-line switch: an exclusive `choice` question listing the cases.

    Convenient, and the weakest mechanism measured here. §3 and §9: an exclusive choice collapses
    onto a majority class and moves several points when the options are reordered. Prefer it for a
    first cut, then check whether a calibrated threshold on a `score` or yes/no question does
    better, as it did for big-vs-small (§14).

    `min_confidence` falls back to `default` below the given confidence. It defaults to 0.0, i.e.
    off, because for a yes/no answer Laya reports `confidence == max(p, 1 - p)` and a confidence
    gate is then the same test as a probability threshold; set it deliberately if you want it.
    """
    qid = f"{name}_case"

    def select(answers: Mapping[str, Any]) -> Branch | None:
        a = answers[qid]
        if is_flat(a):
            return None  # no model behind this
        if float(answer_field(a, "confidence")) < min_confidence:
            return None
        p = answer_probs(a)
        pick = str(answer_field(a, "choice"))
        return Branch(pick, f"chose {pick} (p={p.get(pick, float('nan')):.2f})", p)

    return Switch(
        name=name,
        cases=cases,
        default=default,
        questions={qid: {"type": "choice", "instructions": instructions, "criteria": dict(cases)}},
        select=select,
    )
