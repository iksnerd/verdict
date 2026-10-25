"""Hierarchical classification: a coarse choice among groups, then a fine one inside the best few.

`ask`/`decide` default to at most 20 options. A tree helps Banking77 but loses to a flat choice
on CLINC150 (FINDINGS §50, §58); compare both on labelled examples. Two levels of at most 20 reach up to
400 labels at the price of one pass for the groups and one per group tried. The best leaf is the
one with the highest joint probability, P(group) * P(option | group), over the `beam` best groups:
a wider beam recovers a leaf whose group was ranked second.

The model is injected as `ask(state, questions) -> {"answers": {...}}`, the shape every front door
returns, so the command, the Python client and the tests share this one function.
"""
from __future__ import annotations

from typing import Any, Callable

from ..errors import QuestionError

#: Conservative default for each level, matching `library.lint`; not a measured accuracy cliff.
MAX_LEVEL = 20

Ask = Callable[[Any, dict], dict]


class Classification:
    """The best leaf, runners-up, and the first-stage scores and groups searched."""

    def __init__(self, candidates: list[dict], passes: int, *,
                 group_probabilities: dict[str, float] | None = None,
                 searched_groups: list[str] | None = None):
        self.candidates = candidates
        self.passes = passes
        self.group_probabilities = dict(group_probabilities or {})
        self.searched_groups = list(searched_groups or [])

    @property
    def label(self) -> str:
        return self.candidates[0]["label"]

    @property
    def path(self) -> list[str]:
        return self.candidates[0]["path"]

    @property
    def probability(self) -> float:
        return self.candidates[0]["probability"]

    def as_dict(self) -> dict:
        return {"label": self.label, "path": self.path, "probability": round(self.probability, 4),
                "passes": self.passes,
                "group_probabilities": dict(self.group_probabilities),
                "searched_groups": list(self.searched_groups),
                "candidates": [{**c, "probability": round(c["probability"], 4)}
                               for c in self.candidates]}


def validate(tree: Any) -> None:
    """Refuse a tree the model cannot read well, naming the group at fault."""
    groups = tree.get("groups") if isinstance(tree, dict) else None
    if not isinstance(groups, dict) or not groups:
        raise QuestionError('a tree is {"instructions": "...", "groups": {NAME: {"description": '
                            '"...", "options": {LABEL: "description or null"}}}}; "groups" is '
                            "missing or empty")
    if len(groups) > MAX_LEVEL:
        raise QuestionError(f"{len(groups)} groups; one level takes at most {MAX_LEVEL}. Nest the "
                            "groups into broader ones, or merge some")
    seen: dict[str, str] = {}
    for name, group in groups.items():
        options = group.get("options") if isinstance(group, dict) else None
        if not isinstance(options, dict) or not options:
            raise QuestionError(f"group {name!r} has no options; each group needs "
                                '"options": {LABEL: "description or null"}')
        if len(options) > MAX_LEVEL:
            raise QuestionError(f"group {name!r} has {len(options)} options; one level takes at "
                                f"most {MAX_LEVEL}. Split the group")
        for label in options:
            if label in seen:
                raise QuestionError(f"label {label!r} is in groups {seen[label]!r} and {name!r}; "
                                    "a label must belong to one group")
            seen[label] = name


def _describe(mapping: dict, key: str = "description") -> dict[str, str]:
    return {k: (v.get(key) if isinstance(v, dict) else v) or "" for k, v in mapping.items()}


def _choice(ask: Ask, state: Any, instructions: str, criteria: dict, qid: str) -> dict[str, float]:
    question = {qid: {"type": "choice", "instructions": instructions, "criteria": criteria}}
    return ask(state, question)["answers"][qid]["probabilities"]


def classify(ask: Ask, state: Any, tree: dict, *, beam: int = 2) -> Classification:
    """Classify `state` into one leaf of `tree`. `beam` groups are descended; 1 is the cheapest."""
    if not isinstance(beam, int) or beam < 1:
        raise QuestionError(f"beam is {beam!r}; it takes a whole number from 1 (one group, the "
                            "fewest passes) up")
    validate(tree)
    instructions = tree.get("instructions") or ""
    groups = tree["groups"]
    group_p = _choice(ask, state, instructions, _describe(groups), "group")
    passes = 1
    candidates = []
    searched_groups = sorted(groups, key=lambda g: group_p.get(g, 0.0), reverse=True)[:beam]
    for name in searched_groups:
        options = groups[name]["options"]
        if len(options) == 1:
            (label,) = options
            candidates.append({"label": label, "path": [name, label],
                               "probability": group_p.get(name, 0.0)})
            continue
        option_p = _choice(ask, state, instructions, _describe(options, "description"), "option")
        passes += 1
        for label in options:
            candidates.append({"label": label, "path": [name, label],
                               "probability": group_p.get(name, 0.0) * option_p.get(label, 0.0)})
    candidates.sort(key=lambda c: c["probability"], reverse=True)
    return Classification(candidates[:5], passes, group_probabilities=group_p,
                          searched_groups=searched_groups)
