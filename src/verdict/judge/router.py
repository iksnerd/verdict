"""Route a prompt to a big model or a small one. A `Switch` (see `switch.py`) with two cases.

The bank is `laya.router_questions()`, upstream's own, not one invented here. FINDINGS §14: its
`difficulty` question scores AUC 0.90 to 0.92 against 0.81 for the best question this project made
up, in one question rather than two, and it scores as well on the base checkpoint as on the
fine-tuned one. That is the §2 pattern: a question inside Laya's training mix runs near ceiling and
an invented one does not.

Only two of upstream's four are used. `domain` is an exclusive choice, which §3 and §9 show
collapses onto a majority class, and `needs_tools` measured *below* chance on our prompts (AUC 0.41
and 0.34): it asks about the request rather than about which model should serve it.

So this is the worked example of the point `switch.py` makes. The interface is two cases and a
default. The mechanism underneath is not the obvious exclusive choice over {"small", "big"} but a
calibrated cut on a `score` question, because that is what measured better.

The definitions are copied here rather than imported so that importing this module, and the whole
test suite, needs neither `laya` nor torch. `tests/judge/test_router.py::test_bank_matches_upstream`
asserts the copy still matches `laya.router_questions()` whenever `laya` is installed, so it cannot
drift silently.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..answers import field, is_flat
from ..backends.engine import PROMPT_TOKEN_BUDGET, load
from .switch import Branch, Switch

SMALL = "small"
BIG = "big"

#: Verbatim from `laya.router_questions()` in laya 0.3.4. Do not reword: the value of these two is
#: that they are phrased the way the checkpoint was trained.
ROUTER_BANK: dict[str, dict[str, Any]] = {
    "difficulty": {
        "type": "score",
        "instructions": "How hard is `request` for a language model?",
        "criteria": [
            "trivial: a lookup or one-liner",
            "easy: short answer, no reasoning",
            "moderate: several steps",
            "hard: long multi-step reasoning or specialist knowledge",
        ],
    },
    "is_sensitive": {
        "type": "noul",
        "instructions": "Does `request` involve money, legal, medical or safety consequences?",
    },
}

#: Fitted on the training split of `evals/router_prompts.jsonl` and reported on a held-out split it
#: never saw. `python -m verdict.evaluation.router_eval --refit` reproduces both numbers and refits these. The fit minimises
#: expected *cost*, not accuracy: a hard prompt sent to the small model counts 3x an easy one sent
#: to the big model. Fitting on accuracy instead scored 4 points higher and was a worse router
#: (FINDINGS §15).
#: `difficulty` runs 0 (trivial) to 3 (hard); `is_sensitive` is a probability.
T_DIFFICULTY = 1.6
T_SENSITIVE = 0.3

CASES = {
    SMALL: "a lookup, a definition, a conversion, a one-liner",
    BIG: "multi-step reasoning, code changes, or anything with consequences",
}

#: There is no table of commands to run. verdict answers and the caller acts: a script that
#: wants the small branch to reach a local model writes that one line itself.


def is_uniform(answers: Mapping[str, Any]) -> bool:
    """True when every answer is maximum uncertainty, i.e. there is no model behind this.

    This matters more than it looks. A uniform `difficulty` answer scores 1.5, which is *below*
    `T_DIFFICULTY`, so without this check a backend with no model behind it would route every
    prompt to the small model, which is the expensive direction. `answers.is_flat` explains why
    the check reads the distribution rather than `confidence`.
    """
    return all(is_flat(answers[qid]) for qid in ROUTER_BANK)


def make_selector(t_difficulty: float = T_DIFFICULTY, t_sensitive: float = T_SENSITIVE):
    """The mechanism: a calibrated cut on `difficulty`, with `is_sensitive` as a safety override."""

    def select(answers: Mapping[str, Any]) -> Branch | None:
        if is_uniform(answers):
            return None  # the Switch turns this into `default`, i.e. BIG
        difficulty = float(field(answers["difficulty"], "score"))
        sensitive = float(field(answers["is_sensitive"], "noul"))
        scores = {"difficulty": difficulty, "sensitive": sensitive}
        if sensitive >= t_sensitive:
            return Branch(BIG, f"money, legal, medical or safety consequences (p={sensitive:.2f})", scores)
        if difficulty >= t_difficulty:
            return Branch(BIG, f"too hard for a small model (difficulty {difficulty:.2f}/3)", scores)
        return Branch(SMALL, f"easy (difficulty {difficulty:.2f}/3) and not sensitive", scores)

    return select


BIG_SMALL = Switch(
    name="big-vs-small",
    cases=CASES,
    default=BIG,
    questions=ROUTER_BANK,
    select=make_selector(),
)


def route(
    answers: Mapping[str, Any],
    *,
    t_difficulty: float = T_DIFFICULTY,
    t_sensitive: float = T_SENSITIVE,
) -> Branch:
    """`BIG_SMALL.decide`, with the thresholds exposed so the scorer can sweep them."""
    if (t_difficulty, t_sensitive) == (T_DIFFICULTY, T_SENSITIVE):
        return BIG_SMALL.decide(answers)
    switch = Switch(
        name=BIG_SMALL.name, cases=CASES, default=BIG, questions=ROUTER_BANK,
        select=make_selector(t_difficulty, t_sensitive),
    )
    return switch.decide(answers)


def clip(prompt: str, tokenizer, budget: int = PROMPT_TOKEN_BUDGET) -> str:
    """Deprecated shim. `Engine.clip` owns this now; kept for callers holding a bare tokenizer."""
    ids = tokenizer(prompt)["input_ids"]
    if len(ids) <= budget:
        return prompt
    return tokenizer.decode(ids[1 : budget + 1], skip_special_tokens=True)


def engine_for(model_id: str):
    """The Engine this router is meant to be driven by."""
    return load(model_id)
