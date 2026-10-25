"""The routing policy is pure, so it is tested without a model, like every other test here."""
from __future__ import annotations

import json
import pathlib

import pytest

from verdict.backends.base import UniformBackend
from verdict.judge.router import (
    BIG,
    BIG_SMALL,
    CASES,
    ROUTER_BANK,
    SMALL,
    T_DIFFICULTY,
    T_SENSITIVE,
    is_uniform,
    route,
)
from verdict.schema import DecideRequest

EVALS = pathlib.Path(__file__).resolve().parents[1] / "evals" / "router_prompts.jsonl"


def answers(difficulty=0.5, sensitive=0.1):
    """Shaped like a real laya-mlx answer: a score question and a yes/no one."""
    return {
        "difficulty": {
            "type": "score", "score": difficulty,
            "probabilities": {"0": 0.5, "1": 0.3, "2": 0.15, "3": 0.05},
        },
        "is_sensitive": {"type": "noul", "noul": sensitive},
    }


def test_easy_and_not_sensitive_goes_small():
    b = route(answers())
    assert b.name == SMALL and b.scores["difficulty"] == 0.5


@pytest.mark.parametrize(
    "kwargs", [{"difficulty": 2.5}, {"sensitive": 0.9}], ids=["too hard", "sensitive"]
)
def test_either_signal_escalates(kwargs):
    assert route(answers(**kwargs)).name == BIG


def test_sensitivity_overrides_an_easy_prompt():
    """A trivial question about money still goes big. That is the asymmetric cost, on purpose."""
    assert route(answers(difficulty=0.1, sensitive=0.95)).name == BIG


def test_thresholds_are_tunable():
    assert route(answers(difficulty=2.0), t_difficulty=3.0).name == SMALL
    assert route(answers(difficulty=0.5), t_difficulty=0.1).name == BIG


def test_the_uniform_backend_can_never_route_small():
    """The trap this guards: a uniform `difficulty` scores 1.5, which is BELOW T_DIFFICULTY,
    so without the flat-distribution check a backend with no model would route everything small."""
    resp = UniformBackend().decide(DecideRequest(state="anything", questions=ROUTER_BANK))
    assert resp.answers["difficulty"].score < T_DIFFICULTY   # the trap is real
    assert is_uniform(resp.answers)
    assert route(resp.answers).name == BIG


def test_a_real_answer_is_not_mistaken_for_uniform():
    assert not is_uniform(answers())


def test_route_accepts_pydantic_answers_and_plain_dicts_alike():
    resp = UniformBackend().decide(DecideRequest(state="x", questions=ROUTER_BANK))
    assert route(resp.answers).name == BIG
    assert route(answers()).name == SMALL


def test_missing_answer_is_an_error_not_a_silent_escalation():
    with pytest.raises(KeyError, match="is_sensitive"):
        route({k: v for k, v in answers().items() if k != "is_sensitive"})


def test_the_switch_and_the_function_agree():
    assert BIG_SMALL.decide(answers()).name == route(answers()).name
    assert BIG_SMALL.default == BIG and set(BIG_SMALL.cases) == {SMALL, BIG}


def test_the_bank_is_two_questions_and_neither_is_an_exclusive_choice():
    """FINDINGS §11: cost is n_questions x f(state_tokens). §3, §9, §14: choice collapses."""
    assert len(ROUTER_BANK) == 2
    assert {q["type"] for q in ROUTER_BANK.values()} == {"score", "noul"}


def test_the_questions_we_dropped_stayed_out():
    """§14: `domain` is an exclusive choice; `needs_tools` measured below chance (AUC 0.41)."""
    assert "domain" not in ROUTER_BANK and "needs_tools" not in ROUTER_BANK
    assert "trivial" not in ROUTER_BANK   # §12, invented, AUC 0.60


def test_bank_matches_upstream():
    """The bank is copied from `laya.router_questions()` so tests need no torch. It must not drift.

    Skips when `laya` is not installed, which is the normal case for the fast suite.
    """
    laya = pytest.importorskip("laya", reason="laya is the optional `laya` extra")
    upstream = laya.router_questions()
    for qid, ours in ROUTER_BANK.items():
        assert qid in upstream, f"{qid} no longer exists upstream"
        assert ours == upstream[qid], f"{qid} drifted from laya.router_questions()"


def test_nothing_here_dispatches():
    """verdict answers; the caller acts. There is no table of commands to run."""
    import verdict.judge.router as router

    assert not hasattr(router, "CLIS")


# --- the eval set is an input, so it gets checked like one --------------------------------

def test_eval_set_is_well_formed_and_balanced():
    rows = [json.loads(l) for l in EVALS.read_text().splitlines() if l.strip()]
    assert len(rows) == 72
    assert len({r["id"] for r in rows}) == len(rows), "duplicate id"
    assert len({r["prompt"] for r in rows}) == len(rows), "duplicate prompt"
    for r in rows:
        assert r["want"] in (SMALL, BIG)
        assert r["split"] in ("train", "test")
        assert r["prompt"].strip()
    counts = {(r["split"], r["want"]) for r in rows}
    assert counts == {("train", SMALL), ("train", BIG), ("test", SMALL), ("test", BIG)}


def test_eval_split_is_balanced_within_each_class():
    from collections import Counter
    rows = [json.loads(l) for l in EVALS.read_text().splitlines() if l.strip()]
    c = Counter((r["split"], r["want"]) for r in rows)
    assert c[("train", SMALL)] == c[("train", BIG)] == 24
    assert c[("test", SMALL)] == c[("test", BIG)] == 12


def test_shipped_thresholds_are_inside_the_question_ranges():
    assert 0.0 <= T_DIFFICULTY <= 3.0, "difficulty is a 4-level score, 0 to 3"
    assert 0.0 <= T_SENSITIVE <= 1.0, "is_sensitive is a probability"
