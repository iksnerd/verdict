"""Audit 2: a partly flat router answer is no model; a mismatched fit is an error, never silently
ignored; a flat primary never becomes a decision; a secondary that omits a question fails that
state only; a flat group distribution yields no label."""
from __future__ import annotations

import pytest

from verdict.errors import QuestionError
from verdict.judge import calibrate, cascade, hierarchy
from verdict.judge.router import BIG, is_uniform, route

from fakes import router_answers

BANK = {"spam": {"type": "noul", "instructions": "Is this spam?"}}


def test_partly_flat_router_answers_route_big():
    flat = {"0": .25, "1": .25, "2": .25, "3": .25}
    a = router_answers(flat, 1.5, 0.05)
    assert is_uniform(a)
    assert route(a).name == BIG


def test_flat_sensitive_alone_is_also_no_model():
    sharp = {"0": .9, "1": .05, "2": .03, "3": .02}
    assert is_uniform(router_answers(sharp, 0.2, 0.5))


def test_fit_type_mismatch_raises_in_apply():
    answers = {"spam": {"type": "noul", "noul": 0.7, "confidence": 0.7}}
    fits = {"spam": {"type": "choice", "temperature": 1.0}}
    with pytest.raises(QuestionError, match="spam.*choice.*noul|spam.*noul.*choice"):
        calibrate.apply(answers, fits)


def test_fit_type_mismatch_raises_in_cascade_cuts():
    with pytest.raises(QuestionError, match="spam"):
        cascade._cuts(BANK, {"spam": {"type": "choice", "temperature": 1.0}})


def test_apply_gives_no_decision_on_a_flat_noul():
    out = calibrate.apply({"spam": {"type": "noul", "noul": 0.5, "confidence": 0.5}},
                          {"spam": {"type": "noul", "cut": 0.2}})
    assert out["spam"]["decision"] is None


def test_cascade_escalates_a_flat_primary_and_never_decides_it():
    seen = []

    def primary(state, questions):
        return {"answers": {"spam": {"type": "noul", "noul": 0.5, "confidence": 0.5}}}

    def secondary(state, questions):
        seen.append(state)
        return {"answers": {"spam": {"type": "noul", "noul": 0.9, "confidence": 0.9}}}

    # cut 0.004 makes 0.5 look far from the cut; it must still be treated as least sure
    run = cascade.run(["a"], BANK, primary, secondary, share=1.0,
                      fits={"spam": {"type": "noul", "cut": 0.004}})
    assert run.rows[0]["answers"]["spam"]["tier"] == "escalated"

    def keep(state, questions):
        return {"answers": {"spam": {"type": "noul", "noul": 0.5, "confidence": 0.5}}}

    run = cascade.run(["a", "b"], BANK, primary, keep, share=0.5,
                      fits={"spam": {"type": "noul", "cut": 0.004}})
    # one of the two is escalated; whichever is primary-tier must not carry a boolean decision
    for row in run.rows:
        a = row["answers"]["spam"]
        assert a.get("decision") is None or a["tier"] == "escalated"


def test_cascade_flat_primary_selected_before_a_confident_one():
    table = {"a": 0.5, "b": 0.001}

    def primary(state, questions):
        p = table[state]
        return {"answers": {"spam": {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}}}

    asked = []

    def secondary(state, questions):
        asked.append(state)
        return {"answers": {"spam": {"type": "noul", "noul": 0.9, "confidence": 0.9}}}

    cascade.run(["a", "b"], BANK, primary, secondary, share=0.5,
                fits={"spam": {"type": "noul", "cut": 0.004}})
    assert asked == ["a"]


def test_secondary_omitting_a_question_fails_that_state_only():
    bank = {**BANK, "dept": {"type": "choice", "instructions": "?", "criteria": {"a": "", "b": ""}}}

    def primary(state, questions):
        return {"answers": {
            "spam": {"type": "noul", "noul": 0.55, "confidence": 0.55},
            "dept": {"type": "choice", "choice": "a", "probabilities": {"a": .5, "b": .5}, "confidence": 0.0}}}

    calls = []

    def secondary(state, questions):
        calls.append(state)
        if state == "bad":
            return {"answers": {}}
        return {"answers": {q: primary(state, bank)["answers"][q] for q in questions}}

    run = cascade.run(["ok", "bad"], bank, primary, secondary, share=1.0)
    assert len(run.rows) == 2
    assert run.rows[0]["escalated"] and "escalation_error" not in run.rows[0]
    bad = run.rows[1]
    assert bad["escalated"] == [] and bad["escalation_attempted"] and "escalation_error" in bad
    assert run.failed and run.failed[0][0] == 1
    assert bad["answers"]["spam"]["tier"] == "primary"


TREE = {"instructions": "?", "groups": {
    "g1": {"description": "", "options": {"a": None, "b": None}},
    "g2": {"description": "", "options": {"c": None, "d": None}}}}


def test_flat_group_distribution_gives_no_label():
    def ask(state, questions):
        qid = next(iter(questions))
        keys = list(questions[qid]["criteria"])
        return {"answers": {qid: {"probabilities": {k: 1 / len(keys) for k in keys}}}}

    result = hierarchy.classify(ask, "x", TREE)
    assert result.label is None and result.candidates == []
    assert result.as_dict()["label"] is None
    assert result.passes == 1


def test_a_single_group_is_not_called_flat():
    tree = {"instructions": "?", "groups": {"g1": {"description": "", "options": {"a": None}}}}

    def ask(state, questions):
        return {"answers": {"group": {"probabilities": {"g1": 1.0}}}}

    assert hierarchy.classify(ask, "x", tree).label == "a"
