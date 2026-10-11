"""Laya first, a second backend for the items Laya is least sure of (FINDINGS §52).

The seam is `cascade.run(states, bank, primary, secondary, ...)` with both backends injected as
`ask(state, questions) -> {"answers": {...}}`, so no model and no network are involved. The things
that matter are who gets asked what: the secondary must see only the rows and questions that were
escalated, and nothing else may leave."""
from __future__ import annotations

import pytest

from verdict import cascade
from verdict.errors import QuestionError

BANK = {
    "spam": {"type": "noul", "instructions": "Is this spam?"},
    "dept": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "", "tech": ""}},
}


def answers(spam, dept_billing):
    return {"spam": {"type": "noul", "noul": spam, "confidence": max(spam, 1 - spam)},
            "dept": {"type": "choice", "choice": "billing" if dept_billing >= .5 else "tech",
                     "probabilities": {"billing": dept_billing, "tech": 1 - dept_billing}, "confidence": 0.5}}


#: (spam P(yes), dept P(billing)): row 0 sure, row 1 spam unsure, row 2 dept unsure, row 3 both unsure
ROWS = [(0.99, 0.95), (0.55, 0.9), (0.97, 0.52), (0.45, 0.48)]


def primary(table=ROWS):
    def ask(state, questions):
        return {"model": "laya", "answers": {q: answers(*table[state["i"]])[q] for q in questions}}
    return ask


class Secondary:
    """Records exactly what it was asked."""
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, state, questions):
        self.calls.append((state["i"], sorted(questions)))
        if state["i"] == self.fail_on:
            raise ValueError("escalation backend down")
        out = {}
        for q in questions:
            out[q] = ({"type": "noul", "noul": 0.01} if q == "spam" else
                      {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.1, "tech": 0.9}})
        return {"model": "jev", "answers": out}


STATES = [{"i": i} for i in range(len(ROWS))]


def test_confidence_is_the_probability_of_the_answer_given():
    assert cascade.confidence({"type": "noul", "noul": 0.9}) == pytest.approx(0.9)
    assert cascade.confidence({"type": "noul", "noul": 0.1}) == pytest.approx(0.9)
    assert cascade.confidence({"type": "choice", "probabilities": {"a": 0.6, "b": 0.4}}) == 0.6
    assert cascade.confidence({"type": "score", "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}}) == 0.5


def test_a_fixed_threshold_escalates_only_the_answers_below_it():
    sec = Secondary()
    out = cascade.run(STATES, BANK, primary(), sec, below=0.6, errors=(ValueError,))
    assert sec.calls == [(1, ["spam"]), (2, ["dept"]), (3, ["dept", "spam"])]
    assert out.rows[0]["escalated"] == [] and out.rows[3]["escalated"] == ["dept", "spam"]


def test_a_share_escalates_the_least_sure_per_question():
    sec = Secondary()
    out = cascade.run(STATES, BANK, primary(), sec, share=0.5, errors=(ValueError,))
    asked = {(i, q) for i, qs in sec.calls for q in qs}
    # per question, the two least-sure of four rows
    assert asked == {(3, "spam"), (1, "spam"), (3, "dept"), (2, "dept")}
    assert out.escalated == 4 and out.answers == 8


def test_a_share_of_zero_sends_nothing_and_of_one_sends_everything():
    none = Secondary()
    cascade.run(STATES, BANK, primary(), none, share=0.0, errors=(ValueError,))
    assert none.calls == []
    every = Secondary()
    cascade.run(STATES, BANK, primary(), every, share=1.0, errors=(ValueError,))
    assert len(every.calls) == 4 and all(qs == ["dept", "spam"] for _, qs in every.calls)


def test_each_answer_says_which_tier_gave_it():
    out = cascade.run(STATES, BANK, primary(), Secondary(), below=0.6, errors=(ValueError,))
    row = out.rows[1]
    assert row["answers"]["spam"]["tier"] == "escalated" and row["answers"]["spam"]["noul"] == 0.01
    assert row["answers"]["dept"]["tier"] == "primary" and row["answers"]["dept"]["choice"] == "billing"


def test_a_failed_escalation_keeps_the_primary_answer_and_is_reported():
    out = cascade.run(STATES, BANK, primary(), Secondary(fail_on=1), below=0.6, errors=(ValueError,))
    assert out.rows[1]["answers"]["spam"]["tier"] == "primary"
    assert out.rows[1]["escalation_error"] == "escalation backend down"
    assert [i for i, _ in out.failed] == [1]
    assert out.rows[2]["answers"]["dept"]["tier"] == "escalated"   # the rest carried on


def test_a_state_that_was_not_escalated_is_never_shown_to_the_secondary():
    sec = Secondary()
    cascade.run(STATES, BANK, primary(), sec, below=0.6, errors=(ValueError,))
    assert 0 not in {i for i, _ in sec.calls}


def test_exactly_one_of_share_and_below_is_required():
    with pytest.raises(QuestionError, match="share.*below|below.*share"):
        cascade.run(STATES, BANK, primary(), Secondary())
    with pytest.raises(QuestionError, match="share.*below|below.*share"):
        cascade.run(STATES, BANK, primary(), Secondary(), share=0.2, below=0.6)
    with pytest.raises(QuestionError, match="share"):
        cascade.run(STATES, BANK, primary(), Secondary(), share=1.5)


# ---- the per-task guard: do the two backends, on YOUR labelled examples, justify escalating? ----

LABELS = [{"spam": True, "dept": "billing"}, {"spam": True, "dept": "billing"},
          {"spam": False, "dept": "billing"}, {"spam": False, "dept": "tech"}]


def test_check_reports_primary_cascade_and_secondary_accuracy_per_question():
    report = cascade.check(STATES, BANK, LABELS, primary(), Secondary(), share=0.5, errors=(ValueError,))
    spam, dept = report["spam"], report["dept"]
    assert spam["n"] == 4 and dept["n"] == 4
    # secondary answers spam=0.01 (no) and dept=tech for every row
    assert spam["secondary"] == pytest.approx(0.5) and dept["secondary"] == pytest.approx(0.25)
    assert spam["primary"] == pytest.approx(0.75) and dept["primary"] == pytest.approx(1.0)
    assert 0 <= spam["cascade"] <= 1


def test_check_says_when_escalating_would_not_help():
    report = cascade.check(STATES, BANK, LABELS, primary(), Secondary(), share=0.5, errors=(ValueError,))
    assert report["dept"]["helps"] is False      # the secondary is worse here (AG News in §52)
    assert report["spam"]["helps"] is False      # a tie is not a reason to send text out


def test_check_asks_the_secondary_about_every_labelled_row_once():
    sec = Secondary()
    cascade.check(STATES, BANK, LABELS, primary(), sec, share=0.5, errors=(ValueError,))
    assert len(sec.calls) == 4 and all(qs == ["dept", "spam"] for _, qs in sec.calls)


def test_a_share_sends_exactly_that_many_even_when_confidences_tie():
    from verdict import cascade as c
    assert c.select([0.5, 0.5, 0.5, 0.9], share=0.5) == {0, 1}
    assert c.select([0.9, 0.9, 0.9, 0.9], share=0.25) == {0}
    assert c.select([0.3, 0.9], share=0.0) == set()


# ---- a fitted cut as the uncertainty signal (FINDINGS §55) ---------------------------------------

import math  # noqa: E402


def noul_row(p):
    return {"model": "laya", "answers": {"spam": {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}}}


def spam_primary(table):
    return lambda state, qs: noul_row(table[state["i"]])


class SpamSecond:
    def __init__(self):
        self.calls = []

    def __call__(self, state, qs):
        self.calls.append(state["i"])
        return {"model": "jev", "answers": {"spam": {"type": "noul", "noul": 0.01}}}


SPAM_BANK = {"spam": {"type": "noul", "instructions": "Is this spam?"}}


def test_with_a_cut_confidence_is_the_distance_from_the_cut_not_from_one_half():
    near = cascade.confidence({"type": "noul", "noul": 0.80}, cut=0.80)
    far = cascade.confidence({"type": "noul", "noul": 0.99}, cut=0.80)
    assert near == pytest.approx(0.0, abs=1e-9) and far > 2
    assert cascade.confidence({"type": "noul", "noul": 0.9}) == pytest.approx(0.9)   # unchanged without one


def test_a_row_sitting_on_the_cut_is_escalated_first_even_when_it_looks_confident():
    table = [0.99, 0.80, 0.30]       # raw confidence .99 / .80 / .70; distance from a cut of .8: 3.2 / 0 / 2.2
    fits = {"spam": {"type": "noul", "cut": 0.80}}
    sec = SpamSecond()
    cascade.run([{"i": i} for i in range(3)], SPAM_BANK, spam_primary(table), sec, share=1 / 3, fits=fits)
    assert sec.calls == [1]
    sec2 = SpamSecond()
    cascade.run([{"i": i} for i in range(3)], SPAM_BANK, spam_primary(table), sec2, share=1 / 3)
    assert sec2.calls == [2], "without the fit the least-sure row is the one nearest 0.5"


def test_primary_answers_carry_the_decision_at_the_cut():
    fits = {"spam": {"type": "noul", "cut": 0.80}}
    out = cascade.run([{"i": 0}, {"i": 1}], SPAM_BANK, spam_primary([0.85, 0.70]), SpamSecond(),
                      share=0.0, fits=fits)
    a0, a1 = (r["answers"]["spam"] for r in out.rows)
    assert a0["decision"] is True and a1["decision"] is False and a0["cut"] == 0.80


def test_a_fixed_confidence_cannot_be_combined_with_a_fitted_cut():
    fits = {"spam": {"type": "noul", "cut": 0.80}}
    with pytest.raises(QuestionError, match="share"):
        cascade.run([{"i": 0}], SPAM_BANK, spam_primary([0.5]), SpamSecond(), below=0.6, fits=fits)


def test_check_with_a_holdout_fits_on_part_and_reports_on_the_rest():
    """Scores sit well under 0.5 for both classes, so the default cut calls everything no."""
    n = 20
    table = [0.40 + 0.01 * (i % 3) if i % 2 == 0 else 0.05 + 0.01 * (i % 3) for i in range(n)]
    labels = [{"spam": i % 2 == 0} for i in range(n)]
    report = cascade.check([{"i": i} for i in range(n)], SPAM_BANK, labels, spam_primary(table),
                           SpamSecond(), share=0.5, holdout=0.3, errors=(ValueError,))
    r = report["spam"]
    assert r["n"] == 6 and 0.05 < r["cut"] < 0.40
    assert r["primary_raw"] < r["primary"] == pytest.approx(1.0), "the fitted cut beats the default 0.5"
    assert r["fit"]["type"] == "noul" and r["fit"]["cut"] == r["cut"]


def test_check_without_a_holdout_still_measures_every_row_at_the_given_fit():
    n = 8
    table = [0.9, 0.1] * 4
    report = cascade.check([{"i": i} for i in range(n)], SPAM_BANK, [{"spam": i % 2 == 0} for i in range(n)],
                           spam_primary(table), SpamSecond(), share=0.5, errors=(ValueError,))
    assert report["spam"]["n"] == 8 and report["spam"]["cut"] is None
    assert report["spam"]["primary"] == report["spam"]["primary_raw"]


def test_with_a_fit_an_escalated_yes_no_still_carries_a_decision_and_says_at_what_cut():
    fits = {"spam": {"type": "noul", "cut": 0.80}}
    out = cascade.run([{"i": 0}, {"i": 1}], SPAM_BANK, spam_primary([0.99, 0.80]), SpamSecond(),
                      share=0.5, fits=fits)
    kept, sent = (r["answers"]["spam"] for r in out.rows)
    assert kept["tier"] == "primary" and kept["decision"] is True and kept["cut"] == 0.80
    assert sent["tier"] == "escalated" and sent["decision"] is False and sent["cut"] == 0.5
