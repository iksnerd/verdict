"""Audit 4: a flat secondary never replaces the primary, check() orders flat rows as run does,
a flat stage-2 answer gives no label, a missing rank dimension is not a low value, a malformed
fit is a QuestionError."""
from __future__ import annotations

import pytest

from verdict.errors import QuestionError
from verdict.judge import calibrate, cascade, hierarchy, rank

BANK = {"spam": {"type": "noul", "instructions": "Is this spam?"},
        "dept": {"type": "choice", "instructions": "?", "criteria": {"a": "x", "b": "y"}}}


def noul(p):
    return {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}


def choice(pa):
    return {"type": "choice", "choice": "a" if pa >= .5 else "b",
            "probabilities": {"a": pa, "b": 1 - pa}, "confidence": max(pa, 1 - pa)}


def test_a_flat_secondary_answer_keeps_the_primary_and_counts_as_failed():
    primary = lambda s, q: {"answers": {"spam": noul(0.6), "dept": choice(0.9)}}
    secondary = lambda s, q: {"answers": {"spam": noul(0.5), "dept": choice(0.99)}}
    out = cascade.run(["s"], BANK, primary, secondary, share=1.0)
    row = out.rows[0]
    assert row["answers"]["spam"]["tier"] == "primary" and row["answers"]["spam"]["noul"] == 0.6
    assert row["answers"]["dept"]["tier"] == "escalated"
    assert row["escalated"] == ["dept"] and row["escalation_attempted"] == ["dept", "spam"]
    assert "spam" in row["escalation_error"] and "flat" in row["escalation_error"]
    assert len(out.failed) == 1 and out.escalated == 1


def test_check_orders_a_flat_primary_first_like_run():
    labels = [{"spam": False}, {"spam": True}, {"spam": True}]
    first = {"a": noul(0.5), "b": noul(0.46), "c": noul(0.9)}
    second = {"a": noul(0.0), "b": noul(1.0), "c": noul(1.0)}
    primary = lambda s, q: {"answers": {"spam": first[s]}}
    secondary = lambda s, q: {"answers": {"spam": second[s]}}
    bank = {"spam": BANK["spam"]}
    fits = {"spam": {"type": "noul", "cut": 0.45}}
    ran = cascade.run(["a", "b", "c"], bank, primary, secondary, share=1 / 3, fits=fits)
    assert ran.rows[0]["escalated"] == ["spam"]
    rep = cascade.check(["a", "b", "c"], bank, labels, primary, secondary, share=1 / 3, fits=fits)
    assert rep["spam"]["helps"] is True


def test_flat_stage_two_gives_no_label():
    tree = {"groups": {"g1": {"options": {"x": None, "y": None}},
                       "g2": {"options": {"z": None, "w": None}}}}

    def ask(state, questions):
        (qid,) = questions
        crit = questions[qid]["criteria"]
        if qid == "group":
            p = {"g1": 0.7, "g2": 0.3}
        else:
            p = {k: 1 / len(crit) for k in crit}
        return {"answers": {qid: {"probabilities": p}}}

    r = hierarchy.classify(ask, "s", tree, beam=1)
    assert r.label is None and r.candidates == [] and r.probability == 0.0
    assert r.group_probabilities == {"g1": 0.7, "g2": 0.3}


def test_rank_does_not_treat_a_missing_dimension_as_zero():
    rows = [{"state": "a", "answers": {"m": noul(0.9)}},
            {"state": "b", "answers": {"m": noul(0.1)}},
            {"state": "err", "answers": {}}]
    out = rank.rank(rows, {"m": 1.0})
    assert [r["state"] for r in out] == ["a", "b", "err"]
    assert out[2]["unknown"] == ["m"]
    assert out[0]["score"] == 1.0 and out[1]["score"] == 0.0


def test_rank_percentiles_ignore_the_rows_without_the_dimension():
    rows = [{"state": "a", "answers": {"m": noul(0.9)}},
            {"state": "err", "answers": {}},
            {"state": "b", "answers": {"m": noul(0.1)}}]
    out = {r["state"]: r for r in rank.rank(rows, {"m": 1.0})}
    assert out["b"]["score"] == 0.0 and out["a"]["score"] == 1.0


@pytest.mark.parametrize("fit,key", [({"cut": 0.5}, "type"), ({"type": "noul"}, "cut"),
                                     ({"type": "choice"}, "temperature")])
def test_a_malformed_fit_names_the_qid_and_the_missing_key(fit, key):
    ans = {"q": noul(0.7)} if fit.get("type") != "choice" else {"q": choice(0.7)}
    with pytest.raises(QuestionError, match=rf"q.*{key}"):
        calibrate.apply(ans, {"q": fit})
