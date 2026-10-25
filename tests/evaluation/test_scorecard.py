import math

from verdict.label.scorecard import teacher_agreement


def row(i, model, rep, kind_top, noul):
    kind = {"a": 0.1, "b": 0.1}
    kind[kind_top] = 0.8
    return {"id": i, "model": model, "repeat": rep, "labels": {"kind": kind, "flag": noul}}


def rows(ref_kind="a"):
    out = []
    for i in ("s1", "s2"):
        out += [row(i, "cheap", 0, "a", 0.9), row(i, "cheap", 1, "a", 0.8), row(i, "ref", 0, ref_kind, 0.7)]
    return out


def test_full_agreement_scores_100():
    r = teacher_agreement(rows("a"), cheap=["cheap"], reference="ref")
    assert r["score"] == 100.0 and r["n_states"] == 2
    assert r["self_agreement"] == 100.0


def test_disagreement_on_one_question_halves_that_question():
    r = teacher_agreement(rows("b"), cheap=["cheap"], reference="ref")
    assert r["per_question"]["kind"] == 0.0 and r["per_question"]["flag"] == 100.0
    assert math.isclose(r["score"], 50.0)


def test_states_missing_the_reference_are_skipped_not_counted_as_disagreement():
    rs = rows("a") + [row("s3", "cheap", 0, "a", 0.9), row("s3", "cheap", 1, "a", 0.9)]
    assert teacher_agreement(rs, cheap=["cheap"], reference="ref")["n_states"] == 2
