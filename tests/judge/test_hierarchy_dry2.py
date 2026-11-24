"""Both stages of a tree decide "no model behind this answer" the same way, an empty one included."""
from __future__ import annotations

from verdict.judge import hierarchy

TREE = {"groups": {"a": {"options": {"x": None, "y": None}}, "b": {"options": {"z": None, "w": None}}}}


def asker(group_p, option_p):
    def ask(state, questions):
        (qid,) = questions
        return {"answers": {qid: {"type": "choice", "probabilities": group_p if qid == "group" else option_p}}}
    return ask


def test_an_empty_second_stage_answer_is_no_label_not_a_zero_probability_one():
    result = hierarchy.classify(asker({"a": 0.7, "b": 0.3}, {}), "s", TREE, beam=1)
    assert result.label is None and result.candidates == []


def test_an_empty_first_stage_answer_is_still_flat():
    result = hierarchy.classify(asker({}, {"x": 0.9, "y": 0.1}), "s", TREE)
    assert result.label is None and result.passes == 1


def test_a_flat_answer_is_flat_in_both_stages():
    flat = {"a": 0.5, "b": 0.5}
    assert hierarchy.classify(asker(flat, {"x": 0.9, "y": 0.1}), "s", TREE).label is None
    assert hierarchy.classify(asker({"a": 0.7, "b": 0.3}, {"x": 0.5, "y": 0.5}), "s", TREE,
                              beam=1).label is None
