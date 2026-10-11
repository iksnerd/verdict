"""Hierarchical classification: a coarse choice among groups, then a fine one inside the best few.

Laya's options share one token budget, so past about 20 accuracy collapses (Banking77, 77 options:
0.43, FINDINGS §41). Two levels of at most 20 reach 400 labels with one pass for the groups and one
or two for the options. The seam is `classify(ask, state, tree)`, where `ask(state, questions)`
returns `{"answers": {...}}` as every front door does; a fake stands in for the model.
"""
from __future__ import annotations

import pytest

from verdict import hierarchy
from verdict.errors import QuestionError

TREE = {
    "instructions": "What is the customer asking about?",
    "groups": {
        "card": {"description": "problems with a payment card",
                 "options": {"card_lost": "the card is lost or stolen", "card_late": None}},
        "transfer": {"description": "moving money",
                     "options": {"transfer_failed": "a transfer did not arrive",
                                 "transfer_fee": "a fee on a transfer"}},
        "account": {"description": "the account itself", "options": {"close_account": None}},
    },
}


def fake(group_p, option_p):
    """A model: `group_p` for the group question, `option_p[group]` for each group's options."""
    seen = []

    def ask(state, questions):
        (qid, q), = questions.items()
        seen.append((qid, q))
        if qid == "group":
            p = group_p
        else:
            keys = tuple(q["criteria"])
            p = next(v for k, v in option_p.items() if tuple(k) == keys)
        top = max(p, key=p.get)
        return {"answers": {qid: {"type": "choice", "choice": top, "probabilities": dict(p),
                                  "confidence": 0.5}}}

    ask.seen = seen
    return ask


OPTION_P = {
    ("card_lost", "card_late"): {"card_lost": 0.5, "card_late": 0.5},
    ("transfer_failed", "transfer_fee"): {"transfer_failed": 0.95, "transfer_fee": 0.05},
}


def test_the_best_group_then_the_best_option_inside_it():
    ask = fake({"card": 0.2, "transfer": 0.7, "account": 0.1}, OPTION_P)
    r = hierarchy.classify(ask, "my transfer never arrived", TREE, beam=1)
    assert r.label == "transfer_failed" and r.path == ["transfer", "transfer_failed"]
    assert r.probability == pytest.approx(0.7 * 0.95)
    assert r.passes == 2


def test_a_wider_beam_recovers_when_the_top_group_is_wrong_but_the_best_leaf_is_elsewhere():
    group = {"card": 0.50, "transfer": 0.45, "account": 0.05}
    ask = fake(group, OPTION_P)
    narrow = hierarchy.classify(ask, "x", TREE, beam=1)
    wide = hierarchy.classify(fake(group, OPTION_P), "x", TREE, beam=2)
    assert narrow.label in {"card_lost", "card_late"} and narrow.probability == pytest.approx(0.25)
    assert wide.label == "transfer_failed" and wide.probability == pytest.approx(0.45 * 0.95)
    assert wide.passes == 3


def test_a_single_option_group_costs_no_second_pass():
    ask = fake({"card": 0.1, "transfer": 0.1, "account": 0.8}, OPTION_P)
    r = hierarchy.classify(ask, "x", TREE, beam=1)
    assert r.label == "close_account" and r.probability == pytest.approx(0.8) and r.passes == 1


def test_candidates_are_sorted_and_name_their_path():
    r = hierarchy.classify(fake({"card": 0.5, "transfer": 0.45, "account": 0.05}, OPTION_P),
                           "x", TREE, beam=2)
    probs = [c["probability"] for c in r.candidates]
    assert probs == sorted(probs, reverse=True) and r.candidates[0]["label"] == r.label
    assert r.candidates[0]["path"] == ["transfer", "transfer_failed"]


def test_both_levels_get_the_instructions_and_the_descriptions():
    ask = fake({"card": 0.2, "transfer": 0.7, "account": 0.1}, OPTION_P)
    hierarchy.classify(ask, "x", TREE, beam=1)
    (_, group_q), (_, option_q) = ask.seen
    assert group_q["instructions"] == option_q["instructions"] == TREE["instructions"]
    assert group_q["criteria"]["card"] == "problems with a payment card"
    assert option_q["criteria"]["transfer_failed"] == "a transfer did not arrive"


def test_an_undescribed_option_is_sent_with_an_empty_description():
    ask = fake({"card": 0.9, "transfer": 0.05, "account": 0.05}, OPTION_P)
    hierarchy.classify(ask, "x", TREE, beam=1)
    assert ask.seen[1][1]["criteria"]["card_late"] == ""


def tree(groups, **kw):
    return {"instructions": "?", "groups": groups, **kw}


def test_a_level_past_twenty_is_refused_by_name_with_the_remedy():
    many = {f"g{i}": {"options": {f"o{i}": None}} for i in range(21)}
    with pytest.raises(QuestionError, match="21 groups"):
        hierarchy.validate(tree(many))
    wide = {"g": {"options": {f"o{i}": None for i in range(21)}}}
    with pytest.raises(QuestionError, match="'g' has 21 options"):
        hierarchy.validate(tree(wide))


def test_a_leaf_label_must_be_unique_across_groups():
    with pytest.raises(QuestionError, match="'dup'"):
        hierarchy.validate(tree({"a": {"options": {"dup": None, "x": None}},
                                 "b": {"options": {"dup": None}}}))


@pytest.mark.parametrize("bad,match", [
    ({}, "groups"),
    ({"groups": {}}, "groups"),
    ({"groups": {"a": {"options": {}}}}, "'a'"),
    ({"groups": {"a": {}}}, "'a'"),
    ({"groups": ["a"]}, "groups"),
])
def test_a_malformed_tree_is_named(bad, match):
    with pytest.raises(QuestionError, match=match):
        hierarchy.validate({"instructions": "?", **bad})


def test_a_beam_below_one_is_refused():
    with pytest.raises(QuestionError, match="beam"):
        hierarchy.classify(fake({}, {}), "x", TREE, beam=0)


def test_a_tree_with_one_group_is_valid_and_costs_one_pass_per_option_level():
    one = tree({"only": {"options": {"a": None, "b": None}}})
    ask = fake({"only": 1.0}, {("a", "b"): {"a": 0.3, "b": 0.7}})
    r = hierarchy.classify(ask, "x", one, beam=2)
    assert r.label == "b" and r.passes == 2
