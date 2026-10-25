"""`verdict classify`: a text into one leaf of a two-level tree, for banks past laya's 20 options.

A fake server answers each choice question with a fixed favourite, so the command's wiring (the
tree file, the flags, the output, the exit codes) is tested without a model."""
from __future__ import annotations

import json

import pytest

from verdict import cli, client

TREE = {
    "instructions": "What is the customer asking about?",
    "groups": {
        "card": {"description": "a payment card", "options": {"card lost": None, "card late": None}},
        "transfer": {"description": "moving money",
                     "options": {"transfer failed": None, "transfer fee": None}},
    },
}


@pytest.fixture
def served(monkeypatch, tmp_path):
    """Favourites: group transfer, option transfer failed. Records every question."""
    asked = []

    def decide(state, questions, url=None, model=None, **flags):
        (qid, q), = questions.items()
        asked.append(qid)
        keys = list(q["criteria"])
        fav = "transfer" if qid == "group" else "transfer failed"
        p = {k: (0.8 if k == fav else 0.2 / (len(keys) - 1)) for k in keys}
        top = fav if fav in p else max(p, key=p.get)
        return {"model": "m", "answers": {qid: {"type": "choice", "choice": top,
                                                "probabilities": p, "confidence": 0.5}}}

    monkeypatch.setattr(client, "decide", decide)
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(TREE))
    return asked, str(path)


def run(argv, capsys):
    code = cli.main(["classify", *argv])
    return code, capsys.readouterr()


def test_it_prints_the_leaf_its_probability_and_the_path(served, capsys):
    asked, tree = served
    code, out = run(["my transfer never arrived", "--tree", tree, "--server-only"], capsys)
    assert code == 0
    assert out.out.startswith("transfer failed 0.64") and "transfer > transfer failed" in out.out
    assert asked[0] == "group" and asked.count("option") == 2   # default beam 2


def test_beam_one_asks_one_option_level(served, capsys):
    asked, tree = served
    run(["x", "--tree", tree, "--beam", "1", "--server-only"], capsys)
    assert asked == ["group", "option"]


def test_json_carries_the_candidates_and_the_pass_count(served, capsys):
    _, tree = served
    code, out = run(["x", "--tree", tree, "--json", "--server-only"], capsys)
    doc = json.loads(out.out)
    assert code == 0 and doc["label"] == "transfer failed" and doc["passes"] == 3
    assert doc["candidates"][0]["path"] == ["transfer", "transfer failed"]
    assert set(doc["group_probabilities"]) == {"card", "transfer"}
    assert len(doc["searched_groups"]) == 2


def test_a_missing_tree_file_is_a_not_found_error(served, capsys):
    code, out = run(["x", "--tree", "/no/such/tree.json", "--server-only"], capsys)
    assert code == 2 and "verdict:" in out.err


def test_a_tree_with_a_level_past_twenty_is_refused_and_names_the_group(served, tmp_path, capsys):
    wide = {"instructions": "?", "groups": {"big": {"options": {f"o{i}": None for i in range(21)}}}}
    path = tmp_path / "wide.json"
    path.write_text(json.dumps(wide))
    code, out = run(["x", "--tree", str(path), "--server-only"], capsys)
    assert code == 2 and "'big' has 21 options" in out.err


def test_a_bad_beam_is_a_usage_error(served, capsys):
    _, tree = served
    code, out = run(["x", "--tree", tree, "--beam", "0", "--server-only"], capsys)
    assert code == 2 and "beam" in out.err


def test_a_question_wording_that_measured_at_chance_is_still_refused(served, tmp_path, capsys):
    bad = {**TREE, "instructions": "How dangerous is this message?"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    code, out = run(["x", "--tree", str(path), "--server-only"], capsys)
    assert code == 2 and "dangerous" in out.err


def test_the_command_has_help_with_an_example_and_its_exit_status(capsys):
    with pytest.raises(SystemExit):
        cli.main(["classify", "--help"])
    out = capsys.readouterr().out
    assert "Example" in out and "Exit status" in out and "20" in out
