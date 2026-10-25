"""`verdict rank`: named answers as a vector, ranked by a weighted sum over rescaled dimensions.

Each question is a named dimension, so a match can be explained. Two things from the support-ticket
trial (FINDINGS §39) are pinned here: raw cosine ranked an all-low ticket among the top five,
because it ignores magnitude, and a squeezed dimension (0.12 to 0.65) lost to a wide one until each
was rescaled to its percentile over the collection. No model loads."""
from __future__ import annotations

import json

import pytest

from verdict import cli


def line(text, **answers):
    out = {}
    for qid, v in answers.items():
        if isinstance(v, dict):
            out[qid] = {"type": "choice", "choice": max(v, key=v.get), "probabilities": v,
                        "confidence": 0.5}
        elif isinstance(v, tuple):
            score, levels = v
            out[qid] = {"type": "score", "score": score, "confidence": 0.5,
                        "legend": {str(i): str(i) for i in range(levels)},
                        "probabilities": {str(i): 1 / levels for i in range(levels)}}
        else:
            out[qid] = {"type": "noul", "noul": v, "confidence": 0.5}
    return {"state": {"message": text}, "model": "fake", "answers": out}


@pytest.fixture
def scored(tmp_path):
    def write(*rows):
        f = tmp_path / "scored.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return str(f)
    return write


def ranked(capsys, *argv):
    assert cli.main(["rank", *argv, "--json"]) == 0
    return [json.loads(l) for l in capsys.readouterr().out.splitlines()]


def test_a_squeezed_dimension_counts_as_much_as_a_wide_one(scored, capsys):
    """Raw, `broken` (0.2 to 0.9) swamps `deadline` (0.2 to 0.4): the item with both loses to one
    that is only very broken. Rescaled, both count equally and the item with both wins."""
    f = scored(line("broken and due tomorrow", broken=0.7, deadline=0.4),
               line("very broken, no date", broken=0.9, deadline=0.2),
               line("fine", broken=0.2, deadline=0.25),
               line("fine, dated", broken=0.25, deadline=0.3))
    top = ranked(capsys, f, "broken=1", "deadline=1")
    assert top[0]["state"]["message"] == "broken and due tomorrow"


def test_uniformly_low_answers_do_not_match_by_direction(scored, capsys):
    """Cosine put 'thanks team!' (about 0.13 everywhere) in the top five for money trouble."""
    f = scored(line("thanks team!", money=0.13, leaving=0.12),
               line("charged twice", money=0.9, leaving=0.2),
               line("switching, too pricey", money=0.55, leaving=0.45),
               line("login loop", money=0.08, leaving=0.1))
    names = [r["state"]["message"] for r in ranked(capsys, f, "money=1", "leaving=0.5", "-k", "2")]
    assert names == ["charged twice", "switching, too pricey"]


def test_each_result_says_what_each_dimension_contributed(scored, capsys):
    f = scored(line("a", money=0.9, leaving=0.1), line("b", money=0.1, leaving=0.9))
    top = ranked(capsys, f, "money=1", "leaving=0.5")[0]
    assert set(top["contributions"]) == {"money", "leaving"}
    assert top["score"] == pytest.approx(sum(top["contributions"].values()))


def test_a_negative_weight_ranks_away_from_a_dimension(scored, capsys):
    f = scored(line("question", how=0.9, broken=0.5), line("bug report", how=0.1, broken=0.5))
    assert ranked(capsys, f, "how=-1")[0]["state"]["message"] == "bug report"


def test_choice_options_and_scores_are_dimensions_too(scored, capsys):
    f = scored(line("refund", intent={"refund": 0.8, "bug": 0.2}, urgency=(2.7, 4)),
               line("bug", intent={"refund": 0.1, "bug": 0.9}, urgency=(0.5, 4)))
    assert ranked(capsys, f, "intent.bug=1")[0]["state"]["message"] == "bug"
    assert ranked(capsys, f, "urgency=1")[0]["state"]["message"] == "refund"


def test_an_unknown_dimension_names_the_ones_there_are(scored, capsys):
    f = scored(line("a", money=0.5))
    assert cli.main(["rank", f, "mony=1"]) == 2
    err = capsys.readouterr().err
    assert "mony" in err and "money" in err


def test_a_malformed_weight_is_refused(scored, capsys):
    f = scored(line("a", money=0.5))
    assert cli.main(["rank", f, "money"]) == 2
    assert "NAME=WEIGHT" in capsys.readouterr().err


def test_plain_output_is_one_line_per_result_with_the_top_contributors(scored, capsys):
    f = scored(line("charged twice", money=0.9, leaving=0.2), line("login loop", money=0.1,
                                                                  leaving=0.1))
    assert cli.main(["rank", f, "money=1", "leaving=0.5", "-k", "1"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and "charged twice" in out[0] and "money" in out[0]


@pytest.mark.parametrize("bad", ["not json", "5", "[1, 2]"])
def test_a_line_that_is_not_an_object_names_the_file_and_line(tmp_path, capsys, bad):
    f = tmp_path / "scored.jsonl"
    f.write_text(json.dumps(line("a", money=0.5)) + f"\n{bad}\n")
    assert cli.main(["rank", str(f), "money=1"]) == 2
    err = capsys.readouterr().err
    assert "scored.jsonl" in err and "line 2" in err and "decide --jsonl" in err


def test_an_empty_file_says_so(tmp_path, capsys):
    f = tmp_path / "scored.jsonl"
    f.write_text("")
    assert cli.main(["rank", str(f), "money=1"]) == 2
    assert "no answers" in capsys.readouterr().err


def test_a_negative_count_is_refused(scored, capsys):
    f = scored(line("a", money=0.5))
    assert cli.main(["rank", f, "money=1", "-k", "-3"]) == 2
    assert "-k" in capsys.readouterr().err
