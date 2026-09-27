"""`verdict ask`: one question inline, no JSON file. The server is faked, so no model loads."""
from __future__ import annotations

import pytest

from verdict import cli, client
from conftest import typed


def fake_server(monkeypatch, answer):
    sent = {}

    def fake_decide(state, questions, url=None, model=None, **flags):
        sent["state"], sent["questions"] = state, questions
        return {"model": "fake", "answers": typed(questions, lambda k: answer)}

    monkeypatch.setattr(client, "decide", fake_decide)
    return sent


NOUL = {"type": "noul", "noul": 0.72, "confidence": 0.72}
#: laya's `confidence` for a choice is 1 - H(p)/log(k), not the top probability: 0.25 here.
CHOICE = {"type": "choice", "choice": "refund", "confidence": 0.25,
          "probabilities": {"refund": 0.61, "info": 0.3, "other": 0.09}}


def test_yes_no_question_becomes_a_noul(monkeypatch):
    sent = fake_server(monkeypatch, NOUL)
    cli.main(["ask", "git push --force", "Is this destructive?", "--yesno"])
    assert sent["questions"] == {"q": {"type": "noul", "instructions": "Is this destructive?"}}
    assert sent["state"] == "git push --force"


def test_options_become_a_choice_and_name_equals_description(monkeypatch):
    sent = fake_server(monkeypatch, CHOICE)
    cli.main(["ask", "charged twice", "What do they want?",
              "-o", "refund=money back", "-o", "info", "-o", "other"])
    assert sent["questions"]["q"] == {
        "type": "choice", "instructions": "What do they want?",
        "criteria": {"refund": "money back", "info": "info", "other": "other"},
    }


def test_one_option_is_an_error(monkeypatch, capsys):
    fake_server(monkeypatch, CHOICE)
    assert cli.main(["ask", "x", "which?", "-o", "only"]) == 2


def test_noul_prints_only_the_score_without_a_cut(monkeypatch, capsys):
    """No cut, no yes/no: the model is uncalibrated and 0.5 is not a threshold anyone fitted."""
    fake_server(monkeypatch, NOUL)
    assert cli.main(["ask", "x", "q?"]) == 0
    assert capsys.readouterr().out.strip() == "0.72"


@pytest.mark.parametrize("cut,word,code", [(0.5, "yes", 0), (0.8, "no", 1)])
def test_cut_decides_and_sets_exit_status(monkeypatch, capsys, cut, word, code):
    fake_server(monkeypatch, NOUL)
    assert cli.main(["ask", "x", "q?", "--cut", str(cut)]) == code
    assert capsys.readouterr().out.strip() == f"{word} 0.72"


def test_choice_prints_the_top_option_and_its_probability(monkeypatch, capsys):
    """The probability, not laya's entropy confidence: a 0.50/0.50 split printed "A 0.00", and a
    calibrated run printed a probability in the same slot, so one column held two quantities."""
    fake_server(monkeypatch, CHOICE)
    assert cli.main(["ask", "x", "q?", "-o", "refund", "-o", "info", "-o", "other"]) == 0
    assert capsys.readouterr().out.strip() == "refund 0.61"


def test_calibrated_choice_prints_the_calibrated_probability(monkeypatch, capsys, tmp_path):
    fake_server(monkeypatch, CHOICE)
    fit = tmp_path / "fit.json"
    fit.write_text('{"q": {"type": "choice", "temperature": 0.5}}')
    assert cli.main(["ask", "x", "q?", "-o", "refund", "-o", "info", "-o", "other",
                     "--cut", str(fit)]) == 0
    # 0.61^2 / (0.61^2 + 0.3^2 + 0.09^2) = 0.7922
    assert capsys.readouterr().out.strip() == "refund 0.79"


def test_dash_reads_the_text_from_stdin(monkeypatch):
    sent = fake_server(monkeypatch, NOUL)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO("a diff\n"))
    cli.main(["ask", "-", "q?"])
    assert sent["state"] == "a diff"


def test_json_prints_the_full_answer(monkeypatch, capsys):
    fake_server(monkeypatch, NOUL)
    cli.main(["ask", "x", "q?", "--json"])
    assert '"noul": 0.72' in capsys.readouterr().out
