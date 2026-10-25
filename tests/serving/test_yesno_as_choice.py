"""An unmeasured yes/no is asked as a choice between `no` and `yes`, and answered as a yes/no.

A plain yes/no can follow its own `false`/`true` labels instead of the state: SST-2's "Is `text`
positive?" ranked at 0.79 with no positive over 0.5. The same words asked as a no/yes choice rank
at 0.96 and leave healthy questions unchanged (FINDINGS §38). The measured library's yes/no
questions are kept as they were measured. No model loads here."""
from __future__ import annotations

import json

import pytest

from verdict import cli, client
from verdict.cli import inference
from fakes import noul

NEW = {"q": {"type": "noul", "instructions": "Does `text` mention a deadline?"}}


@pytest.fixture
def server(monkeypatch):
    sent = []

    def fake_decide(state, questions, url=None, model=None, **kw):
        sent.append(questions)
        answers = {}
        for k, q in questions.items():
            if q["type"] == "choice":
                answers[k] = {"type": "choice", "choice": "yes", "confidence": 0.4,
                              "probabilities": {"no": 0.2, "yes": 0.8}}
            else:
                answers[k] = noul(0.3, 0.7)
        return {"model": "fake", "answers": answers}

    monkeypatch.setattr(client, "decide", fake_decide)
    monkeypatch.setattr(inference, "load", lambda *a, **k: pytest.fail("loaded a model"))
    return sent


def run(capsys, *argv):
    assert cli.main(["decide", '{"text": "due friday", "latest": "shipped"}', *argv]) == 0
    return json.loads(capsys.readouterr().out)["answers"]


def test_a_new_yes_no_is_sent_as_a_no_yes_choice(server, capsys):
    run(capsys, "-q", json.dumps(NEW))
    assert server[0]["q"]["type"] == "choice"
    assert list(server[0]["q"]["criteria"]) == ["no", "yes"]
    assert server[0]["q"]["instructions"] == NEW["q"]["instructions"]


def test_the_answer_comes_back_as_a_yes_no(server, capsys):
    """`answers.<qid>.noul` is what the triage script and calibration read."""
    a = run(capsys, "-q", json.dumps(NEW))["q"]
    assert a["type"] == "noul" and a["noul"] == 0.8 and a["asked_as"] == "choice"


def test_described_sides_become_the_options_descriptions(server, capsys):
    q = {"q": {**NEW["q"], "criteria": {"true": "a date is given", "false": "no date"}}}
    run(capsys, "-q", json.dumps(q))
    assert server[0]["q"]["criteria"] == {"no": "no date", "yes": "a date is given"}


def test_a_measured_library_yes_no_is_asked_as_measured(server, capsys, tmp_path):
    """On the checkpoint it was measured on. A directory of that name stands in for it, so the
    test does not depend on this machine having the weights (pre-commit and CI do not)."""
    finetune = tmp_path / "verdict-v1-mlx"
    finetune.mkdir()
    a = run(capsys, "-q", "work_finished", "--model", str(finetune))["work_finished"]
    assert server[0]["work_finished"]["type"] == "noul"
    assert a["noul"] == 0.3 and "asked_as" not in a


def test_yesno_flag_keeps_the_plain_question(server, capsys):
    run(capsys, "-q", json.dumps(NEW), "--yesno")
    assert server[0]["q"]["type"] == "noul"


def test_ask_rewrites_its_default_yes_no_too(server, capsys):
    assert cli.main(["ask", '{"text": "due friday"}', "Does `text` mention a deadline?",
                     "--json"]) == 0
    assert server[0]["q"]["type"] == "choice"
    assert json.loads(capsys.readouterr().out)["answers"]["q"]["noul"] == 0.8


def test_confidence_means_what_it_means_for_a_yes_no(server, capsys):
    """laya's yes/no confidence is max(p, 1 - p); a choice's is 1 - normalized entropy. Passing
    the choice's through turned a 0.43 answer's 0.57 into 0.0128."""
    a = run(capsys, "-q", json.dumps(NEW))["q"]
    assert a["confidence"] == pytest.approx(0.8)


def test_on_base_laya_a_library_yes_no_is_asked_as_a_choice_too(server, capsys):
    """The library's yes/no questions were measured as plain yes/no on the fine-tune. On base
    Laya the plain form is the one that collapses (SST-2 0.51 against 0.94 as a choice, §40)."""
    run(capsys, "-q", "work_finished", "--model", "aac6fef/laya-mlx")
    assert server[0]["work_finished"]["type"] == "choice"
