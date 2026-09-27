"""Questions and states that measured at chance are refused, not answered with a warning.

FINDINGS §25, §29 and §33 found the failing shapes; a warning printed beside an answer was not
enough, because the answer still got used. `--allow-unmeasured` asks anyway. No model loads here."""
from __future__ import annotations

import json

import pytest

from verdict import cli, client
from verdict.cli import inference
from conftest import typed

OFF_PAGE = {"q": {"type": "noul", "instructions": "How hard is `request` for a language model?"}}
ABSENCE = {"q": {"type": "noul", "instructions": "Is `text` an ordinary message?"}}
SURFACE = {"q": {"type": "noul", "instructions": "Does `text` mention a deadline?"}}


@pytest.fixture
def server(monkeypatch):
    calls = []

    def fake_decide(state, questions, url=None, model=None, **kw):
        calls.append((state, questions))
        return {"model": "fake", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": 0.5, "confidence": 0.5})}

    monkeypatch.setattr(client, "decide", fake_decide)
    monkeypatch.setattr(inference, "load", lambda *a, **k: pytest.fail("loaded a model"))
    return calls


@pytest.mark.parametrize("bank", [OFF_PAGE, ABSENCE], ids=["off the page", "an absence"])
def test_decide_refuses_a_question_shape_that_measured_at_chance(server, capsys, bank):
    assert cli.main(["decide", '{"text": "x", "request": "x"}', "-q", json.dumps(bank)]) == 2
    err = capsys.readouterr().err
    assert "FINDINGS" in err and "--allow-unmeasured" in err
    assert server == []


def test_allow_unmeasured_answers_anyway_and_still_warns(server, capsys):
    assert cli.main(["decide", '{"request": "x"}', "-q", json.dumps(OFF_PAGE),
                     "--allow-unmeasured"]) == 0
    assert len(server) == 1
    assert "FINDINGS" in capsys.readouterr().err


def test_decide_refuses_a_state_without_the_field_the_question_reads(server, capsys):
    assert cli.main(["decide", '{"body": "x"}', "-q", json.dumps(SURFACE)]) == 2
    assert "`text`" in capsys.readouterr().err
    assert server == []


def test_a_surface_question_on_a_matching_state_is_answered(server):
    assert cli.main(["decide", '{"text": "due friday"}', "-q", json.dumps(SURFACE)]) == 0
    assert len(server) == 1


def test_ask_refuses_too(server, capsys):
    assert cli.main(["ask", "x", "Could this cause harm that is hard to undo?"]) == 2
    assert server == []


def test_every_measured_library_question_passes(server):
    from verdict import library

    for e in library.load():
        assert cli.main(["validate", "-q", e.name]) == 0, e.name


def test_validate_calls_a_flagged_bank_invalid(capsys):
    assert cli.main(["validate", "-q", json.dumps(OFF_PAGE), "--json"]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["valid"] is False and "FINDINGS" in out["error"]["message"]


def test_validate_with_allow_unmeasured_is_valid_with_warnings(capsys):
    assert cli.main(["validate", "-q", json.dumps(OFF_PAGE), "--json", "--allow-unmeasured"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["valid"] is True and out["warnings"]


def test_calibrate_measures_a_flagged_question_rather_than_refusing_it(server, tmp_path, capsys):
    """Calibration is how a question gets measured, so refusing there would block the proof."""
    data = tmp_path / "l.jsonl"
    data.write_text("\n".join(json.dumps({"state": {"request": str(i)}, "label": i % 2 == 0})
                              for i in range(40)))
    assert cli.main(["calibrate", str(data), "-q", json.dumps(OFF_PAGE), "--pause", "0"]) == 0
    assert "FINDINGS" in capsys.readouterr().err


def many_options(n):
    return {"q": {"type": "choice", "instructions": "Which intent is `text`?",
                  "criteria": {f"intent_{i}": "" for i in range(n)}}}


def test_a_choice_past_20_options_is_refused(server, capsys):
    """The options share one token budget: at 77 each gets 3 or 4 tokens, and Laya scored 0.43
    on Banking77 against Jev's 0.87 (Laya's README; FINDINGS §41)."""
    assert cli.main(["decide", '{"text": "x"}', "-q", json.dumps(many_options(21))]) == 2
    err = capsys.readouterr().err
    assert "21 options" in err and "--allow-unmeasured" in err
    assert server == []


def test_twenty_options_are_still_asked(server):
    assert cli.main(["decide", '{"text": "x"}', "-q", json.dumps(many_options(20))]) == 0


def test_allow_unmeasured_asks_a_long_choice_anyway(server):
    assert cli.main(["decide", '{"text": "x"}', "-q", json.dumps(many_options(21)),
                     "--allow-unmeasured"]) == 0


def test_a_long_state_is_answered_with_a_warning_that_only_its_start_is_read(server, capsys):
    long = json.dumps({"text": "word " * 400})
    assert cli.main(["decide", long, "-q", json.dumps(SURFACE)]) == 0
    assert "first 128 tokens" in capsys.readouterr().err


def test_a_short_state_gets_no_length_warning(server, capsys):
    assert cli.main(["decide", '{"text": "due friday"}', "-q", json.dumps(SURFACE)]) == 0
    assert "first 128 tokens" not in capsys.readouterr().err
