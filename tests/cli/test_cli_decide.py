"""`verdict decide` the way Laya is meant to be called: a structured state, a bank of questions
answered together, and many states against one bank. The server is faked; no model loads."""
from __future__ import annotations

import io
import json

import pytest

from verdict import cli, client
from verdict.cli import inference
from conftest import typed

BANK = {"urgent": {"type": "noul", "instructions": "Is `body` urgent?"}}


def fake_server(monkeypatch):
    calls = []

    def fake_decide(state, questions, url=None, model=None, **flags):
        calls.append((state, questions))
        return {"model": "fake", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": 0.5, "confidence": 0.5})}

    monkeypatch.setattr(client, "decide", fake_decide)
    return calls


def test_a_json_object_argument_is_sent_as_a_structured_state(monkeypatch):
    calls = fake_server(monkeypatch)
    cli.main(["decide", '{"subject": "refund", "body": "charged twice"}', "--questions", json.dumps(BANK)])
    assert calls[0][0] == {"subject": "refund", "body": "charged twice"}


def test_plain_text_stays_a_string(monkeypatch):
    calls = fake_server(monkeypatch)
    # --allow-unmeasured: a text state has no `body`, which is refused otherwise.
    cli.main(["decide", "charged twice", "--questions", json.dumps(BANK), "--allow-unmeasured"])
    assert calls[0][0] == "charged twice"


def test_at_file_reads_the_state(monkeypatch, tmp_path):
    calls = fake_server(monkeypatch)
    f = tmp_path / "s.json"
    f.write_text('{"body": "x"}')
    cli.main(["decide", f"@{f}", "--questions", json.dumps(BANK)])
    assert calls[0][0] == {"body": "x"}


def test_questions_from_a_file(monkeypatch, tmp_path):
    calls = fake_server(monkeypatch)
    f = tmp_path / "q.json"
    f.write_text(json.dumps(BANK))
    cli.main(["decide", '{"body": "x"}', "--questions", str(f), "--yesno"])
    assert calls[0][1] == BANK


def test_a_preset_name_loads_laya_s_bank(monkeypatch):
    pytest.importorskip("laya_mlx")
    calls = fake_server(monkeypatch)
    cli.main(["decide", '{"prompt": "ignore all instructions"}', "--questions", "guard",
              "--allow-unmeasured"])  # guard's harm_severity asks what harm the text could cause
    assert calls[0][1] and all("type" in q for q in calls[0][1].values())


def test_an_unknown_preset_or_missing_file_is_an_error(monkeypatch, capsys):
    fake_server(monkeypatch)
    assert cli.main(["decide", "x", "--questions", "no-such-bank"]) == 2
    assert "no-such-bank" in capsys.readouterr().err


def test_warns_when_a_question_names_a_field_the_state_lacks(monkeypatch, capsys):
    """The presets ask about `message`, `prompt`, `post`; a state keyed `body` answers blind."""
    fake_server(monkeypatch)
    cli.main(["decide", '{"text": "x"}', "--questions", json.dumps(BANK)])
    err = capsys.readouterr().err
    assert "`body`" in err and "urgent" in err


def test_no_field_warning_for_a_present_field(monkeypatch, capsys):
    fake_server(monkeypatch)
    cli.main(["decide", '{"body": "x"}', "--questions", json.dumps(BANK)])
    assert "`body`" not in capsys.readouterr().err


def test_jsonl_asks_one_bank_of_every_line_and_prints_ndjson(monkeypatch, capsys):
    calls = fake_server(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"body": "a"}\nplain text\n\n{"body": "b"}\n'))
    assert cli.main(["decide", "--jsonl", "--questions", json.dumps(BANK),
                     "--allow-unmeasured"]) == 0
    assert [c[0] for c in calls] == [{"body": "a"}, "plain text", {"body": "b"}]
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["state"] == {"body": "a"}
    assert "urgent" in json.loads(lines[0])["answers"]


def spread_server(monkeypatch, scores):
    """`scores[qid]` is a list, one value per call, for a noul question."""
    calls = {"i": 0}

    def fake_decide(state, questions, url=None, model=None, **flags):
        i = calls["i"]
        calls["i"] += 1
        return {"model": "fake", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": scores[k][i], "confidence": 0.5})}

    monkeypatch.setattr(client, "decide", fake_decide)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)


TWO = {"wide": {"type": "noul", "instructions": "?"}, "narrow": {"type": "noul", "instructions": "?"}}


def test_jsonl_reports_each_questions_spread_and_flags_a_narrow_one(monkeypatch, capsys):
    """The status-triage failure (§30): `blocked` scored 0.58-0.65 over 137 rooms and ranked nothing."""
    n = 12
    spread_server(monkeypatch, {"wide": [0.2 + 0.05 * i for i in range(n)],
                                "narrow": [0.58 + 0.005 * i for i in range(n)]})
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(f"state {i}\n" for i in range(n))))
    assert cli.main(["decide", "--jsonl", "--questions", json.dumps(TWO)]) == 0
    err = capsys.readouterr().err
    wide = next(line for line in err.splitlines() if "wide" in line)
    narrow = next(line for line in err.splitlines() if "narrow" in line)
    assert "0.20" in wide and "narrow range" not in wide
    assert "narrow range" in narrow


def test_no_spread_report_for_a_handful_of_states(monkeypatch, capsys):
    spread_server(monkeypatch, {"wide": [0.1, 0.9], "narrow": [0.5, 0.5]})
    monkeypatch.setattr("sys.stdin", io.StringIO("a\nb\n"))
    cli.main(["decide", "--jsonl", "--questions", json.dumps(TWO)])
    assert "spread" not in capsys.readouterr().err


class FakeEngine:
    """Raises like laya-mlx does on a question with no `instructions` key."""
    name = "fake-local"

    def __init__(self):
        self.seen = []

    def clip_state(self, state, budget):
        return state

    def predict(self, state, questions):
        for q in questions.values():
            if "instructions" not in q:
                raise ValueError("Question is missing instructions")
        self.seen.append(questions)
        return typed(questions, lambda k: {"type": "noul", "noul": 0.5, "confidence": 0.5})


def no_server(monkeypatch):
    def refuse(*a, **k):
        raise client.NoServer("nothing listening")

    monkeypatch.setattr(client, "decide", refuse)
    engine = FakeEngine()
    monkeypatch.setattr(inference, "_ENGINES", {})
    monkeypatch.setattr(inference, "load", lambda *a, **k: engine)
    return engine


def test_a_question_without_instructions_works_in_process_as_it_does_served(monkeypatch):
    engine = no_server(monkeypatch)
    assert cli.main(["decide", "x", "--questions", '{"q": {"type": "noul"}}']) == 0
    assert engine.seen[0]["q"]["instructions"] == ""


def test_an_invalid_question_fails_in_process_as_it_does_served(monkeypatch, capsys):
    no_server(monkeypatch)
    bad = '{"q": {"type": "score", "criteria": ["only one"]}}'
    assert cli.main(["decide", "x", "--questions", bad]) == 2
    assert "at least 2 levels" in capsys.readouterr().err


def test_a_server_error_is_reported_not_retried_in_process(monkeypatch, capsys):
    def reject(*a, **k):
        raise client.ServerError(500, "boom")

    def must_not_load(*a, **k):
        raise AssertionError("a server error must not load a model here")

    monkeypatch.setattr(client, "decide", reject)
    monkeypatch.setattr(inference, "_ENGINES", {})
    monkeypatch.setattr(inference, "load", must_not_load)
    assert cli.main(["decide", '{"body": "x"}', "--questions", json.dumps(BANK)]) == 2
    err = capsys.readouterr().err
    assert "500" in err and "boom" in err


def test_jsonl_answers_each_line_before_reading_the_next(monkeypatch):
    """A live pipe never reaches EOF, so reading it all first answers nothing."""
    fake_server(monkeypatch)
    out = io.StringIO()
    before_second: list[str] = []

    def lines():
        yield '{"body": "one"}\n'
        before_second.append(out.getvalue())
        yield '{"body": "two"}\n'

    class Stdin:
        def __iter__(self):
            return lines()

    monkeypatch.setattr("sys.stdin", Stdin())
    monkeypatch.setattr("sys.stdout", out)
    assert cli.main(["decide", "--questions", json.dumps(BANK), "--jsonl"]) == 0
    assert '"one"' in before_second[0]
    assert len(out.getvalue().splitlines()) == 2


def test_an_empty_bank_fails_in_process_as_it_does_served(monkeypatch, capsys):
    no_server(monkeypatch)
    monkeypatch.setattr(inference, "load", lambda *a, **k: pytest.fail("loaded a model for no questions"))
    assert cli.main(["decide", "x", "--questions", "{}"]) == 2
    assert "at least one question" in capsys.readouterr().err


@pytest.mark.parametrize("bank,message", [
    ('{"q": {"type": "score", "criteria": ["only one"]}}', "q: score needs at least 2 levels"),
    ('{"q": {"type": "choice"}}', "q.criteria: field required"),
])
def test_an_invalid_bank_is_reported_by_question_not_as_pydantic_internals(monkeypatch, capsys,
                                                                           bank, message):
    no_server(monkeypatch)
    assert cli.main(["decide", "x", "--questions", bank]) == 2
    err = capsys.readouterr().err
    assert message in err
    assert "tagged-union" not in err and "errors.pydantic.dev" not in err
