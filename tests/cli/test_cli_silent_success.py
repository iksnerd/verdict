"""Failures an agent would read as success: a silent second model load, malformed input answered
as text, output rows that cannot be paired with their input, and arguments accepted out of range.
Each exits 2 with one `verdict:` line, or says on stderr what it did instead."""
import io
import json
import urllib.error

import pytest

from verdict import cli, client
from verdict.cli import inference
from conftest import typed

BANK = '{"q": {"type": "noul", "instructions": "Is `message` about money?"}}'


class FakeEngine:
    name = "fake-local"

    def clip_state(self, state, budget):
        return state

    def predict(self, state, questions):
        return typed(questions, lambda k: {"type": "noul", "noul": 0.7, "confidence": 0.7})


@pytest.fixture(autouse=True)
def no_inference(monkeypatch):
    monkeypatch.setattr(inference, "load", lambda *a, **k: pytest.fail("loaded a model"))
    monkeypatch.setattr(inference, "_ENGINES", {})


def served(monkeypatch):
    sent = []

    def fake_decide(state, questions, url=None, model=None, **flags):
        sent.append(state)
        return {"model": "served", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": 0.6, "confidence": 0.6})}

    monkeypatch.setattr(client, "decide", fake_decide)
    return sent


# 1. never load a second model without saying so

def test_a_slow_live_server_is_a_timeout_not_a_missing_server(monkeypatch):
    def slow(*a, **k):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(client.urllib.request, "urlopen", slow)
    with pytest.raises(client.ServerTimeout):
        client.decide("x", {"q": {"type": "noul", "instructions": "?"}}, "http://127.0.0.1:1")
    monkeypatch.setattr(client.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("timed out")))
    with pytest.raises(client.ServerTimeout):
        client.route("x", "http://127.0.0.1:1")


def test_route_waits_for_a_real_answer(monkeypatch):
    seen = {}

    def fake_post(base, path, body, timeout, expect):
        seen["timeout"] = timeout
        return {"model": "m", "branch": "small", "reason": "", "scores": {}}

    monkeypatch.setattr(client, "_post", fake_post)
    client.route("x")
    assert seen["timeout"] >= client.BATCH_TIMEOUT_BASE


def test_a_timeout_fails_instead_of_loading_locally(monkeypatch, capsys):
    def slow(*a, **k):
        raise client.ServerTimeout("http://127.0.0.1:8799", 5.0)

    monkeypatch.setattr(client, "decide", slow)
    assert cli.main(["decide", '{"message": "refund"}', "-q", BANK]) == 2
    err = capsys.readouterr().err
    assert "did not answer within" in err and "Traceback" not in err


def test_falling_back_to_a_local_load_says_so_once(monkeypatch, capsys):
    def offline(*a, **k):
        raise client.NoServer("no verdict server at http://127.0.0.1:8799 (refused)")

    monkeypatch.setattr(client, "decide", offline)
    monkeypatch.setattr(inference, "load", lambda *a, **k: FakeEngine())
    monkeypatch.setattr("sys.stdin", io.StringIO('{"message": "a"}\n{"message": "b"}\n'))
    assert cli.main(["decide", "--jsonl", "-q", BANK, "--lang", "en", "--pause", "0"]) == 0
    err = capsys.readouterr().err
    assert err.count("loading the model in this process") == 1


# 2. malformed JSON is an error, not text

@pytest.mark.parametrize("cmd", [["decide", '{"message": ', "-q", BANK],
                                 ["ask", '{"message": ', "Is `message` about money?"]])
def test_a_state_that_looks_like_json_must_parse(monkeypatch, capsys, cmd):
    served(monkeypatch)
    assert cli.main(cmd) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_bracketed_text_is_still_text(monkeypatch, capsys):
    sent = served(monkeypatch)
    assert cli.main(["ask", "[WIP] fix the login bug", "Is this about a bug?"]) == 0
    assert sent == ["[WIP] fix the login bug"]


def test_jsonl_names_a_malformed_line_and_keeps_earlier_output(monkeypatch, capsys):
    served(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"message": "a"}\n{"message": \n'))
    assert cli.main(["decide", "--jsonl", "-q", BANK, "--pause", "0"]) == 2
    out, err = capsys.readouterr()
    answered, failure = [json.loads(l) for l in out.splitlines()]
    assert answered["index"] == 0
    assert failure["error"]["code"] == "usage" and "line 2" in failure["error"]["message"]
    assert "line 2" in err and "not valid JSON" in err


def test_jsonl_warns_about_a_plain_text_line(monkeypatch, capsys):
    sent = served(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"message": "a"}\nnot json\n'))
    assert cli.main(["decide", "--jsonl", "-q", "{\"q\": {\"type\": \"noul\", \"instructions\": \"?\"}}",
                     "--pause", "0"]) == 0
    assert sent[1] == "not json"
    assert "line 2 is not JSON" in capsys.readouterr().err


# 3. every --jsonl row can be paired with its input line

def test_jsonl_rows_carry_their_input_line_index(monkeypatch, capsys):
    served(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"message": "a"}\n\n{"message": "b"}\n'))
    assert cli.main(["decide", "--jsonl", "-q", BANK, "--pause", "0"]) == 0
    rows = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert [(r["index"], r["state"]["message"]) for r in rows] == [(0, "a"), (2, "b")]


# 6. arguments are checked before anything is asked

@pytest.mark.parametrize("cut,needle", [("1.5", "between 0 and 1"),
                                        ("abc", "neither a number between 0 and 1 nor a file")])
def test_cut_must_be_a_probability_or_a_file(monkeypatch, capsys, cut, needle):
    served(monkeypatch)
    assert cli.main(["ask", "refund me", "Is this about money?", "--cut", cut]) == 2
    assert needle in capsys.readouterr().err


@pytest.mark.parametrize("cmd", [["ask", "   ", "Is this about money?"],
                                 ["decide", "", "-q", '{"q": {"type": "noul", "instructions": "?"}}']])
def test_an_empty_state_is_refused(monkeypatch, capsys, cmd):
    served(monkeypatch)
    assert cli.main(cmd) == 2
    assert "empty" in capsys.readouterr().err


def test_no_bank_on_stdin_says_so(monkeypatch, capsys):
    served(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main(["decide", '{"message": "x"}']) == 2
    err = capsys.readouterr().err
    assert "no --questions" in err and "Expecting value" not in err


def test_a_missing_bank_file_is_named_as_missing(capsys, tmp_path):
    assert cli.main(["decide", "x", "-q", str(tmp_path / "bank.json")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_a_missing_field_is_not_blamed_on_chance(monkeypatch, capsys):
    served(monkeypatch)
    assert cli.main(["decide", '{"text": "x"}', "-q", BANK]) == 2
    err = capsys.readouterr().err
    assert "message" in err and "measured at chance" not in err


# an agent's guessed flag must not turn into a different mode and wait on stdin

def test_an_abbreviated_flag_is_not_expanded(monkeypatch, capsys):
    served(monkeypatch)
    with pytest.raises(SystemExit):
        cli.main(["decide", '{"message": "x"}', "-q", BANK, "--json"])
    assert "unrecognized arguments: --json" in capsys.readouterr().err


def test_jsonl_with_a_state_argument_is_refused(monkeypatch, capsys):
    served(monkeypatch)
    assert cli.main(["decide", '{"message": "x"}', "-q", BANK, "--jsonl"]) == 2
    assert "reads states from stdin" in capsys.readouterr().err


def test_jsonl_from_a_terminal_is_refused_instead_of_waiting(monkeypatch, capsys):
    served(monkeypatch)

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", Terminal(""))
    assert cli.main(["decide", "-q", BANK, "--jsonl"]) == 2
    assert "terminal" in capsys.readouterr().err
