"""Agent workflows must validate offline and never load a fallback when forbidden."""
import io
import json

import pytest

from verdict import cli, client, config
from verdict.cli import inference
from conftest import typed


@pytest.fixture(autouse=True)
def no_inference(monkeypatch):
    monkeypatch.setattr(inference, "load", lambda *a, **k: pytest.fail("loaded a model"))


def test_validate_stdin_returns_normalized_bank(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO('{"q":{"type":"noul"}}'))
    assert cli.main(["validate", "-q", "-", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["valid"] is True
    assert out["questions"]["q"]["instructions"] == ""


def test_validate_invalid_bank_is_machine_readable(capsys):
    assert cli.main(["validate", "-q", '{"q":{"type":"score","criteria":["one"]}}',
                     "--json"]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["valid"] is False and out["error"]


def test_validate_reports_missing_file_as_json(capsys, tmp_path):
    assert cli.main(["validate", "-q", str(tmp_path / "missing.json"), "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_server_only_does_not_resolve_or_load_local_weights(monkeypatch, capsys):
    monkeypatch.setattr(config, "resolve_model", lambda *a, **k: pytest.fail("resolved weights"))
    def offline(*a, **k):
        raise client.NoServer("connection refused")
    monkeypatch.setattr(client, "decide", offline)
    assert cli.main(["ask", "hello", "Is this a greeting?", "--server-only", "--lang", "en"]) == 2
    assert "verdict serve" in capsys.readouterr().err


@pytest.mark.parametrize("bank", ['[]', '{"q":null}', '{}'])
def test_bad_bank_fails_before_network_or_lint(monkeypatch, capsys, bank):
    monkeypatch.setattr(client, "decide", lambda *a, **k: pytest.fail("called server"))
    assert cli.main(["decide", "x", "-q", bank]) == 2
    assert "verdict:" in capsys.readouterr().err


def test_server_only_success_keeps_answer_shape(monkeypatch, capsys):
    monkeypatch.setattr(client, "decide", lambda state, questions, *a, **k: {
        "model": "fake", "answers": typed(
            questions, lambda q: {"type": "noul", "noul": 0.7, "confidence": 0.7})})
    assert cli.main(["ask", "hello", "Is this a greeting?", "--server-only", "--json",
                     "--lang", "en"]) == 0
    assert json.loads(capsys.readouterr().out)["answers"]["q"]["noul"] == 0.7


def test_validate_json_error_names_the_question_and_the_problem(capsys):
    assert cli.main(["validate", "-q", '{"q":{"type":"score","criteria":["one"]}}', "--json"]) == 2
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["message"] == "invalid questions: q: score needs at least 2 levels"
