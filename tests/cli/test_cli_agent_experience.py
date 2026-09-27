"""What an agent needs from the CLI beyond right answers: errors it can branch on, a stderr it can
read, suggestions for a near miss, the resolved settings, machine-readable catalogs, and help
that names every setting. No model loads."""
import io
import json
import warnings

import pytest

from verdict import cli, client, config
from verdict.cli import inference
from conftest import typed

BANK = '{"q": {"type": "noul", "instructions": "Is `message` about money?"}}'


@pytest.fixture(autouse=True)
def no_inference(monkeypatch):
    monkeypatch.setattr(inference, "load", lambda *a, **k: pytest.fail("loaded a model"))


def served(monkeypatch, model="served"):
    def fake_decide(state, questions, url=None, model_=None, **k):
        return {"model": model, "answers": typed(
            questions, lambda q: {"type": "noul", "noul": 0.6, "confidence": 0.6})}

    monkeypatch.setattr(client, "decide", lambda s, q, url=None, model=None: fake_decide(s, q))


# 4. errors an agent can branch on

@pytest.mark.parametrize("cmd,code", [
    (["ask", " ", "Is this about money?", "--json"], "usage"),
    (["ask", '{"text": "x"}', "Is `message` about money?", "--json"], "refused"),
    (["decide", '{"message": "x"}', "-q", "nope_missing_question", "--json"], "usage"),
])
def test_json_mode_errors_are_json_on_stdout(monkeypatch, capsys, cmd, code):
    served(monkeypatch)
    if "--json" in cmd and cmd[0] == "decide":
        cmd = [c for c in cmd if c != "--json"] + ["--jsonl"]
        monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main(cmd) == 2
    out, err = capsys.readouterr()
    error = json.loads(out.strip().splitlines()[-1])["error"]
    assert error["code"] == code and error["message"]
    assert err.startswith("verdict: ")


def test_json_mode_reports_a_missing_server_as_such(monkeypatch, capsys):
    def offline(*a, **k):
        raise client.NoServer("no verdict server at http://127.0.0.1:8799 (refused)")

    monkeypatch.setattr(client, "decide", offline)
    assert cli.main(["ask", "hi", "Is this a greeting?", "--server-only", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "no_server"


# 5. third-party noise stays off stderr

def test_the_cli_silences_laya_temperature_warnings_and_hub_chatter(monkeypatch):
    monkeypatch.delenv("HF_HUB_VERBOSITY", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        cli.main(["presets"])
        warnings.warn("laya-mlx: this checkpoint ships temperatures outside [0.5, 5]",
                      RuntimeWarning)
    assert not caught
    import os
    assert os.environ["HF_HUB_VERBOSITY"] == "error"


# 7. a near miss gets a suggestion

@pytest.mark.parametrize("cmd,suggestion", [
    (["aks", "x", "y"], "ask"),
    (["questions", "is_instrution"], "is_instruction"),
    (["presets", "triag"], "triage"),
    (["decide", '{"message": "x"}', "-q", "is_instrution"], "is_instruction"),
    (["examples", "test-outptu"], "test-output"),
    (["docs", "gide"], "guide"),
])
def test_a_typo_gets_a_did_you_mean(monkeypatch, capsys, cmd, suggestion):
    served(monkeypatch)
    with pytest.raises(SystemExit) if cmd[0] in ("aks", "docs") else _nullcontext():
        assert cli.main(cmd) == 2
    assert f"did you mean {suggestion}" in capsys.readouterr().err


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# 8. the resolved settings, and where each came from

def test_config_shows_each_value_and_its_source(monkeypatch, tmp_path, capsys):
    path = tmp_path / "verdict.toml"
    path.write_text('[model]\nbits = 8\n')
    monkeypatch.setattr(config, "CONFIG_PATHS", (path,))
    monkeypatch.setenv("VERDICT_LANG", "multi")
    assert cli.main(["config", "--json"]) == 0
    rows = {r["key"]: r for r in json.loads(capsys.readouterr().out)["settings"]}
    assert (rows["model.bits"]["value"], rows["model.bits"]["source"]) == (8, str(path))
    assert (rows["model.lang"]["value"], rows["model.lang"]["source"]) == ("multi", "$VERDICT_LANG")
    assert rows["server.url"]["source"] == "default"
    assert cli.main(["config"]) == 0
    assert "$VERDICT_LANG" in capsys.readouterr().out


def test_model_flag_ignored_by_a_server_says_so(monkeypatch, capsys, tmp_path):
    served(monkeypatch, model="laya-mlx:aac6fef/laya-mlx")
    model = tmp_path / "other-mlx"
    model.mkdir()
    assert cli.main(["ask", "hi", "Is this a greeting?", "--model", str(model)]) == 0
    assert "--model" in capsys.readouterr().err


# 9. catalogs an agent can parse, and an installed example's path

def test_questions_and_presets_as_json(capsys):
    assert cli.main(["questions", "--json"]) == 0
    questions = json.loads(capsys.readouterr().out)
    assert questions["is_instruction"]["question"]["type"] == "noul"
    assert "measured" in questions["is_instruction"]
    assert cli.main(["presets", "--json"]) == 0
    assert "intent" in json.loads(capsys.readouterr().out)["triage"]


def test_examples_path_prints_the_directory(capsys):
    assert cli.main(["examples", "--path", "test-output"]) == 0
    from pathlib import Path
    path = Path(capsys.readouterr().out.strip())
    assert (path / "bank.json").is_file()


# 10. help and metadata tell the truth

def test_top_level_help_names_every_environment_variable(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for var in config.ENV_OVERRIDES.values():
        assert var in out


def test_openapi_reports_the_package_version():
    from importlib.metadata import version
    from verdict.api import create_app

    assert create_app().version == version("verdict")


@pytest.mark.parametrize("topic", ["pipeline", "routing"])
def test_docs_opens_every_doc(capsys, topic):
    assert cli.main(["docs", topic]) == 0
    assert capsys.readouterr().out.startswith("#")


def test_examples_output_labels_a_choice_probability(monkeypatch, capsys):
    from verdict.cli import examples_cmd

    line = examples_cmd._format({"kind": {"type": "choice", "choice": "nit", "confidence": 0.1,
                                          "probabilities": {"nit": 0.7, "praise": 0.3}}})
    assert line == "kind=nit(p=0.70)"
