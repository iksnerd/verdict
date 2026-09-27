"""One contract across the surface: the same error shape everywhere, the same checks for text and
JSON states, config files that layer, and an HTTP API that answers as the CLI does. From a second
third-party audit (0.5.0). No model loads."""
import io
import json

import pytest

from verdict import cli, client, config
from verdict.cli import inference, parser, setup
from conftest import typed

BANK = '{"q": {"type": "noul", "instructions": "Is `message` about money?"}}'


@pytest.fixture(autouse=True)
def no_inference(monkeypatch):
    monkeypatch.setattr(inference, "load", lambda *a, **k: pytest.fail("loaded a model"))


def served(monkeypatch, noul=0.6):
    sent = []

    def fake_decide(state, questions, url=None, model=None, **flags):
        sent.append((state, questions, flags))
        return {"model": "served", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": noul, "confidence": 0.6})}

    monkeypatch.setattr(client, "decide", fake_decide)
    return sent


# 1. the flagship example is one that works on the default model

def test_the_help_and_init_example_is_the_measured_wording():
    import inspect

    for text in (parser.EPILOG, inspect.getsource(setup._init_cmd)):
        assert '"Is this an instruction?"' not in text
    assert "Is this an instruction to perform an action?" in parser.EPILOG


# 2. one error envelope

def test_validate_json_errors_use_the_shared_envelope(capsys):
    assert cli.main(["validate", "-q", '{"q":{"type":"score","criteria":["one"]}}', "--json"]) == 2
    out = json.loads(capsys.readouterr().out)
    assert out["valid"] is False
    assert out["error"]["code"] == "refused" and out["error"]["message"]


def test_single_state_decide_puts_its_error_on_stdout_as_json(monkeypatch, capsys):
    served(monkeypatch)
    assert cli.main(["decide", '{"text": "x"}', "-q", BANK]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "refused"


# 3. a text state cannot hold the field a question names

def test_a_plain_text_state_is_refused_for_a_field_question(monkeypatch, capsys):
    served(monkeypatch)
    assert cli.main(["decide", "hello there", "-q", BANK]) == 2
    assert "plain text" in capsys.readouterr().err


# 4. nothing to read is an error

def test_jsonl_with_nothing_on_stdin_is_an_error(monkeypatch, capsys):
    served(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("\n\n"))
    assert cli.main(["decide", "--jsonl", "-q", BANK]) == 2
    assert "no states on stdin" in capsys.readouterr().err


# 5. and 6. config files are checked, and layer

@pytest.mark.parametrize("text,needle", [('[sever]\nurl = "http://127.0.0.1:1"\n', "[sever]"),
                                         ("[model]\ntypo_key = 1\n", "typo_key")])
def test_unknown_config_keys_are_errors(tmp_path, text, needle):
    path = tmp_path / "verdict.toml"
    path.write_text(text)
    with pytest.raises(config.ConfigError, match=needle.replace("[", r"\[").replace("]", r"\]")):
        config.load(path)


def test_a_local_file_layers_over_the_user_file(monkeypatch, tmp_path):
    user, local = tmp_path / "user.toml", tmp_path / "verdict.toml"
    user.write_text('[server]\nurl = "http://127.0.0.1:9100"\n[model]\nlang = "multi"\n')
    local.write_text("[model]\nbits = 8\n")
    monkeypatch.setattr(config, "CONFIG_PATHS", (local, user))
    s = config.load()
    assert (s.url, s.lang, s.bits) == ("http://127.0.0.1:9100", "multi", 8)
    rows = {r["key"]: r["source"] for r in config.explain()}
    assert rows["server.url"] == str(user) and rows["model.bits"] == str(local)


# 7. the HTTP API answers as the CLI does

def api(questions_seen):
    from fastapi.testclient import TestClient

    from verdict.api import create_app
    from verdict.schema import DecideResponse

    class Recording:
        name = "recording"
        model_id = "aac6fef/laya-mlx"

        def decide(self, request):
            questions_seen.append({qid: q.model_dump(exclude_none=True)
                                   for qid, q in request.questions.items()})
            return DecideResponse(model="recording", answers={
                qid: ({"type": "choice", "choice": "yes", "confidence": 0.5,
                       "probabilities": {"no": 0.2, "yes": 0.8}} if q.type == "choice"
                      else {"type": "noul", "noul": 0.3, "confidence": 0.7})
                for qid, q in request.questions.items()})

    return TestClient(create_app(Recording()))


def test_the_api_asks_a_new_yes_no_as_a_choice_like_the_cli():
    seen = []
    body = api(seen).post("/v1/decide", json={"state": {"message": "refund"},
                                              "questions": json.loads(BANK)}).json()
    assert seen[-1]["q"]["type"] == "choice"
    assert body["answers"]["q"] == {"type": "noul", "noul": 0.8, "confidence": 0.8}


def test_the_api_yesno_param_sends_a_plain_yes_no():
    seen = []
    api(seen).post("/v1/decide?yesno=true", json={"state": {"message": "refund"},
                                                   "questions": json.loads(BANK)})
    assert seen[-1]["q"]["type"] == "noul"


@pytest.mark.parametrize("state,question", [
    ({"text": "x"}, "Is `message` about money?"),
    ({"message": "x"}, "Could this cause harm?"),
])
def test_the_api_refuses_what_the_cli_refuses(state, question):
    seen = []
    c = api(seen)
    bank = {"q": {"type": "noul", "instructions": question}}
    assert c.post("/v1/decide", json={"state": state, "questions": bank}).status_code == 422
    assert c.post("/v1/decide?allow_unmeasured=true",
                  json={"state": state, "questions": bank}).status_code == 200


def test_the_cli_tells_the_server_its_flags(monkeypatch, capsys):
    sent = served(monkeypatch)
    cli.main(["ask", "refund", "Could this cause harm?", "--allow-unmeasured", "--yesno"])
    assert sent[-1][2] == {"allow_unmeasured": True, "yesno": True}


# 8. a failed line says which one

def test_a_jsonl_error_carries_the_failing_line_index(monkeypatch, capsys):
    served(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"message": "a"}\n\n{"message": \n'))
    assert cli.main(["decide", "--jsonl", "-q", BANK, "--pause", "0"]) == 2
    failure = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert failure["error"]["index"] == 2


# 9. arguments that cannot mean anything

@pytest.mark.parametrize("cmd,needle", [
    (["ask", "hi", "  "], "question is empty"),
    (["ask", "hi", "Which?", "-o", "a", "-o", "b", "--cut", "0.5"], "a numeric --cut applies to a yes/no"),
])
def test_meaningless_ask_arguments_are_refused(monkeypatch, capsys, cmd, needle):
    served(monkeypatch)
    assert cli.main(cmd) == 2
    assert needle in capsys.readouterr().err


def test_a_negative_pause_is_refused(capsys):
    with pytest.raises(SystemExit):
        cli.main(["decide", "--jsonl", "-q", BANK, "--pause", "-1"])
    assert "--pause" in capsys.readouterr().err


# 10. what is printed is exactly true

def test_a_cut_verdict_never_prints_a_score_that_reads_the_other_way(monkeypatch, capsys):
    served(monkeypatch, noul=0.4964)
    assert cli.main(["ask", "hi", "Is this a greeting?", "--cut", "0.5", "--yesno"]) == 1
    assert capsys.readouterr().out.strip() == "no 0.4964"


def test_init_does_not_offer_the_uniform_backend(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "CONFIG_PATHS", (tmp_path / "none.toml",))
    monkeypatch.setattr(setup, "_port_free", lambda *a: True)
    monkeypatch.chdir(tmp_path)
    cli.main(["init", "--yes", "--out", str(tmp_path / "c.toml")])
    assert "uniform" not in capsys.readouterr().out


def test_the_bench_hint_covers_a_tool_install(monkeypatch):
    import builtins

    from verdict import bench

    real = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", lambda name, *a, **k: (
        (_ for _ in ()).throw(ImportError(name)) if name == "datasets" else real(name, *a, **k)))
    with pytest.raises(ValueError, match="uv tool install"):
        bench.fetch(bench.load_suites()[0])


def test_docs_help_lists_every_topic(capsys):
    with pytest.raises(SystemExit):
        cli.main(["docs", "--help"])
    out = capsys.readouterr().out
    assert "pipeline" in out and "routing" in out


@pytest.mark.parametrize("cmd,needle", [
    (["examples", "nosuchthing"], "no example nosuchthing. There are:"),
    (["decide", "@/nonexistent/state.json", "-q", BANK], "no such file"),
    (["decide", "x", "-q", "[]"], "a question bank is a JSON object"),
])
def test_clear_messages(monkeypatch, capsys, cmd, needle):
    served(monkeypatch)
    assert cli.main(cmd) == 2
    assert needle in capsys.readouterr().err


def test_validate_names_an_empty_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main(["validate", "-q", "-"]) == 2
    assert "no bank on stdin" in capsys.readouterr().err


def test_route_json_exits_zero_on_success(monkeypatch, capsys):
    from verdict.cli import routing
    from verdict.switch import Branch

    monkeypatch.setattr(routing, "decide", lambda *a, **k: (Branch("big", "r", {}), 1.0, "server"))
    assert cli.main(["route", "hard prompt", "--json"]) == 0


def test_validate_warns_about_a_question_with_no_instructions(capsys):
    assert cli.main(["validate", "-q", '{"q": {"type": "noul"}}', "--json"]) == 0
    warnings = json.loads(capsys.readouterr().out)["warnings"]
    assert any("no instructions" in w for w in warnings)
