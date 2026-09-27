"""The rest of the CLI surface: score and sides in `ask`, `presets`, `calibrate`, applying a fit,
`--version` and help. The server is faked throughout; no model loads."""
from __future__ import annotations

import io
import json
import re

import pytest

from verdict import cli, client
from conftest import typed


def fake_server(monkeypatch, answer_for):
    sent = []

    def fake_decide(state, questions, url=None, model=None, **flags):
        sent.append((state, questions, model))
        return {"model": "fake", "answers": typed(
            questions, lambda k: answer_for(state, k, questions[k]))}

    monkeypatch.setattr(client, "decide", fake_decide)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    return sent


SCORE = {"type": "score", "score": 1.3, "confidence": 0.2, "legend": {"0": "low", "1": "mid", "2": "high"},
         "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5}}


def test_init_preserves_model_settings(monkeypatch, tmp_path):
    from verdict import config
    from verdict.cli import setup

    path = tmp_path / "config.toml"
    before = config.Settings("http://127.0.0.1:8799", "org/model", 128, bits=8,
                             lang="multi", extra_checkpoints={"support": "org/support"})
    path.write_text(config.to_toml(before))
    monkeypatch.setattr(config, "CONFIG_PATHS", (path,))
    monkeypatch.setattr(setup, "_port_free", lambda *a: True)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--yes", "--out", str(path)]) == 0
    after = config.load(path)
    assert (after.bits, after.lang, after.extra_checkpoints) == (8, "multi", {"support": "org/support"})


def test_calibrate_missing_choice_labels_fails_before_inference(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(client, "decide", lambda *a, **kw: pytest.fail("must validate before asking"))
    path = tmp_path / "labelled.jsonl"
    path.write_text("\n".join(json.dumps({"state": "x", "labels": {"a": "yes"}}) for _ in range(10)))
    bank = {k: {"type": "choice", "criteria": {"yes": "yes", "no": "no"}} for k in ("a", "b")}
    assert cli.main(["calibrate", str(path), "-q", json.dumps(bank)]) == 2
    assert "b" in capsys.readouterr().err


def test_levels_make_a_score_and_print_expectation_and_top_level(monkeypatch, capsys):
    sent = fake_server(monkeypatch, lambda s, k, q: SCORE)
    assert cli.main(["ask", "x", "how urgent?", "-l", "low", "-l", "mid", "-l", "high"]) == 0
    assert sent[0][1]["q"] == {"type": "score", "instructions": "how urgent?",
                               "criteria": ["low", "mid", "high"]}
    assert capsys.readouterr().out.strip() == "1.30/2 high"


def test_true_false_describe_the_sides(monkeypatch):
    sent = fake_server(monkeypatch, lambda s, k, q: {"type": "noul", "noul": 0.5, "confidence": 0.5})
    cli.main(["ask", "x", "safe?", "--true", "read-only", "--false", "writes", "--yesno"])
    assert sent[0][1]["q"]["criteria"] == {"true": "read-only", "false": "writes"}


def test_options_and_levels_together_is_an_error(monkeypatch, capsys):
    fake_server(monkeypatch, lambda s, k, q: SCORE)
    assert cli.main(["ask", "x", "q?", "-o", "a", "-o", "b", "-l", "lo", "-l", "hi"]) == 2


def test_presets_lists_every_bank_with_its_field(capsys):
    pytest.importorskip("laya_mlx")
    assert cli.main(["presets"]) == 0
    out = capsys.readouterr().out
    for name in ("guard", "triage", "moderation", "router", "email"):
        assert name in out
    assert "`message`" in out


def test_presets_name_prints_the_bank_as_json(capsys):
    pytest.importorskip("laya_mlx")
    assert cli.main(["presets", "guard"]) == 0
    bank = json.loads(capsys.readouterr().out)
    assert bank and all("type" in q for q in bank.values())


def test_calibrate_fits_a_cut_and_writes_a_file_decide_applies(monkeypatch, tmp_path, capsys):
    score = {"bad": 0.7, "fine": 0.3}
    fake_server(monkeypatch, lambda s, k, q: {"type": "noul", "noul": score[s["kind"]] + s["j"],
                                              "confidence": 0.5})
    rows = [{"state": {"kind": "bad", "j": i / 1000}, "label": True} for i in range(20)]
    rows += [{"state": {"kind": "fine", "j": i / 1000}, "label": "no"} for i in range(20)]
    data = tmp_path / "labelled.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    bank = json.dumps({"risky": {"type": "noul", "instructions": "Is `kind` marked bad?"}})
    out = tmp_path / "fit.json"

    assert cli.main(["calibrate", str(data), "--questions", bank, "--out", str(out)]) == 0
    fit = json.loads(out.read_text())["questions"]["risky"]
    assert 0.32 < fit["cut"] < 0.7 and fit["heldout"]["auc"] == 1.0
    capsys.readouterr()

    cli.main(["decide", '{"kind": "bad", "j": 0}', "--questions", bank, "--calibration", str(out)])
    assert json.loads(capsys.readouterr().out)["answers"]["risky"]["decision"] is True


def test_calibrate_limit_caps_the_calls(monkeypatch, tmp_path):
    sent = fake_server(monkeypatch, lambda s, k, q: {"type": "noul", "noul": 0.5, "confidence": 0.5})
    data = tmp_path / "l.jsonl"
    data.write_text("\n".join(json.dumps({"state": str(i), "label": i % 2 == 0}) for i in range(50)))
    cli.main(["calibrate", str(data), "--questions", '{"q": {"type": "noul", "instructions": "?"}}',
              "--limit", "10"])
    assert len(sent) == 10


def test_ask_cut_from_a_calibration_file(monkeypatch, tmp_path, capsys):
    fake_server(monkeypatch, lambda s, k, q: {"type": "noul", "noul": 0.45, "confidence": 0.45})
    f = tmp_path / "fit.json"
    f.write_text(json.dumps({"questions": {"q": {"type": "noul", "cut": 0.4}}}))
    assert cli.main(["ask", "x", "q?", "--cut", str(f)]) == 0
    assert capsys.readouterr().out.strip() == "yes 0.45"


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert "verdict" in capsys.readouterr().out


@pytest.mark.parametrize("cmd", ["ask", "decide", "presets", "calibrate", "route", "serve"])
def test_every_command_has_examples_in_its_help(cmd, capsys):
    with pytest.raises(SystemExit):
        cli.main([cmd, "--help"])
    out = capsys.readouterr().out
    assert "verdict " + cmd in out and "Examples" in out


def test_route_has_no_exec(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["route", "x", "--exec"])
    assert e.value.code == 2


def test_help_never_mentions_ollama(capsys):
    for cmd in ([], ["ask"], ["decide"], ["route"], ["init"], ["serve"], ["cases"]):
        with pytest.raises(SystemExit):
            cli.main(cmd + ["--help"])
    assert "ollama" not in capsys.readouterr().out.lower()


def test_init_writes_a_loadable_file_with_absolute_paths_and_no_routes(monkeypatch, tmp_path):
    from verdict import config

    (tmp_path / "models" / "verdict-v1-mlx").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATHS", (tmp_path / "none.toml",))
    out = tmp_path / "written.toml"
    assert cli.main(["init", "--yes", "--out", str(out)]) == 0
    text = out.read_text()
    assert "[routes" not in text and "ollama" not in text
    s = config.load(out)
    assert s.model_path == str(tmp_path / "models" / "verdict-v1-mlx")


def test_init_defaults_to_the_user_config(monkeypatch):
    from verdict import config

    parser_default = None
    real = cli.argparse.ArgumentParser.parse_args

    def spy(self, argv=None, namespace=None):
        nonlocal parser_default
        ns = real(self, argv, namespace)
        parser_default = getattr(ns, "out", None)
        raise SystemExit(0)

    monkeypatch.setattr(cli.argparse.ArgumentParser, "parse_args", spy)
    with pytest.raises(SystemExit):
        cli.main(["init"])
    assert parser_default == str(config.USER_CONFIG)


def test_calibrate_prints_a_rounded_cut_and_saves_the_exact_one(monkeypatch, tmp_path, capsys):
    """The file keeps the exact cut (a rounded one can cross tightly packed scores); the printed
    line is for a person. Scores in thirds of a thousandth make the midpoint cut a long float."""
    score = {"bad": 0.7, "fine": 0.3}
    fake_server(monkeypatch, lambda s, k, q: {"type": "noul", "noul": score[s["kind"]] + s["j"],
                                              "confidence": 0.5})
    rows = [{"state": {"kind": k, "j": i / 3000}, "label": k == "bad"}
            for k in ("bad", "fine") for i in range(20)]
    data = tmp_path / "labelled.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    bank = json.dumps({"risky": {"type": "noul", "instructions": "Is `kind` marked bad?"}})
    out = tmp_path / "fit.json"
    assert cli.main(["calibrate", str(data), "--questions", bank, "--out", str(out)]) == 0
    saved = json.loads(out.read_text())["questions"]["risky"]["cut"]
    assert len(repr(saved).split(".")[1]) > 4, "precondition: the exact cut is long"
    printed = re.search(r"cut (\S+)", capsys.readouterr().out).group(1)
    assert len(printed.split(".")[1]) <= 4
    assert abs(float(printed) - saved) < 1e-4


def test_init_out_reads_its_own_target_not_the_search_path(monkeypatch, tmp_path):
    from verdict import config
    from verdict.cli import setup

    target = tmp_path / "custom.toml"
    target.write_text(config.to_toml(config.Settings(
        "http://127.0.0.1:8799", "org/model", 128, bits=8, lang="multi",
        extra_checkpoints={"support": "org/support"})))
    monkeypatch.setattr(config, "CONFIG_PATHS", (tmp_path / "absent.toml",))
    monkeypatch.setattr(setup, "_port_free", lambda *a: True)
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--yes", "--out", str(target)]) == 0
    after = config.load(target)
    assert (after.bits, after.lang, after.extra_checkpoints) == (8, "multi", {"support": "org/support"})


def test_calibrate_records_the_model_that_answered(monkeypatch, tmp_path):
    fake_server(monkeypatch, lambda s, k, q: {"type": "noul", "noul": 0.9 if s["y"] else 0.1,
                                              "confidence": 0.5})
    data = tmp_path / "l.jsonl"
    data.write_text("\n".join(json.dumps({"state": {"y": i % 2 == 0, "i": i}, "label": i % 2 == 0})
                              for i in range(40)))
    out = tmp_path / "fit.json"
    assert cli.main(["calibrate", str(data), "--questions",
                     '{"q": {"type": "noul", "instructions": "?"}}', "--out", str(out)]) == 0
    assert json.loads(out.read_text())["model"] == "fake"


def test_bench_systemone_with_a_bad_url_is_a_usage_error(monkeypatch, capsys):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert cli.main(["bench", "--systemone", "not-a-url"]) == 2
    assert "not-a-url" in capsys.readouterr().err


def test_calibrate_refuses_answers_from_two_models(monkeypatch, tmp_path, capsys):
    calls = iter(range(10**6))

    def fake_decide(state, questions, url=None, model=None, **flags):
        return {"model": f"m{next(calls) % 2}", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": 0.9 if state["y"] else 0.1, "confidence": 0.5})}

    monkeypatch.setattr(client, "decide", fake_decide)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    data = tmp_path / "l.jsonl"
    data.write_text("\n".join(json.dumps({"state": {"y": i % 2 == 0, "i": i}, "label": i % 2 == 0})
                              for i in range(40)))
    assert cli.main(["calibrate", str(data), "--questions",
                     '{"q": {"type": "noul", "instructions": "?"}}']) == 2
    assert "more than one model" in capsys.readouterr().err
