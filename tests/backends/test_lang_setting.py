"""`[model].lang`: which checkpoint reads the state, set once instead of `--lang` on every call.

For Cyrillic (and other non-English) text the English checkpoint is confidently wrong: a glowing
Bulgarian review read "negative 0.72" there and "positive 1.00" on the multilingual one. A script
that forgot `--lang multi` got that wrong answer with only a warning."""
from __future__ import annotations

import argparse

from verdict import cli, client, config
from conftest import typed
from fakes import noul


def test_lang_defaults_to_auto():
    assert config.load().lang == "auto"


def test_lang_from_the_environment(monkeypatch):
    monkeypatch.setenv("VERDICT_LANG", "multi")
    assert config.load().lang == "multi"


def test_lang_from_the_file(config_files):
    config_files('[model]\nlang = "multi"\n')
    assert config.load().lang == "multi"


def test_a_bad_lang_is_a_clean_error(monkeypatch, capsys):
    monkeypatch.setenv("VERDICT_LANG", "bulgarian")
    assert cli.main(["ask", "x", "q?", "--server-only"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("verdict: ") and "auto, en or multi" in err


def test_init_writes_the_lang_line():
    assert 'lang = "auto"' in config.to_toml(config.load())


def asked_model(monkeypatch, argv):
    seen = {}

    def fake_decide(state, questions, url=None, model=None, **flags):
        seen["model"] = model
        return {"model": "fake", "answers": typed(
            questions, lambda k: noul(0.5))}
    monkeypatch.setattr(client, "decide", fake_decide)
    assert cli.main(argv) == 0
    return seen["model"]


def test_the_setting_sends_calls_to_the_multilingual_checkpoint(monkeypatch):
    monkeypatch.setenv("VERDICT_LANG", "multi")
    assert asked_model(monkeypatch, ["ask", "Благодаря ви", "Positive?"]) == "multilingual"


def test_the_flag_still_wins_over_the_setting(monkeypatch):
    monkeypatch.setenv("VERDICT_LANG", "multi")
    assert asked_model(monkeypatch, ["ask", "thanks", "Positive?", "--lang", "en"]) is None


def test_without_the_setting_nothing_changes(monkeypatch):
    assert asked_model(monkeypatch, ["ask", "thanks", "Positive?", "--lang", "en"]) is None
