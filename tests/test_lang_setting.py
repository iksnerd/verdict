"""`[model].lang`: which checkpoint reads the state, set once instead of `--lang` on every call.

For Cyrillic (and other non-English) text the English checkpoint is confidently wrong: a glowing
Bulgarian review read "negative 0.72" there and "positive 1.00" on the multilingual one. A script
that forgot `--lang multi` got that wrong answer with only a warning."""
from __future__ import annotations

import argparse

import pytest

from verdict import cli, client, config
from conftest import typed


@pytest.fixture
def no_file(monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATHS", ())
    for var in ("VERDICT_LANG", "VERDICT_URL", "VERDICT_MODEL", "VERDICT_BITS"):
        monkeypatch.delenv(var, raising=False)


def test_lang_defaults_to_auto(no_file):
    assert config.load().lang == "auto"


def test_lang_from_the_environment(no_file, monkeypatch):
    monkeypatch.setenv("VERDICT_LANG", "multi")
    assert config.load().lang == "multi"


def test_lang_from_the_file(monkeypatch, tmp_path, no_file):
    f = tmp_path / "verdict.toml"
    f.write_text('[model]\nlang = "multi"\n')
    monkeypatch.setattr(config, "CONFIG_PATHS", (f,))
    assert config.load().lang == "multi"


def test_a_bad_lang_is_a_clean_error(no_file, monkeypatch, capsys):
    monkeypatch.setenv("VERDICT_LANG", "bulgarian")
    assert cli.main(["ask", "x", "q?", "--server-only"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("verdict: ") and "auto, en or multi" in err


def test_init_writes_the_lang_line(no_file):
    assert 'lang = "auto"' in config.to_toml(config.load())


def asked_model(monkeypatch, argv):
    seen = {}

    def fake_decide(state, questions, url=None, model=None, **flags):
        seen["model"] = model
        return {"model": "fake", "answers": typed(
            questions, lambda k: {"type": "noul", "noul": 0.5, "confidence": 0.5})}
    monkeypatch.setattr(client, "decide", fake_decide)
    assert cli.main(argv) == 0
    return seen["model"]


def test_the_setting_sends_calls_to_the_multilingual_checkpoint(no_file, monkeypatch):
    monkeypatch.setenv("VERDICT_LANG", "multi")
    assert asked_model(monkeypatch, ["ask", "Благодаря ви", "Positive?"]) == "multilingual"


def test_the_flag_still_wins_over_the_setting(no_file, monkeypatch):
    monkeypatch.setenv("VERDICT_LANG", "multi")
    assert asked_model(monkeypatch, ["ask", "thanks", "Positive?", "--lang", "en"]) is None


def test_without_the_setting_nothing_changes(no_file, monkeypatch):
    assert asked_model(monkeypatch, ["ask", "thanks", "Positive?", "--lang", "en"]) is None
