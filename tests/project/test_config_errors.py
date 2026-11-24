"""A bad setting is a one-line `verdict:` error and exit 2, never a traceback.

Found sweeping failure modes after 0.8.0: a non-number token budget, a TOML syntax error, a URL
with no port or no scheme each ended in a Python traceback (exit 1), the kind of output an agent
cannot act on and a person should not have to read."""
from __future__ import annotations

import pytest

from verdict import cli, config


def with_file(monkeypatch, config_files, text):
    f, = config_files(text)
    for var in ("VERDICT_URL", "VERDICT_MODEL", "VERDICT_BITS", "VERDICT_MULTILINGUAL"):
        monkeypatch.delenv(var, raising=False)
    return f


@pytest.mark.parametrize("text,needle", [
    ('[model]\nprompt_token_budget = "lots"\n', "prompt_token_budget"),
    ('[model]\nprompt_token_budget = -5\n', "prompt_token_budget"),
    ('[model\nbroken', "verdict.toml"),
    ('[server]\nurl = "http://127.0.0.1"\n', "port"),
    ('[server]\nurl = "notaurl"\n', "http"),
])
def test_a_bad_file_setting_is_a_clean_error(monkeypatch, config_files, capsys, text, needle):
    with_file(monkeypatch, config_files, text)
    assert cli.main(["ask", "x", "q?", "--lang", "en", "--server-only"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("verdict: ") and needle in err and "Traceback" not in err


def test_a_bad_url_in_the_environment_is_a_clean_error(monkeypatch, config_files, capsys):
    with_file(monkeypatch, config_files, "")
    monkeypatch.setenv("VERDICT_URL", "notaurl")
    assert cli.main(["ask", "x", "q?", "--lang", "en", "--server-only"]) == 2
    assert "VERDICT_URL" in capsys.readouterr().err


def test_serve_with_a_portless_url_is_a_clean_error(monkeypatch, config_files, capsys):
    with_file(monkeypatch, config_files, '[server]\nurl = "http://127.0.0.1"\n')
    assert cli.main(["serve"]) == 2
    assert "port" in capsys.readouterr().err


def test_a_good_file_still_loads(monkeypatch, config_files):
    with_file(monkeypatch, config_files, '[server]\nurl = "http://127.0.0.1:8800/"\n'
                                     '[model]\nprompt_token_budget = 0\nbits = 8\n')
    s = config.load()
    assert (s.url, s.port, s.prompt_token_budget, s.bits) == ("http://127.0.0.1:8800", 8800, 0, 8)


@pytest.mark.parametrize("argv", [
    ["ask", "x", "q?", "--lang", "en", "--server-only", "--url", "notaurl"],
    ["route", "x", "--url", "notaurl"],
])
def test_a_bad_url_flag_is_a_clean_error(monkeypatch, config_files, capsys, argv):
    with_file(monkeypatch, config_files, "")
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert "--url" in err and "Traceback" not in err
