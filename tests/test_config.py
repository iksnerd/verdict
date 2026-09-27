"""Precedence is the whole point of this module, so it is tested rather than described."""
from __future__ import annotations

import pathlib

import pytest

from verdict import config

MINIMAL = """
[server]
url = "http://127.0.0.1:9001"
[model]
path = "models/other-mlx"
prompt_token_budget = 64
multilingual = "models/multi-mlx"
"""


def write(tmp_path, body=MINIMAL):
    p = tmp_path / "verdict.toml"
    p.write_text(body)
    return p


def test_defaults_apply_with_no_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDICT_URL", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATHS", (pathlib.Path("verdict.toml"),))
    s = config.load()
    assert s.source is None
    assert s.url == config.DEFAULTS["server"]["url"]
    assert set(s.defaulted) == set(config.DEFAULTS)


def test_a_file_beats_the_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDICT_URL", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATHS", (pathlib.Path("verdict.toml"),))
    write(tmp_path)
    s = config.load()
    assert s.url == "http://127.0.0.1:9001" and s.port == 9001
    assert s.model_path == "models/other-mlx" and s.prompt_token_budget == 64
    assert s.multilingual_path == "models/multi-mlx"
    assert s.source is not None


def test_the_environment_beats_the_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATHS", (pathlib.Path("verdict.toml"),))
    write(tmp_path)
    monkeypatch.setenv("VERDICT_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("VERDICT_MULTILINGUAL", "models/fromenv-mlx")
    s = config.load()
    assert s.url == "http://127.0.0.1:9999"
    assert s.multilingual_path == "models/fromenv-mlx"
    assert s.model_path == "models/other-mlx", "the file should still win where env is silent"


def test_a_partial_file_falls_back_per_section(monkeypatch, tmp_path):
    """Setting one table must not wipe the others."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDICT_URL", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATHS", (pathlib.Path("verdict.toml"),))
    write(tmp_path, '[server]\nurl = "http://127.0.0.1:9002"\n')
    s = config.load()
    assert s.url == "http://127.0.0.1:9002"
    assert s.model_path == config.DEFAULTS["model"]["path"]
    assert "model" in s.defaulted and "server" not in s.defaulted


def test_a_trailing_slash_is_stripped(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDICT_URL", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATHS", (pathlib.Path("verdict.toml"),))
    write(tmp_path, '[server]\nurl = "http://127.0.0.1:9003/"\n')
    assert config.load().url == "http://127.0.0.1:9003"


def test_round_trip_through_the_written_file(monkeypatch, tmp_path):
    """`verdict init` writes with to_toml; load() must read back exactly what it wrote."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDICT_URL", raising=False)
    monkeypatch.delenv("VERDICT_MULTILINGUAL", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATHS", (pathlib.Path("verdict.toml"),))
    original = config.Settings(
        url="http://127.0.0.1:9100", model_path="models/x-mlx", prompt_token_budget=96,
        multilingual_path="models/m-mlx",
    )
    (tmp_path / "verdict.toml").write_text(config.to_toml(original))
    back = config.load()
    for attr in ("url", "model_path", "prompt_token_budget", "multilingual_path"):
        assert getattr(back, attr) == getattr(original, attr), attr


def test_there_is_no_thresholds_key():
    """A key that is read and ignored is worse than no key: the cuts are applied inside the
    server process, so a [thresholds] table could not affect the server path."""
    assert "thresholds" not in config.DEFAULTS


def test_extra_checkpoints_default_to_empty(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATHS", (pathlib.Path("verdict.toml"),))
    assert config.load().extra_checkpoints == {}


def test_a_file_can_name_extra_checkpoints(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATHS", (pathlib.Path("verdict.toml"),))
    write(tmp_path, MINIMAL + '\n[model.extra]\nsupport = "org/support-mlx"\nlegal = "org/legal-mlx"\n')
    s = config.load()
    assert s.extra_checkpoints == {"support": "org/support-mlx", "legal": "org/legal-mlx"}


def test_extra_checkpoints_round_trip_through_the_written_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDICT_URL", raising=False)
    monkeypatch.delenv("VERDICT_MULTILINGUAL", raising=False)
    monkeypatch.setattr(config, "CONFIG_PATHS", (pathlib.Path("verdict.toml"),))
    original = config.Settings(
        url="http://127.0.0.1:9100", model_path="models/x-mlx", prompt_token_budget=96,
        multilingual_path="models/m-mlx", extra_checkpoints={"support": "org/support-mlx"},
    )
    (tmp_path / "verdict.toml").write_text(config.to_toml(original))
    back = config.load()
    assert back.extra_checkpoints == {"support": "org/support-mlx"}
    assert not hasattr(config.Settings("u", "m", 1, "s", "b"), "t_difficulty")


def test_settings_is_not_a_dataclass():
    """Same reason as switch.Branch: `dataclasses` pulls `inspect`, 3.7 ms of the CLI budget."""
    import dataclasses

    assert not dataclasses.is_dataclass(config.Settings)


def test_a_routes_table_is_reported_not_silently_ignored(monkeypatch, tmp_path, capsys):
    """verdict no longer dispatches anywhere, so `[routes]` does nothing. Say so, once."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATHS", (pathlib.Path("verdict.toml"),))
    write(tmp_path, '[routes.small]\ncmd = "ollama run x"\n')
    config.load()
    err = capsys.readouterr().err
    assert "[routes]" in err and "no longer" in err


def test_nothing_dispatches():
    assert not hasattr(config.Settings, "small_cmd")
    assert "routes" not in config.DEFAULTS


def test_awkward_names_and_paths_round_trip_through_the_written_file(tmp_path):
    """A dotted name is a nested table in TOML unless quoted, and a quote or backslash in a
    path ends the string early unless escaped."""
    original = config.Settings(
        url="http://127.0.0.1:9100", model_path='C:\\models\\a "b"-mlx', prompt_token_budget=96,
        multilingual_path="models/m-mlx",
        extra_checkpoints={"support.v2": "org/support-v2", "a b": "x\\y\"z"},
    )
    path = tmp_path / "verdict.toml"
    path.write_text(config.to_toml(original))
    back = config.load(path)
    assert back.extra_checkpoints == original.extra_checkpoints
    assert back.model_path == original.model_path
