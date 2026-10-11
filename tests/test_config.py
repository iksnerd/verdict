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


# [systemone]: a hosted or remote backend for `verdict serve`. The key is never a setting.

def _isolated(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for var in ("VERDICT_URL", "VERDICT_SYSTEMONE_URL", "VERDICT_SYSTEMONE_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config, "CONFIG_PATHS", (pathlib.Path("verdict.toml"),))


def test_no_systemone_backend_by_default(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    s = config.load()
    assert s.systemone_url is None and s.systemone_model == "jev-latest"


def test_a_file_can_name_a_systemone_backend(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    write(tmp_path, MINIMAL + '\n[systemone]\nurl = "https://api.typesafe.ai/"\nmodel = "jev-1.13.0"\n')
    s = config.load()
    assert s.systemone_url == "https://api.typesafe.ai" and s.systemone_model == "jev-1.13.0"


def test_the_environment_beats_the_file_for_systemone(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    write(tmp_path, MINIMAL + '\n[systemone]\nurl = "https://api.typesafe.ai"\n')
    monkeypatch.setenv("VERDICT_SYSTEMONE_URL", "http://127.0.0.1:8009")
    monkeypatch.setenv("VERDICT_SYSTEMONE_MODEL", "kev-latest")
    s = config.load()
    assert s.systemone_url == "http://127.0.0.1:8009" and s.systemone_model == "kev-latest"


def test_a_systemone_url_without_a_scheme_is_named_not_a_traceback(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    write(tmp_path, MINIMAL + '\n[systemone]\nurl = "api.typesafe.ai"\n')
    with pytest.raises(config.ConfigError, match=r"\[systemone\]\.url"):
        config.load()


def test_a_key_in_the_file_is_refused_not_ignored(monkeypatch, tmp_path):
    """A secret in a file that `verdict init` writes and people commit."""
    _isolated(monkeypatch, tmp_path)
    write(tmp_path, MINIMAL + '\n[systemone]\nurl = "https://api.typesafe.ai"\nkey = "sk-x"\n')
    with pytest.raises(config.ConfigError, match="key"):
        config.load()


def test_systemone_round_trips_through_the_written_file(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    original = config.Settings(
        url="http://127.0.0.1:9100", model_path="m", prompt_token_budget=96,
        systemone_url="https://api.typesafe.ai", systemone_model="jev-1.13.0")
    (tmp_path / "verdict.toml").write_text(config.to_toml(original))
    back = config.load()
    assert back.systemone_url == "https://api.typesafe.ai" and back.systemone_model == "jev-1.13.0"


def test_an_unset_systemone_is_written_as_a_comment_and_reads_back_unset(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    text = config.to_toml(config.Settings("http://127.0.0.1:9100", "m", 96))
    assert "TYPESAFE_API_KEY" in text and "\n[systemone]" not in text
    (tmp_path / "verdict.toml").write_text(text)
    assert config.load().systemone_url is None


def test_config_explain_says_where_systemone_came_from(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("VERDICT_SYSTEMONE_URL", "http://127.0.0.1:8009")
    rows = {r["key"]: r["source"] for r in config.explain()}
    assert rows["systemone.url"] == "$VERDICT_SYSTEMONE_URL"
    assert rows["systemone.model"] == "default"


def test_the_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-live")
    assert config.systemone_key("https://api.typesafe.ai") == "sk-live"


def test_a_hosted_url_without_a_key_names_what_is_sent_and_billed(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(config.ConfigError, match="TYPESAFE_API_KEY") as info:
        config.systemone_key("https://api.typesafe.ai")
    assert "billed" in str(info.value)


def test_a_loopback_server_needs_no_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert config.systemone_key("http://127.0.0.1:8009") == "local"


# ---- the key must not travel in cleartext, and errors name where the value came from ----------

def test_a_remote_systemone_server_must_be_https(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    write(tmp_path, MINIMAL + '\n[systemone]\nurl = "http://api.typesafe.ai"\n')
    with pytest.raises(config.ConfigError, match="https") as info:
        config.load()
    assert "TYPESAFE_API_KEY" in str(info.value) and "[systemone].url" in str(info.value)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8009", "http://localhost:8009", "https://api.typesafe.ai"])
def test_loopback_may_be_http_and_a_remote_may_be_https(monkeypatch, tmp_path, url):
    _isolated(monkeypatch, tmp_path)
    write(tmp_path, MINIMAL + f'\n[systemone]\nurl = "{url}"\n')
    assert config.load().systemone_url == url


def test_the_flag_path_names_the_flag(monkeypatch):
    with pytest.raises(config.ConfigError) as info:
        config.check_systemone_url("ftp://x", "--systemone")
    assert "--systemone" in str(info.value) and "[systemone]" not in str(info.value)
    with pytest.raises(config.ConfigError, match="--systemone.*https"):
        config.check_systemone_url("http://api.typesafe.ai", "--systemone")


def test_the_environment_path_names_the_variable(monkeypatch, tmp_path):
    _isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("VERDICT_SYSTEMONE_URL", "ftp://x")
    with pytest.raises(config.ConfigError, match=r"\$VERDICT_SYSTEMONE_URL"):
        config.load()
