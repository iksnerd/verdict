"""Audit 2: provenance is recorded once, by `load`, and `explain` reads it."""
from __future__ import annotations

import pytest

from verdict import config


def _two_files(monkeypatch, tmp_path):
    local, user = tmp_path / "local.toml", tmp_path / "user.toml"
    local.write_text('[model]\nbits = 8\n')
    user.write_text('[model]\nlang = "multi"\n[server]\nurl = "http://127.0.0.1:9001"\n')
    monkeypatch.setattr(config, "CONFIG_PATHS", (local, user))
    return local, user


def test_each_key_names_the_file_that_won(monkeypatch, tmp_path):
    local, user = _two_files(monkeypatch, tmp_path)
    s = config.load()
    assert s.sources[("model", "bits")] == str(local)
    assert s.sources[("model", "lang")] == str(user)
    assert ("model", "path") not in s.sources
    rows = {r["key"]: r["source"] for r in config.explain()}
    assert rows["model.bits"] == str(local) and rows["model.lang"] == str(user)
    assert rows["model.path"] == "default"


def test_the_environment_wins_in_both_load_and_explain(monkeypatch, tmp_path):
    _two_files(monkeypatch, tmp_path)
    monkeypatch.setenv("VERDICT_URL", "http://127.0.0.1:9777")
    assert config.load().sources[("server", "url")] == "$VERDICT_URL"
    assert {r["key"]: r["source"] for r in config.explain()}["server.url"] == "$VERDICT_URL"


def test_a_forced_path_that_is_not_a_file_is_not_the_source(tmp_path):
    s = config.load(tmp_path / "missing.toml")
    assert s.source is None and s.sources == {}


def test_a_bad_url_is_named_by_where_it_came_from_not_by_a_guess(monkeypatch, tmp_path):
    f = tmp_path / "v.toml"
    f.write_text('[server]\nurl = "nonsense"\n')
    monkeypatch.setattr(config, "CONFIG_PATHS", (f,))
    monkeypatch.setenv("VERDICT_URL", "nonsense")   # same text as the file's: the old guess said $VAR
    with pytest.raises(config.ConfigError, match=r"^\$VERDICT_URL"):
        config.load()
    monkeypatch.delenv("VERDICT_URL")
    with pytest.raises(config.ConfigError, match=r"^\[server\]\.url"):
        config.load()


def test_a_relative_directory_that_exists_is_local(monkeypatch, tmp_path):
    (tmp_path / "foo" / "bar").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert config._is_local("foo/bar")
    assert config.resolve_model("foo/bar") == ("foo/bar", None)
    assert not config._is_local("owner/not-here")
