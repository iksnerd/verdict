"""Audit 4: config memoised per state, per-entry provenance for merged tables, explicit path errors."""
from __future__ import annotations

import os

import pytest

from verdict import config


@pytest.fixture
def one_file(config_files):
    return config_files('[model]\nbits = 8\n')[0]


def _count_reads(monkeypatch):
    calls = []
    real = config._read
    monkeypatch.setattr(config, "_read", lambda p: calls.append(p) or real(p))
    return calls


def test_an_unchanged_state_is_parsed_once(monkeypatch, one_file):
    calls = _count_reads(monkeypatch)
    a, b = config.load(), config.load()
    assert len(calls) == 1 and a.bits == b.bits == 8


def test_an_env_change_is_picked_up(monkeypatch, one_file):
    monkeypatch.setenv("VERDICT_URL", "http://127.0.0.1:9001")
    assert config.load().url == "http://127.0.0.1:9001"
    monkeypatch.setenv("VERDICT_URL", "http://127.0.0.1:9002")
    assert config.load().url == "http://127.0.0.1:9002"


def test_an_edit_to_the_file_reparses(monkeypatch, one_file):
    calls = _count_reads(monkeypatch)
    config.load()
    one_file.write_text('[model]\nbits = 16\n')
    st = one_file.stat()
    os.utime(one_file, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))
    assert config.load().bits == 16 and len(calls) == 2


def test_a_returned_settings_cannot_poison_the_next(monkeypatch, one_file):
    config.load().sources[("x", "y")] = "junk"
    assert ("x", "y") not in config.load().sources


def test_the_routes_warning_prints_once_per_file_state(monkeypatch, tmp_path, capsys):
    f = tmp_path / "r.toml"
    f.write_text('[routes]\na = 1\n')
    for _ in range(3):
        config._read(f)
    assert capsys.readouterr().err.count("[routes]") == 1


def test_merged_extra_tables_record_each_entrys_file(config_files):
    local, user = config_files('[model.extra]\nb = "org/b"\n', '[model.extra]\na = "org/a"\n')
    s = config.load()
    assert s.sources[("model", "extra", "a")] == str(user)
    assert s.sources[("model", "extra", "b")] == str(local)
    row = {r["key"]: r["source"] for r in config.explain()}["model.extra"]
    assert str(user) in row and str(local) in row


def test_an_explicit_path_that_is_not_a_file_is_an_error_naming_it(tmp_path):
    with pytest.raises(config.ConfigError, match="missing.toml"):
        config.load(tmp_path / "missing.toml")


def test_no_file_and_no_path_still_yields_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CONFIG_PATHS", (tmp_path / "none.toml",))
    assert config.load().source is None
