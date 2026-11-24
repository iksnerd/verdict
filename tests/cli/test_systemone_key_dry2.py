"""`bench --systemone` and `examples --systemone` read the key through config, so a loopback host
in any spelling config accepts (`[::1]` too) needs no $TYPESAFE_API_KEY."""
from __future__ import annotations

import pytest

from verdict import cli, config
from verdict.evaluation import bench


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)


def test_key_or_none_matches_systemone_key(no_key, monkeypatch):
    assert config.systemone_key_or_none("http://[::1]:8000") == "local"
    assert config.systemone_key_or_none("https://api.example") is None
    with pytest.raises(config.ConfigError):
        config.systemone_key("https://api.example")
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    assert config.systemone_key_or_none("https://api.example") == "k"


@pytest.mark.parametrize("cmd", [["bench"], ["examples", "citation-check"]])
def test_ipv6_loopback_needs_no_key(no_key, monkeypatch, capsys, cmd):
    seen = []

    def fake_asker(url, model, key):
        seen.append(key)
        raise ValueError("stop here")

    monkeypatch.setattr(bench, "systemone_asker", fake_asker)
    cli.main([*cmd, "--systemone", "http://[::1]:8000"])
    assert seen == ["local"]
    assert "TYPESAFE_API_KEY" not in capsys.readouterr().err
