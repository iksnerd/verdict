"""No machine configuration reaches a test.

Settings too: with `lang = "multi"` in ~/.config/verdict/config.toml, four language tests failed
in the pre-commit export, and passed in the checkout only because its gitignored verdict.toml
shadowed the user file. No config file and no VERDICT_* variable reaches a test unless it sets one."""

from __future__ import annotations

import os

from pathlib import Path

import pytest

from fakes import FakeLayaAgent, install_laya, typed  # noqa: F401  (tests import `typed` from here)
from verdict import config


@pytest.fixture(autouse=True)
def _no_machine_config(monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATHS", ())
    for name in [n for n in os.environ if n.startswith("VERDICT_")]:
        monkeypatch.delenv(name)



@pytest.fixture(autouse=True)
def _no_real_server(monkeypatch):
    """A test never reaches a `verdict serve` that happens to be running on the developer's
    machine (the CLI asks a server what it fronts, at /v1/capabilities, before it answers). The
    network layer refuses unless the test installs its own transport."""
    import urllib.error

    from verdict import client

    def refuse(request, timeout=None):
        raise urllib.error.URLError("no server in tests")

    monkeypatch.setattr(client.urllib.request, "urlopen", refuse)


@pytest.fixture
def fake_laya(monkeypatch):
    """A stand-in `laya_mlx` module. Returns `{model_id: agent}` of every checkpoint the code
    under test loaded, so a test can assert which were (and were not) loaded and how often each
    was asked."""
    loaded = {}
    install_laya(monkeypatch, lambda model_id, **kw: loaded.setdefault(model_id, FakeLayaAgent(model_id)))
    return loaded


@pytest.fixture
def config_files(monkeypatch, tmp_path):
    """`use(*texts) -> [Path]`: writes each text as a TOML file in `tmp_path` and makes them
    `config.CONFIG_PATHS`, in order. The first is called `name` (default `verdict.toml`), the
    others `user1.toml`, `user2.toml`."""
    def use(*texts: str, name: str = "verdict.toml") -> list[Path]:
        paths = [tmp_path / (name if i == 0 else f"user{i}.toml") for i in range(len(texts))]
        for path, text in zip(paths, texts):
            path.write_text(text)
        monkeypatch.setattr(config, "CONFIG_PATHS", tuple(paths))
        return paths
    return use
