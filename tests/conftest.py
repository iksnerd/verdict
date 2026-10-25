"""No machine configuration reaches a test.

Settings too: with `lang = "multi"` in ~/.config/verdict/config.toml, four language tests failed
in the pre-commit export, and passed in the checkout only because its gitignored verdict.toml
shadowed the user file. No config file and no VERDICT_* variable reaches a test unless it sets one."""

from __future__ import annotations

import os
import sys
import types

import pytest

from fakes import FakeLayaAgent, typed  # noqa: F401  (tests import `typed` from here)
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
    module = types.ModuleType("laya_mlx")
    module.load = lambda model_id, **kw: loaded.setdefault(model_id, FakeLayaAgent(model_id))
    monkeypatch.setitem(sys.modules, "laya_mlx", module)
    return loaded
