"""Fixtures shared by the CLI tests.

Plain helpers are fixtures that return the function, not names to import. Pytest loads this file
under the module name `conftest`, which hides `tests/conftest.py` from `from conftest import typed`,
so `typed` is re-exported below."""
from __future__ import annotations

import pytest

from fakes import FakeEngine, typed  # noqa: F401  (tests import `typed` from `conftest`)
from verdict import config
from verdict.cli import inference, setup, support, update

#: One yes/no bank, as the JSON a `--questions` flag takes.
MONEY_BANK = '{"q": {"type": "noul", "instructions": "Is `message` about money?"}}'


class ToolShell:
    """Stands in for `update._run` in a `uv tool` install: no checkout, release tags from git."""

    def __init__(self, tags="v0.1.0\nv0.2.1\nv0.10.0\nv0.3.0"):
        self.calls, self.tags = [], tags

    def __call__(self, argv, cwd=None):
        self.calls.append(argv)
        if argv[:2] == ["git", "ls-remote"]:
            return 0, "\n".join(f"abc{i}\trefs/tags/{t}" for i, t in enumerate(self.tags.split()))
        return 0, ""


@pytest.fixture
def no_release_lookup(monkeypatch):
    """Nothing asks GitHub which tags were released, unless a test says what it answers."""
    monkeypatch.setattr(update, "_released", lambda url: None)


@pytest.fixture
def tool(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "_checkout", lambda: tmp_path)  # no .git: a tool install
    monkeypatch.setattr(update, "_server_up", lambda: False)
    monkeypatch.setattr(support, "_version", lambda: "verdict 0.2.1")
    monkeypatch.setattr(update, "_has_jev", lambda: False)  # whatever this venv happens to hold
    fake = ToolShell()
    monkeypatch.setattr(update, "_run", fake)
    return fake


@pytest.fixture
def scripted_installs():
    def scripted(tool, *results):
        """Make successive `uv tool install` calls return these (code, output) pairs."""
        queue = list(results)
        plain = tool.__call__

        def call(argv, cwd=None):
            if argv[:3] == ["uv", "tool", "install"]:
                tool.calls.append(argv)
                return queue.pop(0)
            return plain(argv, cwd)
        return call
    return scripted


@pytest.fixture
def restores():
    def restored(tool):
        """Installs of the OLD version: the rollback after a failed update."""
        return [c for c in tool.calls if c[:3] == ["uv", "tool", "install"] and c[-1].endswith("@v0.2.1")]
    return restored


@pytest.fixture
def local_engine(monkeypatch):
    """`install(engine=None)`: the CLI's in-process engine is `engine` (a `FakeEngine` named
    `local` when none is given), with no engine cached from an earlier test. Returns the engine."""
    def install(engine=None):
        engine = FakeEngine(name="local") if engine is None else engine
        monkeypatch.setattr(inference, "load", lambda *a, **k: engine)
        monkeypatch.setattr(inference, "_ENGINES", {})
        return engine
    return install


@pytest.fixture
def init_env(monkeypatch, tmp_path):
    """`init(free=...)`: what `verdict init` needs to run in `tmp_path` without touching the
    machine: no config file to read, the working directory in `tmp_path`, and `free(host, port)`
    deciding which ports are free (all of them by default)."""
    def init(free=lambda *a: True):
        monkeypatch.setattr(config, "CONFIG_PATHS", (tmp_path / "absent.toml",))
        monkeypatch.setattr(setup, "_port_free", free)
        monkeypatch.chdir(tmp_path)
    return init


@pytest.fixture
def no_inference(monkeypatch):
    """Nothing here may load a model; a test that needs a fake overrides `load` itself."""
    monkeypatch.setattr(inference, "_ENGINES", {})
    monkeypatch.setattr(inference, "load", lambda *a, **k: pytest.fail("loaded a model"))
