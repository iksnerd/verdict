"""`verdict update`: pull the checkout the CLI runs from and re-sync with the extras.

Nothing real runs: `update._run` is replaced, so no git, no uv, no network."""
from __future__ import annotations

from pathlib import Path

import pytest

from verdict import cli
from verdict.cli import support, update


class FakeShell:
    def __init__(self, dirty="", behind="0", head=("aaa1111", "bbb2222")):
        self.calls: list[list[str]] = []
        self.dirty, self.behind = dirty, behind
        self.heads = list(head)

    def __call__(self, argv, cwd=None):
        self.calls.append(argv)
        joined = " ".join(argv)
        if "status --porcelain" in joined:
            return 0, self.dirty
        if "rev-list --count" in joined:
            return 0, self.behind
        if "rev-parse --short HEAD" in joined:
            return 0, self.heads.pop(0) if len(self.heads) > 1 else self.heads[0]
        if "log --oneline" in joined:
            return 0, "bbb2222 feat: something new"
        return 0, ""


@pytest.fixture
def shell(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "verdict"\nversion = "0.2.0"\n')
    monkeypatch.setattr(update, "_checkout", lambda: tmp_path)
    monkeypatch.setattr(update, "_server_up", lambda: False)
    monkeypatch.setattr(update, "_has_jev", lambda: False)
    fake = FakeShell(behind="3")
    monkeypatch.setattr(update, "_run", fake)
    return fake


def ran(fake, needle):
    return any(needle in " ".join(c) for c in fake.calls)


def test_checkout_finds_the_real_repo_root():
    """The real `_checkout`, not the mocked one every other test in this file uses: it must
    resolve to the checkout root regardless of how many directories deep `update.py` itself
    lives, since that depth is an implementation detail of the cli package layout."""
    import verdict

    expected_root = Path(verdict.__file__).resolve().parents[2]
    assert update._checkout() == expected_root
    assert (update._checkout() / "pyproject.toml").is_file()


def test_update_pulls_fast_forward_only_then_syncs_with_both_extras(shell, capsys):
    assert cli.main(["update"]) == 0
    pull = next(i for i, c in enumerate(shell.calls) if c[:2] == ["git", "pull"])
    sync = next(i for i, c in enumerate(shell.calls) if c[:2] == ["uv", "sync"])
    assert "--ff-only" in shell.calls[pull]
    assert pull < sync
    assert {"--extra", "mlx", "laya"} <= set(shell.calls[sync]), "plain uv sync drops the extras"
    out = capsys.readouterr().out
    assert "0.2.0" in out and "aaa1111" in out and "bbb2222" in out


def test_check_reports_and_changes_nothing(shell, capsys):
    """Exit 1 when behind, so a script can branch on it."""
    assert cli.main(["update", "--check"]) == 1
    assert ran(shell, "fetch") and not ran(shell, "pull") and not ran(shell, "uv sync")
    assert "3" in capsys.readouterr().out


def test_up_to_date_does_not_pull_or_sync(shell, capsys):
    shell.behind = "0"
    assert cli.main(["update"]) == 0
    assert not ran(shell, "pull") and not ran(shell, "uv sync")
    assert "up to date" in capsys.readouterr().out


def test_uncommitted_changes_refuse(shell, capsys):
    shell.dirty = " M src/verdict/cli.py"
    assert cli.main(["update"]) == 2
    assert not ran(shell, "pull")
    assert "uncommitted" in capsys.readouterr().err


def test_a_running_server_is_told_to_restart(shell, monkeypatch, capsys):
    monkeypatch.setattr(update, "_server_up", lambda: True)
    cli.main(["update"])
    assert "restart" in capsys.readouterr().out.lower()


def test_a_tool_install_with_no_reachable_releases_is_an_error(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(update, "_checkout", lambda: tmp_path)  # no .git: a tool install
    monkeypatch.setattr(update, "_run", FakeShell())  # ls-remote lists nothing
    assert cli.main(["update"]) == 2
    assert "release" in capsys.readouterr().err


def test_a_failing_step_stops_and_reports(shell, monkeypatch, capsys):
    def failing(argv, cwd=None):
        shell.calls.append(argv)
        if argv[:2] == ["git", "pull"]:
            return 1, "fatal: Not possible to fast-forward"
        return FakeShell(behind="3")(argv, cwd)

    monkeypatch.setattr(update, "_run", failing)
    assert cli.main(["update"]) == 2
    assert not ran(shell, "uv sync")
    assert "fast-forward" in capsys.readouterr().err


# --- installed as a uv tool: no checkout, so update follows release tags ---------------------

class ToolShell:
    def __init__(self, tags="v0.1.0\nv0.2.1\nv0.10.0\nv0.3.0"):
        self.calls, self.tags = [], tags

    def __call__(self, argv, cwd=None):
        self.calls.append(argv)
        if argv[:2] == ["git", "ls-remote"]:
            return 0, "\n".join(f"abc{i}\trefs/tags/{t}" for i, t in enumerate(self.tags.split()))
        return 0, ""


@pytest.fixture
def tool(monkeypatch, tmp_path):
    monkeypatch.setattr(update, "_checkout", lambda: tmp_path)  # no .git: a tool install
    monkeypatch.setattr(update, "_server_up", lambda: False)
    monkeypatch.setattr(support, "_version", lambda: "verdict 0.2.1")
    monkeypatch.setattr(update, "_has_jev", lambda: False)  # whatever this venv happens to hold
    fake = ToolShell()
    monkeypatch.setattr(update, "_run", fake)
    return fake


def test_tool_install_reinstalls_the_newest_release_tag_with_extras(tool, capsys):
    assert cli.main(["update"]) == 0
    install = next(c for c in tool.calls if c[:3] == ["uv", "tool", "install"])
    spec = install[-1]
    assert "--force" in install
    assert spec.startswith("verdict[mlx,laya] @ git+")
    assert spec.endswith("@v0.10.0"), "versions sort numerically, not as strings"
    assert "0.2.1 -> 0.10.0" in capsys.readouterr().out


def test_tool_install_check_reports_the_newer_release_and_installs_nothing(tool, capsys):
    assert cli.main(["update", "--check"]) == 1
    assert not any(c[:3] == ["uv", "tool", "install"] for c in tool.calls)
    assert "0.10.0" in capsys.readouterr().out


def test_tool_install_already_on_the_newest_release(tool, capsys):
    tool.tags = "v0.1.0 v0.2.1"
    assert cli.main(["update"]) == 0
    assert not any(c[:3] == ["uv", "tool", "install"] for c in tool.calls)
    assert "up to date" in capsys.readouterr().out


def test_tool_install_ignores_tags_that_are_not_versions(tool, capsys):
    tool.tags = "v0.2.1 nightly v0.3.0-rc1 latest"
    assert cli.main(["update", "--check"]) == 0
    assert "up to date" in capsys.readouterr().out


# --- a uv cache that lags a just-pushed tag -----------------------------------------------------

STALE = "fatal: Could not parse object '8d3d7e48e0f79bc66ac74c0eee8df387fafa6f09'."


def scripted_installs(tool, *results):
    """Make successive `uv tool install` calls return these (code, output) pairs."""
    queue = list(results)
    plain = tool.__call__

    def call(argv, cwd=None):
        if argv[:3] == ["uv", "tool", "install"]:
            tool.calls.append(argv)
            return queue.pop(0)
        return plain(argv, cwd)
    return call


def installs(tool):
    return [c for c in tool.calls if c[:3] == ["uv", "tool", "install"]]


def test_a_stale_uv_git_cache_is_refreshed_and_retried_once(tool, monkeypatch, capsys):
    """Right after the v0.8.0 tag was pushed, uv resolved the tag but its cached clone lacked the
    commit; an immediate retry worked. `--refresh-package` makes the retry fetch."""
    monkeypatch.setattr(update, "_run", scripted_installs(tool, (1, STALE), (0, "")))
    assert cli.main(["update"]) == 0
    first, second = installs(tool)
    assert "--refresh-package" not in first
    assert second[second.index("--refresh-package") + 1] == "verdict"
    assert "0.2.1 -> 0.10.0" in capsys.readouterr().out


def test_any_other_install_failure_is_not_retried(tool, monkeypatch, capsys):
    monkeypatch.setattr(update, "_run", scripted_installs(tool, (1, "error: no such tag")))
    assert cli.main(["update"]) == 2
    assert len(installs(tool)) == 1
    assert "no such tag" in capsys.readouterr().err


def test_a_stale_cache_twice_fails_instead_of_looping(tool, monkeypatch, capsys):
    monkeypatch.setattr(update, "_run", scripted_installs(tool, (1, STALE), (1, STALE)))
    assert cli.main(["update"]) == 2
    assert len(installs(tool)) == 2


def test_check_exits_0_when_up_to_date(shell):
    shell.behind = "0"
    assert cli.main(["update", "--check"]) == 0


def test_update_keeps_the_jev_extra_when_it_is_installed(shell, monkeypatch):
    """Like mlx and laya: a sync without it removes TypeSafe's SDK and breaks hosted Jev."""
    monkeypatch.setattr(update, "_has_jev", lambda: True)
    assert cli.main(["update"]) == 0
    sync = next(c for c in shell.calls if c[:2] == ["uv", "sync"])
    assert {"mlx", "laya", "jev"} <= set(sync)


def test_update_does_not_add_the_jev_extra_to_a_machine_without_it(shell, monkeypatch):
    monkeypatch.setattr(update, "_has_jev", lambda: False)
    assert cli.main(["update"]) == 0
    sync = next(c for c in shell.calls if c[:2] == ["uv", "sync"])
    assert "jev" not in sync


def test_a_tool_update_keeps_the_jev_extra_too(monkeypatch):
    calls = []
    monkeypatch.setattr(update, "_has_jev", lambda: True)
    monkeypatch.setattr(update, "_release_tags", lambda url: [(9, 9, 9)])
    monkeypatch.setattr(update, "_run", lambda argv, cwd=None: (calls.append(argv), (0, ""))[1])
    monkeypatch.setattr(update, "_server_up", lambda: False)
    monkeypatch.setattr(support, "_version", lambda: "verdict 0.1.0")
    assert update._update_tool(type("A", (), {"check": False})()) == 0
    assert any("verdict[mlx,laya,jev]" in " ".join(c) for c in calls)
