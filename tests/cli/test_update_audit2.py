"""`verdict update`, second audit: interruption, the restore pin, a silent release filter, a stale
environment under --check, and a server probe that cannot fail a finished update."""
from __future__ import annotations

import sys

import pytest

from verdict import cli, config
from verdict.cli import support, update

pytestmark = pytest.mark.usefixtures("no_release_lookup")



def test_ctrl_c_during_the_install_restores_the_old_version(tool, monkeypatch, capsys, restores):
    plain = tool.__call__

    def call(argv, cwd=None):
        if argv[:3] == ["uv", "tool", "install"]:
            tool.calls.append(argv)
            if argv[-1].endswith("@v0.10.0"):
                raise KeyboardInterrupt
            return 0, ""
        return plain(argv, cwd)

    monkeypatch.setattr(update, "_run", call)
    assert cli.main(["update"]) == 130
    assert len(restores(tool)) == 1
    assert "restored v0.2.1" in capsys.readouterr().err


def test_restore_never_guesses_a_pin_for_an_untagged_version(monkeypatch, tool, capsys, scripted_installs):
    monkeypatch.setattr(support, "_version", lambda: "verdict (not installed)")
    monkeypatch.setattr(update, "_run", scripted_installs(tool, (1, "error: boom")))
    assert cli.main(["update"]) == 2
    err = capsys.readouterr().err
    assert "@vinstalled)" not in err and "@v(" not in err
    assert len([c for c in tool.calls if c[:3] == ["uv", "tool", "install"]]) == 1, "no restore tried"
    assert "uv tool install" in err and "reinstall" in err.lower()


def test_restore_uses_the_extras_computed_before_the_removal(tool, monkeypatch, capsys, restores):
    state = {"jev": True}
    monkeypatch.setattr(update, "_has_jev", lambda: state["jev"])
    plain = tool.__call__

    def call(argv, cwd=None):
        if argv[:3] == ["uv", "tool", "install"]:
            tool.calls.append(argv)
            if argv[-1].endswith("@v0.10.0"):
                state["jev"] = False  # the old venv is gone, and the SDK with it
                return 1, "error: boom"
            return 0, ""
        return plain(argv, cwd)

    monkeypatch.setattr(update, "_run", call)
    assert cli.main(["update"]) == 2
    assert restores(tool)[0][-1].startswith("verdict[mlx,laya,jev] @ git+")


def test_a_failed_release_lookup_on_github_says_so_on_stderr(monkeypatch, capsys):
    import urllib.request

    monkeypatch.undo()

    def urlopen(request, timeout=None):
        raise OSError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert update._released("https://github.com/iksnerd/verdict.git") is None
    err = capsys.readouterr().err
    assert "release" in err and "offline" in err and err.count("\n") == 1


def test_a_non_github_remote_falls_back_silently(monkeypatch, capsys):
    monkeypatch.undo()
    assert update._released("https://gitlab.com/someone/verdict.git") is None
    assert capsys.readouterr().err == ""


def test_check_up_to_date_notes_that_the_environment_was_not_checked(monkeypatch, tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(update, "_checkout", lambda: tmp_path)
    monkeypatch.setattr(update, "_run", lambda argv, cwd=None: (0, "0" if "rev-list" in argv else ""))
    assert cli.main(["update", "--check"]) == 0
    out = capsys.readouterr().out
    assert "up to date" in out and "environment" in out


def test_a_config_error_after_the_install_does_not_fail_the_update(tool, monkeypatch, capsys):
    monkeypatch.undo()
    monkeypatch.setattr(update, "_released", lambda url: None)
    monkeypatch.setattr(update, "_checkout", lambda: __import__("pathlib").Path("/nonexistent"))
    monkeypatch.setattr(support, "_version", lambda: "verdict 0.2.1")
    monkeypatch.setattr(update, "_has_jev", lambda: False)
    monkeypatch.setattr(update, "_run", tool)

    def boom():
        raise config.ConfigError("bad verdict.toml")

    monkeypatch.setattr(support, "_settings", boom)
    assert cli.main(["update"]) == 0
    out = capsys.readouterr().out
    assert "updated" in out and "server" in out.lower()
