"""verdict is not on PyPI. The name `verdict` there belongs to an unrelated project (Haize Labs' LLM
evaluation library), so `pip install verdict` or `uv tool install --reinstall 'verdict[jev]'` installs
or replaces it with the wrong software. Every install hint must name the GitHub source. 0.7.0 shipped
such hints in four places (README, docs/api.md, the JevBackend import error, the serve error)."""
from __future__ import annotations

import re
import sys

import pytest
import uvicorn

import verdict.backends.mlx as backend_mlx
from verdict import cli, config
from fakes import REPO_ROOT

ROOT = REPO_ROOT
SCANNED = [*ROOT.glob("*.md"), *ROOT.glob("docs/*.md"), *ROOT.glob("examples/**/*.md"),
           *ROOT.glob("plugins/**/*.md"), *(ROOT / "src").rglob("*.py")]
#: An install command that names `verdict` (optionally with extras) and not a source after it.
BARE = re.compile(r"(?:pip|uv)(?: tool| pip)? install[^\n`]*?[\"']?verdict(?:\[[\w,]+\])?[\"']?(?![\w\[@/.-])")


#: Sentences that name the bare command to warn against it, not to recommend it. Exact phrases, so a
#: new offender cannot hide behind a vague word like "not".
WARNINGS = ("`pip install verdict` gets the wrong software", "Not `pip install verdict`",
            "fails on a bare\n", "`pip install verdict` or `uv tool install 'verdict[...]'`",
            "never `pip install\nverdict`", "`pip install verdict`: that name", "bare\n  `pip install verdict`")


def is_bare(line: str) -> bool:
    """An install of `verdict` that names no source: it would resolve to PyPI's unrelated project."""
    return (bool(BARE.search(line)) and "git+" not in line and "@ git" not in line
            and "] @" not in line and not any(w in line for w in WARNINGS))


def offences():
    for path in SCANNED:
        if path.name == "FINDINGS.md":
            continue
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if is_bare(line):
                yield f"{path.relative_to(ROOT)}:{n}: {line.strip()[:110]}"


def test_no_install_hint_names_the_unrelated_pypi_project():
    found = list(offences())
    assert not found, "install hints that would fetch PyPI's `verdict`:\n" + "\n".join(found)


def test_the_scan_catches_the_hints_that_shipped_and_allows_the_git_form():
    assert is_bare("`pip install 'verdict[jev]'` adds TypeSafe's SDK")
    assert is_bare("uv tool install --reinstall 'verdict[mlx,laya,jev]' (or `uv sync`")
    assert is_bare("pip install verdict")
    assert not is_bare("uv tool install --force 'verdict[mlx,laya] @ git+https://github.com/x/verdict.git'")
    assert not is_bare("pip install typesafe-sdk")


def test_serve_without_the_sdk_prints_a_command_that_installs_this_project(monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATHS", ())
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "verdict.backends.jev", None)         # the import raises
    monkeypatch.setattr(backend_mlx, "MlxBackend", lambda *a, **k: pytest.fail("no local model"))
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("must not serve"))
    assert cli.main(["serve", "--port", "0", "--systemone", "https://api.typesafe.ai"]) == 2
    err = capsys.readouterr().err
    assert "git+https://github.com/iksnerd/verdict.git@v" in err and "jev" in err
    assert "--reinstall" not in err


def test_the_library_import_error_installs_the_sdk_not_the_wrong_project(monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    monkeypatch.delitem(sys.modules, "verdict.backends.jev", raising=False)
    with pytest.raises(ImportError) as info:
        import verdict.backends.jev  # noqa: F401
    text = str(info.value)
    assert "pip install typesafe-sdk" in text and "verdict[" not in text
