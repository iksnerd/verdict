"""Fixtures for the hook tests; the helpers they use are in `hook_support.py`.

Pytest loads this file under the module name `conftest`, which hides `tests/conftest.py` from
`from conftest import typed`, so `typed` is re-exported below, as `tests/cli/conftest.py` does."""
from __future__ import annotations

import shutil

import pytest

from fakes import typed  # noqa: F401  (tests import `typed` from `conftest`)
from hook_support import HOOK, SYSTEM_PATH, git, write_exe


@pytest.fixture
def env(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / ".githooks").mkdir()
    shutil.copy(HOOK, repo / ".githooks" / "pre-commit")
    (repo / "a.txt").write_text("a\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")
    bin_ = tmp_path / "bin"
    # fake gitleaks: records what it scanned, fails when it sees SECRET
    write_exe(bin_ / "gitleaks", f'cat > "{tmp_path}/scanned"\n! grep -q SECRET "{tmp_path}/scanned"\n')
    # fake python: the extras probe passes, pytest passes
    write_exe(repo / ".venv" / "bin" / "python",
              f'case "$1" in -c) exit 0;; esac\necho 1 passed\n'
              f'[ -n "${{HOOK_SLEEP:-}}" ] && {{ echo $$ > "{tmp_path}/child"; sleep 20; }}\nexit 0\n')
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    e = {"PATH": f"{bin_}:{SYSTEM_PATH}", "TMPDIR": str(tmp), "HOME": str(tmp_path),
         "GIT_CONFIG_GLOBAL": "/dev/null"}
    return repo, bin_, tmp_path, e
