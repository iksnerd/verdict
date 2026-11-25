"""Helpers for the pre-commit hook and release workflow tests. A plain module, not conftest.py:
pytest loads every conftest under the name `conftest`, which would hide the root one from
`from conftest import typed`. The `env` fixture is in `conftest.py` here."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import yaml
from fakes import REPO_ROOT

HOOK = REPO_ROOT / ".githooks" / "pre-commit"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
SYSTEM_PATH = "/usr/bin:/bin"
_GIT_LOCAL = {"GIT_INDEX_FILE", "GIT_DIR", "GIT_WORK_TREE", "GIT_PREFIX", "GIT_OBJECT_DIRECTORY"}


def hook_code():
    """The pre-commit hook's source with its comment lines removed."""
    return "\n".join(l for l in HOOK.read_text().splitlines() if not l.lstrip().startswith("#"))


def git(cwd, *args):
    # A hook runs with GIT_INDEX_FILE and GIT_DIR set when the commit used a private index: these
    # throwaway repos must not inherit them, or `git add` writes into the real commit's index.
    clean = {k: v for k, v in os.environ.items() if k not in _GIT_LOCAL}
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True,
        env={**clean, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
    )


def write_exe(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + text)
    path.chmod(0o755)


def run_hook(repo, e, **kw):
    return subprocess.run(["sh", ".githooks/pre-commit"], cwd=repo, env=e,
                          capture_output=True, text=True, **kw)


def start_hook(repo, e, tmp_path, **popen_kw):
    """Starts the hook with the fake pytest asleep (`HOOK_SLEEP`), waits until it has written its
    pid to the marker file, and returns `(process, marker)`."""
    p = subprocess.Popen(["sh", ".githooks/pre-commit"], cwd=repo, env={**e, "HOOK_SLEEP": "1"},
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                         **popen_kw)
    marker = tmp_path / "child"
    for _ in range(100):
        if marker.exists():
            break
        time.sleep(0.1)
    assert marker.exists()
    return p, marker


def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def release_steps():
    return workflow()["jobs"]["test-build-release"]["steps"]


def step_index(steps, needle):
    hits = [i for i, s in enumerate(steps) if needle in s.get("run", "")]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def create_run():
    """The shell of the step that runs `gh release create`."""
    steps = release_steps()
    return steps[step_index(steps, "gh release create")]["run"]
