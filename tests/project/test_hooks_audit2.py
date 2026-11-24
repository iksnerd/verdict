"""The pre-commit hook and release workflow keep the guarantees the audit asked for."""

import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / ".githooks" / "pre-commit"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
SYSTEM_PATH = "/usr/bin:/bin"


def _git(cwd, *args):
    subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
    )


def _exe(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + text)
    path.chmod(0o755)


@pytest.fixture
def env(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / ".githooks").mkdir()
    shutil.copy(HOOK, repo / ".githooks" / "pre-commit")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    bin_ = tmp_path / "bin"
    # fake gitleaks: records what it scanned, fails when it sees SECRET
    _exe(bin_ / "gitleaks", f'cat > "{tmp_path}/scanned"\n! grep -q SECRET "{tmp_path}/scanned"\n')
    # fake python: the extras probe passes, pytest passes
    _exe(repo / ".venv" / "bin" / "python",
         f'case "$1" in -c) exit 0;; esac\necho 1 passed\n'
         f'[ -n "${{HOOK_SLEEP:-}}" ] && {{ echo $$ > "{tmp_path}/child"; sleep 20; }}\nexit 0\n')
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    e = {"PATH": f"{bin_}:{SYSTEM_PATH}", "TMPDIR": str(tmp), "HOME": str(tmp_path),
         "GIT_CONFIG_GLOBAL": "/dev/null"}
    return repo, bin_, tmp_path, e


def _run(repo, e, **kw):
    return subprocess.run(["sh", ".githooks/pre-commit"], cwd=repo, env=e,
                          capture_output=True, text=True, **kw)


def test_hook_takes_one_index_snapshot():
    code = "\n".join(l for l in HOOK.read_text().splitlines() if not l.lstrip().startswith("#"))
    assert code.count("git write-tree") == 1
    for banned in ("--cached", "--pre-commit --staged", "gitleaks git", "checkout-index"):
        assert banned not in code


def test_hook_passes_clean_and_scans_the_snapshot_diff(env):
    repo, _, tmp_path, e = env
    (repo / "b.txt").write_text("hello\n")
    _git(repo, "add", "b.txt")
    r = _run(repo, e)
    assert r.returncode == 0, r.stderr
    assert "hello" in (tmp_path / "scanned").read_text()


def test_hook_blocks_secret_and_forbidden_paths(env):
    repo, _, _, e = env
    (repo / "s.txt").write_text("SECRET\n")
    _git(repo, "add", "s.txt")
    assert _run(repo, e).returncode == 1
    _git(repo, "rm", "-q", "--cached", "s.txt")
    (repo / "data").mkdir()
    (repo / "data" / "x.jsonl").write_text("x\n")
    _git(repo, "add", "-f", "data/x.jsonl")
    r = _run(repo, e)
    assert r.returncode == 1 and "never be committed" in r.stderr


def test_hook_tests_run_on_staged_content_not_worktree(env):
    repo, _, _, e = env
    _exe(repo / ".venv" / "bin" / "python",
         'case "$1" in -c) exit 0;; esac\ngrep -q staged src/m.txt\n')
    (repo / "src").mkdir()
    (repo / "src" / "m.txt").write_text("staged\n")
    _git(repo, "add", "src/m.txt")
    (repo / "src" / "m.txt").write_text("worktree only\n")
    assert _run(repo, e).returncode == 0


def test_skipped_checks_are_announced(env):
    repo, bin_, _, e = env
    (bin_ / "gitleaks").unlink()
    r = _run(repo, e)
    assert r.returncode == 0
    assert re.search(r"SKIPPED secret scan", r.stderr)
    _exe(repo / ".venv" / "bin" / "python",
         'case "$1" in -c) exit 1;; esac\necho 1 passed\n')
    r = _run(repo, e)
    assert r.returncode == 0 and "SKIPPED mlx/laya tests" in r.stderr


def test_missing_venv_still_aborts(env):
    repo, _, _, e = env
    shutil.rmtree(repo / ".venv")
    r = _run(repo, e)
    assert r.returncode == 1 and "no .venv" in r.stderr


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_snapshot_removed_on_signal(env, sig):
    repo, _, tmp_path, e = env
    p = subprocess.Popen(["sh", ".githooks/pre-commit"], cwd=repo,
                         env={**e, "HOOK_SLEEP": "1"}, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    marker = tmp_path / "child"
    for _ in range(100):
        if marker.exists():
            break
        time.sleep(0.1)
    assert marker.exists()
    assert list((tmp_path / "tmp").glob("verdict-precommit.*"))
    os.kill(int(marker.read_text()), signal.SIGKILL)  # sh runs its trap after the child ends
    os.kill(p.pid, sig)
    p.wait(timeout=10)
    p.stdout.close()
    p.stderr.close()
    assert p.returncode != 0
    assert not list((tmp_path / "tmp").glob("verdict-precommit.*"))


# ---- release workflow ----

def _steps():
    wf = yaml.safe_load(WORKFLOW.read_text())
    return wf["jobs"]["test-build-release"]["steps"]


def _idx(steps, needle):
    hits = [i for i, s in enumerate(steps) if needle in s.get("run", "")]
    assert len(hits) == 1, (needle, hits)
    return hits[0]


def test_release_is_draft_then_upload_then_publish_last():
    steps = _steps()
    create = _idx(steps, "gh release create")
    upload = _idx(steps, "gh release upload")
    publish = _idx(steps, "--draft=false")
    assert create < upload < publish == len(steps) - 1
    assert "--draft" in steps[create]["run"]
    assert "dist/*" not in steps[create]["run"]
    assert "dist/*" in steps[upload]["run"]


def test_release_rerun_deletes_only_a_draft():
    run = _steps()[_idx(_steps(), "gh release create")]["run"]
    assert "isDraft" in run
    assert run.index("gh release delete") < run.index("gh release create")
    assert "select(.tagName" in run and ".isDraft" in run  # delete is guarded by the draft filter
    assert "--cleanup-tag" not in run


def test_release_keeps_existing_gates():
    steps = _steps()
    runs = [s.get("run", "") for s in steps]
    gate = max(_idx(steps, "gh release create"), 0)
    for needle in ("docs/release-notes/$GITHUB_REF_NAME.md", "pytest -q",
                   "does not match pyproject version", "verdict bench --verify", "uv build"):
        assert any(needle in r for r in runs[:gate]), needle
    assert yaml.safe_load(WORKFLOW.read_text())["jobs"]["test-build-release"]["runs-on"] == "macos-26"
