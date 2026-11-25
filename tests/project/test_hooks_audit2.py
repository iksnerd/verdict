"""The pre-commit hook and release workflow keep the guarantees the audit asked for."""

import os
import re
import shutil
import signal

import pytest

from hook_support import (HOOK, create_run, git, hook_code, release_steps, run_hook, start_hook, step_index,
                          workflow, write_exe)


def test_hook_takes_one_index_snapshot():
    code = hook_code()
    assert code.count("git write-tree") == 2  # once, and once more if HEAD moved in between
    for banned in ("--cached", "--pre-commit --staged", "gitleaks git", "checkout-index"):
        assert banned not in code


def test_hook_passes_clean_and_scans_the_snapshot_diff(env):
    repo, _, tmp_path, e = env
    (repo / "b.txt").write_text("hello\n")
    git(repo, "add", "b.txt")
    r = run_hook(repo, e)
    assert r.returncode == 0, r.stderr
    assert "hello" in (tmp_path / "scanned").read_text()


def test_hook_blocks_secret_and_forbidden_paths(env):
    repo, _, _, e = env
    (repo / "s.txt").write_text("SECRET\n")
    git(repo, "add", "s.txt")
    assert run_hook(repo, e).returncode == 1
    git(repo, "rm", "-q", "--cached", "s.txt")
    (repo / "data").mkdir()
    (repo / "data" / "x.jsonl").write_text("x\n")
    git(repo, "add", "-f", "data/x.jsonl")
    r = run_hook(repo, e)
    assert r.returncode == 1 and "never be committed" in r.stderr


def test_hook_tests_run_on_staged_content_not_worktree(env):
    repo, _, _, e = env
    write_exe(repo / ".venv" / "bin" / "python",
         'case "$1" in -c) exit 0;; esac\ngrep -q staged src/m.txt\n')
    (repo / "src").mkdir()
    (repo / "src" / "m.txt").write_text("staged\n")
    git(repo, "add", "src/m.txt")
    (repo / "src" / "m.txt").write_text("worktree only\n")
    assert run_hook(repo, e).returncode == 0


def test_skipped_checks_are_announced(env):
    repo, bin_, _, e = env
    (bin_ / "gitleaks").unlink()
    r = run_hook(repo, e)
    assert r.returncode == 0
    assert re.search(r"SKIPPED secret scan", r.stderr)
    write_exe(repo / ".venv" / "bin" / "python",
         'case "$1" in -c) exit 1;; esac\necho 1 passed\n')
    r = run_hook(repo, e)
    assert r.returncode == 0 and "SKIPPED mlx/laya tests" in r.stderr


def test_missing_venv_still_aborts(env):
    repo, _, _, e = env
    shutil.rmtree(repo / ".venv")
    r = run_hook(repo, e)
    assert r.returncode == 1 and "no .venv" in r.stderr


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
def test_snapshot_removed_on_signal(env, sig):
    repo, _, tmp_path, e = env
    p, marker = start_hook(repo, e, tmp_path)
    assert list((tmp_path / "tmp").glob("verdict-precommit.*"))
    os.kill(int(marker.read_text()), signal.SIGKILL)  # sh runs its trap after the child ends
    os.kill(p.pid, sig)
    p.wait(timeout=10)
    p.stdout.close()
    p.stderr.close()
    assert p.returncode != 0
    assert not list((tmp_path / "tmp").glob("verdict-precommit.*"))


# ---- release workflow ----

def test_release_is_draft_then_upload_then_publish_last():
    steps = release_steps()
    create = step_index(steps, "gh release create")
    upload = step_index(steps, "gh release upload")
    publish = step_index(steps, "--draft=false")
    assert create < upload < publish == len(steps) - 1
    assert "--draft" in steps[create]["run"]
    assert "dist/*" not in steps[create]["run"]
    assert "dist/*" in steps[upload]["run"]


def test_release_rerun_deletes_only_a_draft():
    run = create_run()
    assert "isDraft" in run
    assert run.index("gh release delete") < run.index("gh release create")
    assert '= "true"' in run  # delete is guarded by the draft check
    assert "--cleanup-tag" not in run


def test_release_keeps_existing_gates():
    steps = release_steps()
    runs = [s.get("run", "") for s in steps]
    gate = max(step_index(steps, "gh release create"), 0)
    for needle in ("docs/release-notes/$GITHUB_REF_NAME.md", "pytest -q",
                   "does not match pyproject version", "verdict bench --verify", "uv build"):
        assert any(needle in r for r in runs[:gate]), needle
    assert workflow()["jobs"]["test-build-release"]["runs-on"] == "macos-26"
