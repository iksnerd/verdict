"""Fourth audit: the pre-commit hook checks each command's own exit status (POSIX sh has no
pipefail), reads HEAD before the index, and dies with its pytest child; the release workflow
serialises per tag and looks a draft up by tag."""
import os
import signal
import subprocess
import time

import pytest

from hook_support import HOOK, create_run, git, hook_code, run_hook, start_hook, workflow, write_exe

GIT = "/usr/bin/git"


def _wrap_git(bin_, fail: str):
    """A `git` ahead of the real one on PATH whose subcommand `fail` exits 1 after running."""
    write_exe(bin_ / "git", f'"{GIT}" "$@"\nrc=$?\n[ "$1" = "{fail}" ] && exit 1\nexit $rc\n')


def test_a_failing_diff_tree_blocks_the_commit(env):
    repo, bin_, _, e = env
    (repo / "b.txt").write_text("hello\n")
    git(repo, "add", "b.txt")
    _wrap_git(bin_, "diff-tree")
    r = run_hook(repo, e)
    assert r.returncode != 0 and "diff-tree" in r.stderr


def test_a_failing_git_archive_blocks_the_commit(env):
    repo, bin_, _, e = env
    _wrap_git(bin_, "archive")
    r = run_hook(repo, e)
    assert r.returncode != 0 and "archive" in r.stderr


def test_head_is_read_before_the_index_tree():
    text = hook_code()
    assert text.index("rev-parse HEAD") < text.index("git write-tree")


@pytest.mark.parametrize("sig,code", [(signal.SIGINT, 130), (signal.SIGTERM, 143),
                                      (signal.SIGHUP, 129)])
def test_a_signal_to_sh_alone_kills_the_pytest_child(env, sig, code):
    repo, _, tmp_path, e = env
    p, marker = start_hook(repo, e, tmp_path,
                           # The real hook runs this suite as a background job, and a background job of
                           # a non-interactive sh starts with SIGINT ignored; a shell cannot trap a
                           # signal that was ignored on entry, so restore the default for the child.
                           preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
    child = int(marker.read_text())
    os.kill(p.pid, sig)  # the child is left alone: the trap must end it
    try:
        _, err = p.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate()
        pytest.fail("the hook kept waiting for its pytest child")
    assert p.returncode == code, err
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)
    assert not list((tmp_path / "tmp").glob("verdict-precommit.*"))


def test_release_runs_are_serialised_per_tag_and_never_cancelled():
    conc = workflow().get("concurrency") or workflow()["jobs"]["test-build-release"].get("concurrency")
    assert conc == {"group": "release-${{ github.ref }}", "cancel-in-progress": False}


def test_a_leftover_draft_is_found_by_tag_not_by_a_capped_list():
    run = create_run()
    assert "gh release list" not in run and "--limit" not in run
    assert 'gh release view "$GITHUB_REF_NAME" --json isDraft' in run
    assert run.index("gh release view") < run.index("gh release delete") < run.index("gh release create")
    assert "--cleanup-tag" not in run
