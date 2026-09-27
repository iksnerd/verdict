"""`verdict update`: pull a checkout and re-sync with both extras, or reinstall a `uv tool` at the
newest release tag. Nothing else here runs anything on a verdict's behalf (see the module
docstring at the package's top level)."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import setup, support

#: Where releases come from. A `uv tool install` has no checkout, so it updates to the newest
#: `vX.Y.Z` tag here, which the release workflow has tested and built.
REPO_URL = os.environ.get("VERDICT_REPO", "https://github.com/iksnerd/verdict.git")
TOOL_EXTRAS = "mlx,laya"


def _checkout() -> Path:
    """The repo the running CLI was installed from: `src/verdict/cli/update.py` sits four levels
    in. The `~/.local/bin` shim runs this repo's venv with an editable install, so updating the
    checkout is updating the tool."""
    return Path(__file__).resolve().parents[3]


def _run(argv: list[str], cwd: Path | None = None) -> tuple[int, str]:
    """Run a maintenance command and return (status, combined output). Only `verdict update`
    uses this: every decision command still only answers, and nothing here runs on a verdict."""
    import subprocess

    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _server_up() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{support._settings().url}/healthz", timeout=0.3) as r:
            return r.status == 200
    except OSError:
        return False


def _release_tags(url: str) -> list[tuple[int, ...]]:
    """`vX.Y.Z` tags on the remote, as sortable tuples. Anything else (rc, nightly) is ignored."""
    import re

    code, out = _run(["git", "ls-remote", "--tags", "--refs", url])
    if code != 0:
        raise RuntimeError(f"could not list releases at {url}:\n{out}")
    found = re.findall(r"refs/tags/v(\d+)\.(\d+)\.(\d+)$", out, flags=re.M)
    return sorted(tuple(int(p) for p in t) for t in found)


def _update_tool(args: argparse.Namespace) -> int:
    """Update a `uv tool install`: reinstall the newest release tag, with the extras."""
    current = support._version().split()[-1]
    try:
        tags = _release_tags(REPO_URL)
    except RuntimeError as exc:
        return support._fail(exc)
    if not tags:
        return support._fail(f"no vX.Y.Z release tags found at {REPO_URL}")
    newest = ".".join(map(str, tags[-1]))
    try:
        have = tuple(int(p) for p in current.split("."))
    except ValueError:
        have = (0,)
    if have >= tags[-1]:
        print(f"verdict {current} is up to date (newest release v{newest})")
        return 0
    if args.check:
        print(f"verdict {current} installed; v{newest} is available. `verdict update` installs it")
        return 1
    spec = f"verdict[{TOOL_EXTRAS}] @ git+{_git_url(REPO_URL)}@v{newest}"
    install = ["uv", "tool", "install", "--force", "--python", "3.11", spec]
    code, out = _run(install)
    if code != 0 and "Could not parse object" in out:
        # uv resolved the new tag but its cached clone predates the commit, seen right after a tag
        # was pushed. Refreshing makes it fetch; once, so a real failure is not looped.
        code, out = _run(install[:3] + ["--refresh-package", "verdict"] + install[3:])
    if code != 0:
        return support._fail(f"`uv tool install {spec}` failed:\n{out}")
    print(f"updated {current} -> {newest}")
    if _server_up():
        print("A verdict server is running the old code: restart `verdict serve` to use it.")
    return 0


def _git_url(url: str) -> str:
    """`git@host:owner/repo.git` as the `ssh://` form a pip VCS URL needs."""
    if url.startswith("git@"):
        host, path = url[4:].split(":", 1)
        return f"ssh://git@{host}/{path}"
    return url


def _update_cmd(args: argparse.Namespace) -> int:
    """Fast-forward the checkout and re-sync with both extras.

    The extras are the point: `uv sync` alone removes `mlx` and `laya`, which silently breaks the
    CLI and the tests that need them (CLAUDE.md). Refuses on uncommitted changes rather than
    mixing an update into someone's work, and only ever fast-forwards.
    """
    repo = _checkout()
    if not (repo / ".git").exists():
        return _update_tool(args)

    def step(argv: list[str]) -> str | None:
        code, out = _run(argv, cwd=repo)
        if code != 0:
            raise RuntimeError(f"`{' '.join(argv)}` failed:\n{out}")
        return out

    try:
        step(["git", "fetch", "--quiet", "origin"])
        behind = int(step(["git", "rev-list", "--count", "HEAD..@{upstream}"]) or 0)
        if args.check:
            print(f"{support._version()} is {behind} commit(s) behind ({repo}); `verdict update` pulls them"
                  if behind else f"{support._version()} is up to date ({repo})")
            return 1 if behind else 0
        if not behind:
            print(f"{support._version()} is up to date ({repo})")
            return 0
        if step(["git", "status", "--porcelain", "--untracked-files=no"]):
            return support._fail(f"{repo} has uncommitted changes; commit or stash them, then update")
        before = step(["git", "rev-parse", "--short", "HEAD"])
        step(["git", "pull", "--ff-only", "--quiet"])
        step(["uv", "sync", "--quiet", "--extra", "mlx", "--extra", "laya"])
        after = step(["git", "rev-parse", "--short", "HEAD"])
        log = step(["git", "log", "--oneline", f"{before}..{after}"])
    except RuntimeError as exc:
        return support._fail(exc)

    import tomllib

    new = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["version"]
    print(f"updated {support._version().split()[-1]} -> {new} ({before}..{after}, {behind} commit(s))")
    for line in (log or "").splitlines()[:10]:
        print(f"  {line}")
    if _server_up():
        print("A verdict server is running the old code: restart `verdict serve` to use it.")
    return 0
