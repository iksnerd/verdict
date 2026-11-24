"""`verdict update`: pull a checkout and re-sync with both extras, or reinstall a `uv tool` at the
newest release tag. Nothing else here runs anything on a verdict's behalf (see the module
docstring at the package's top level)."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from . import setup, support

#: Where releases come from. A `uv tool install` has no checkout, so it updates to the newest
#: `vX.Y.Z` tag here that has a published release (see `_released`).
REPO_URL = os.environ.get("VERDICT_REPO", "https://github.com/iksnerd/verdict.git")
TOOL_EXTRAS = "mlx,laya"


def _has_jev() -> bool:
    """Whether TypeSafe's SDK (the `jev` extra) is installed here. An update keeps it if so, like
    mlx and laya, but does not add it to a machine that never asked for hosted Jev."""
    import importlib.util

    return importlib.util.find_spec("typesafe_sdk") is not None


def _extras() -> list[str]:
    return TOOL_EXTRAS.split(",") + (["jev"] if _has_jev() else [])


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


def _server_up() -> bool | None:
    """Whether a verdict server answers, or None when that cannot be known (the settings do not
    load). This runs after an update has finished, so it must not be able to fail one."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"{support._settings().url}/healthz", timeout=0.3) as r:
            if r.status != 200:
                return False
            try:
                body = json.loads(r.read())
            except ValueError:
                return True  # a 200 that is not JSON still answered; only `ready: false` says no
            # A server that is up but still loading says so; no `ready` key means ready.
            return not (isinstance(body, dict) and body.get("ready") is False)
    except OSError:
        return False
    except Exception:
        return None


def _note_server() -> None:
    up = _server_up()
    if up:
        print("A verdict server is running the old code: restart `verdict serve` to use it.")
    elif up is None:
        print("Could not tell whether a verdict server is running (settings did not load); "
              "if one is, restart `verdict serve` to use the new code.")


def _github_slug(url: str) -> str | None:
    """`owner/name` when the remote is on github.com, else None: only there is a release a thing
    separate from a tag that can be asked about."""
    import re

    m = re.fullmatch(r"(?:https://github\.com/|git@github\.com:|ssh://git@github\.com/)"
                     r"([^/\s]+/[^/\s]+?)(?:\.git)?/?", url)
    return m.group(1) if m else None


def _released(url: str) -> set[tuple[int, ...]] | None:
    """The `vX.Y.Z` tags that have a published release, or None when that cannot be known (the
    remote is not GitHub, or GitHub did not answer). A release is published only after its gates
    passed, so a tag without one is a release that was refused or is still being made."""
    import json
    import re
    import urllib.request

    slug = _github_slug(url)
    if slug is None:
        return None
    request = urllib.request.Request(
        f"https://api.github.com/repos/{slug}/releases?per_page=100",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "verdict-update"})
    try:
        with urllib.request.urlopen(request, timeout=10) as r:
            releases = json.loads(r.read())
    except (OSError, ValueError) as exc:
        support._warn(f"could not read published releases from GitHub ({exc}); "
                      "the published-release filter was skipped, so any vX.Y.Z tag counts")
        return None
    if not isinstance(releases, list):
        support._warn("GitHub's release list had an unexpected shape; the published-release "
                      "filter was skipped, so any vX.Y.Z tag counts")
        return None
    out = set()
    for rel in releases:
        if not isinstance(rel, dict):
            continue
        m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", str(rel.get("tag_name", "")))
        if m and not rel.get("draft") and not rel.get("prerelease"):
            out.add(tuple(int(p) for p in m.groups()))
    return out


def _release_tags(url: str) -> list[tuple[int, ...]]:
    """`vX.Y.Z` tags on the remote that are releases, as sortable tuples. Anything else (rc,
    nightly) is ignored. On GitHub a tag counts only once its release is published; where that
    cannot be asked, every such tag counts."""
    import re

    code, out = _run(["git", "ls-remote", "--tags", "--refs", url])
    if code != 0:
        raise RuntimeError(f"could not list releases at {url}:\n{out}")
    found = re.findall(r"refs/tags/v(\d+)\.(\d+)\.(\d+)$", out, flags=re.M)
    tags = sorted(tuple(int(p) for p in t) for t in found)
    released = _released(url)
    return tags if released is None else [t for t in tags if t in released]


def _update_tool(args: argparse.Namespace) -> int:
    """Update a `uv tool install`: reinstall the newest release tag, with the extras."""
    current = support._version().split()[-1]
    pin = current if re.fullmatch(r"\d+\.\d+\.\d+", current) else None
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
    extras = _extras()  # before the install: `--force` removes the old venv, and the SDK with it
    spec = _tool_spec(extras, newest)
    install = _tool_install_argv(spec)
    try:
        code, out = _run(install)
        if code != 0 and any(marker in out for marker in STALE_CACHE):
            # uv resolved the new tag but its cached clone predates the commit, seen right after a tag
            # was pushed. Refreshing makes it fetch; once, so a real failure is not looped.
            code, out = _run(install[:3] + ["--refresh-package", "verdict"] + install[3:])
    except KeyboardInterrupt:
        # The old install may already be gone; put it back before leaving.
        support._warn(f"interrupted. {_restore_guarded(pin, extras)[0]}")
        return 130
    if code != 0:
        note, interrupted = _restore_guarded(pin, extras)
        if interrupted:
            support._warn(f"`uv tool install {spec}` failed:\n{out}\n{note}")
            return 130
        return support._fail(f"`uv tool install {spec}` failed:\n{out}\n{note}")
    print(f"updated {current} -> {newest}")
    _note_server()
    return 0


#: What uv prints when its cached clone of the repo lacks a commit it just resolved: a tag pushed
#: seconds ago. Both were seen releasing 0.7.0; a refresh fixes either.
STALE_CACHE = ("Could not parse object", "unable to read tree")


def _restore_guarded(version: str | None, extras: list[str]) -> tuple[str, bool]:
    """`_restore`, but a second Ctrl-C during it ends in a plain message, not a bare exit 130:
    the old install may be gone, and the line to bring it back is the one thing to print."""
    try:
        return _restore(version, extras), False
    except KeyboardInterrupt:
        return ("interrupted again while restoring, so the old install may be missing and `verdict` "
                f"may not run. Reinstall it with:\n  {_reinstall_line(extras, version)}"), True


def _restore(version: str | None, extras: list[str]) -> str:
    """Put the version that was installed back, and say what happened.

    `uv tool install --force` removes the old install before it builds the new one, so a failed
    update left no `verdict` on the PATH at all (found installing 0.7.0 from its own tag)."""
    if version is None:
        return ("the previous version is not a release tag, so it was not restored and `verdict` "
                "may be missing. Reinstall it by hand with:\n"
                f"  {_reinstall_line(extras, None)}")
    spec = _tool_spec(extras, version)
    install = _tool_install_argv(spec)
    code, _ = _run(install)
    if code == 0:
        return (f"restored v{version} by reinstalling it, so `verdict` works again. The failed "
                "install may have changed the tool before it stopped: `verdict --version` says "
                "which version is installed")
    return (f"v{version} could not be restored either, so `verdict` may be missing. Reinstall it with:\n"
            f"  {_reinstall_line(extras, version)}")


def _reinstall_line(extras: list[str], version: str | None) -> str:
    """The `uv tool install` line to bring `version` back; a placeholder tag, marked as such, when
    the previous version was not a release."""
    if version is None:
        return f"{_tool_install_cmd(_tool_spec(extras, 'X.Y.Z'))}  (a release tag)"
    return _tool_install_cmd(_tool_spec(extras, version))


def _tool_spec(extras: list[str], tag: str) -> str:
    """The `uv tool install` requirement for release `tag` (no leading `v`) with `extras`."""
    return f"verdict[{','.join(extras)}] @ git+{_git_url(REPO_URL)}@v{tag}"


def _tool_install_argv(spec: str) -> list[str]:
    return ["uv", "tool", "install", "--force", "--python", "3.11", spec]


def _tool_install_cmd(spec: str) -> str:
    """The install as a shell line to print, the spec quoted."""
    return " ".join(_tool_install_argv(f"'{spec}'"))


def _git_url(url: str) -> str:
    """`git@host:owner/repo.git` as the `ssh://` form a pip VCS URL needs."""
    if url.startswith("git@"):
        host, path = url[4:].split(":", 1)
        return f"ssh://git@{host}/{path}"
    return url


def _update_cmd(args: argparse.Namespace) -> int:
    """Fast-forward the checkout and re-sync with both extras, also when nothing was pulled.

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

    def sync() -> None:
        """Runs when nothing was pulled too: a pull that worked and a sync that failed leaves HEAD
        current and the environment stale, and the next run is the only thing that can repair it."""
        argv = ["uv", "sync", "--quiet", *(a for extra in _extras() for a in ("--extra", extra))]
        code, out = _run(argv, cwd=repo)
        if code != 0:
            raise RuntimeError(f"`{' '.join(argv)}` failed:\n{out}\n"
                               f"{repo} is current but its environment was not synced; "
                               "rerun `verdict update` to sync it")

    try:
        step(["git", "fetch", "--quiet", "origin"])
        behind = int(step(["git", "rev-list", "--count", "HEAD..@{upstream}"]) or 0)
        if args.check:
            print(f"{support._version()} is {behind} commit(s) behind ({repo}); `verdict update` pulls them"
                  if behind else f"{support._version()} is up to date ({repo}); the environment "
                  "was not checked, `verdict update` re-syncs it")
            return 1 if behind else 0
        if not behind:
            sync()
            print(f"{support._version()} is up to date ({repo}); environment synced")
            return 0
        if step(["git", "status", "--porcelain", "--untracked-files=no"]):
            return support._fail(f"{repo} has uncommitted changes; commit or stash them, then update")
        before = step(["git", "rev-parse", "--short", "HEAD"])
        step(["git", "pull", "--ff-only", "--quiet"])
        sync()
        after = step(["git", "rev-parse", "--short", "HEAD"])
        log = step(["git", "log", "--oneline", f"{before}..{after}"])
    except RuntimeError as exc:
        return support._fail(exc)

    import tomllib

    new = tomllib.loads((repo / "pyproject.toml").read_text())["project"]["version"]
    print(f"updated {support._version().split()[-1]} -> {new} ({before}..{after}, {behind} commit(s))")
    for line in (log or "").splitlines()[:10]:
        print(f"  {line}")
    _note_server()
    return 0
