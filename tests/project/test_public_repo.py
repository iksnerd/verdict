"""The repository is public; the author's machine and private work are not.

An audit on 2026-09-26 found home paths, private repo and skill names, the author's own prompts
quoted from agent transcripts, private room contents, and a sentence about a person's birth data
in tracked files. Nothing secret, but none of it belongs in a public project. This keeps it out:
every text file that ships is scanned for the markers of that material. `CLAUDE.local.md` is
gitignored and is where machine-specific agent notes belong.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fakes import REPO_ROOT

ROOT = REPO_ROOT

#: Directories that are gitignored or not ours; the pre-commit hook tests an export of the index,
#: which has none of them, so this matches what would be committed.
SKIP_DIRS = {".git", ".venv", "data", "runs", "experiments", "models", "node_modules",
             "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build"}
SKIP_FILES = {"CLAUDE.local.md", Path(__file__).name, "test_plugin.py"}  # the last two list the markers
SUFFIXES = {".md", ".py", ".json", ".jsonl", ".toml", ".yml", ".yaml", ".txt", ".cfg", ".sh"}

#: Markers of the author's machine and private work. Each hit is a path, a private repo, skill or
#: tool name, or a description of real private data.
PRIVATE = (
    "/Users/",
    "PycharmProjects",
    "CodeEditorLand",
    "personal-skills",
    "personal:verdict",
    "personal:council-hub",
    "gpt-alpha",
    "Council Hub",
    "council-hub",
    "local-whisper",
    "natal",
    "~/.claude/projects",
    "iksnerd/verdict-train",
    "apol-findings",
    "iksnerd/apol",
    ".apol/",
    "iksnerd/personal-skills",
)


def shipped_files():
    for path in ROOT.rglob("*"):
        if (path.is_file() and path.suffix in SUFFIXES and path.name not in SKIP_FILES
                and not SKIP_DIRS & set(path.relative_to(ROOT).parts)):
            yield path


def test_there_is_something_to_scan():
    assert sum(1 for _ in shipped_files()) > 50


@pytest.mark.parametrize("marker", PRIVATE)
def test_no_shipped_file_mentions_the_authors_private_setup(marker):
    hits = []
    for path in shipped_files():
        for n, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if marker in line:
                hits.append(f"{path.relative_to(ROOT)}:{n}")
    assert not hits, f"{marker!r} in {len(hits)} place(s): {', '.join(hits[:12])}"
