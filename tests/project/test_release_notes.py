"""A release page says what changed. `gh release create --generate-notes` writes only a "Full
Changelog" link for direct commits (v0.7.0's page had no description), so the notes are a file in the
repo, reviewed with the code, and the workflow publishes that file and refuses a tag without one."""
from __future__ import annotations

import tomllib

from fakes import REPO_ROOT

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
NOTES = REPO_ROOT / "docs" / "release-notes"


def test_the_release_workflow_publishes_the_notes_file_not_generated_notes():
    text = WORKFLOW.read_text()
    assert '--notes-file "docs/release-notes/$GITHUB_REF_NAME.md"' in text
    assert "--generate-notes" not in text


def test_the_release_workflow_refuses_a_tag_with_no_notes_before_it_runs_the_suite():
    text = WORKFLOW.read_text()
    check = text.index('notes="docs/release-notes/$GITHUB_REF_NAME.md"')
    assert check < text.index("uv run --no-sync pytest")
    assert 'if [ ! -s "$notes" ]' in text


def test_the_current_version_has_release_notes():
    """Written before the version is bumped, so the tag cannot be the first time anyone sees a gap."""
    version = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["version"]
    notes = NOTES / f"v{version}.md"
    assert notes.is_file() and notes.read_text().strip(), f"{notes.relative_to(REPO_ROOT)} is missing"
