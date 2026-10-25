"""The agent skill ships from this repo as a Claude Code plugin (`claude plugin marketplace add
iksnerd/verdict`). It is the canonical copy: an older one in `skill/` drifted into contradicting
the maintained skill and was retired, so the checks here are what keep this one honest."""
from __future__ import annotations

import json
import re
import tomllib

import pytest
from fakes import REPO_ROOT

ROOT = REPO_ROOT
PLUGIN = ROOT / "plugins" / "verdict"
SKILL = PLUGIN / "skills" / "verdict"


def test_the_marketplace_lists_the_plugin():
    market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text())
    assert [p["source"] for p in market["plugins"]] == ["./plugins/verdict"]


def test_the_plugin_version_is_the_package_version():
    """`claude plugin update` compares version strings, not content: a plugin version left behind
    serves the old skill forever. Tied to the package, every release bumps it."""
    plugin = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    package = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert plugin["version"] == package
    assert isinstance(plugin["author"], dict), "a string author fails `claude plugin install`"


def test_the_skill_names_its_directory():
    text = (SKILL / "SKILL.md").read_text()
    assert re.search(r"^name: verdict$", text, re.M)


@pytest.mark.parametrize("path", [SKILL / "SKILL.md", SKILL / "references" / "questions.md"],
                         ids=lambda p: p.name)
def test_nothing_private_ships(path):
    """The skill is public; the author's machine is not. EVOLUTION.md is history and exempt."""
    text = path.read_text()
    for private in ("/Users/", "CodeEditorLand", "PycharmProjects", "gpt-alpha", "HF_TOKEN lives"):
        assert private not in text, f"{path.name} mentions {private}"


def test_every_eval_case_has_a_prompt_and_graders():
    cases = sorted((PLUGIN / "evals" / "verdict").iterdir())
    assert len(cases) >= 5
    for case in cases:
        assert (case / "prompt.md").is_file(), case.name
        assert any((case / "graders").glob("*.md")), case.name
        assert "personal:verdict" not in "".join(p.read_text() for p in case.rglob("*.md"))
