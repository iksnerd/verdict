"""`verdict docs`: the README, guide, API notes and FINDINGS, readable from an installed tool.

The help and the refusals cite "FINDINGS §25" throughout, and a `uv tool` install had no copy of
FINDINGS to look it up in. The wheel now carries the docs; in a checkout they are read in place."""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from verdict import cli, docs
from fakes import REPO_ROOT

ROOT = REPO_ROOT


def show(capsys, *argv):
    assert cli.main(["docs", *argv]) == 0
    return capsys.readouterr().out


def test_with_no_topic_it_prints_the_readme(capsys):
    assert show(capsys) == (ROOT / "README.md").read_text()


@pytest.mark.parametrize("topic,path", [("guide", "docs/guide.md"), ("api", "docs/api.md"),
                                        ("findings", "docs/FINDINGS.md")])
def test_each_topic_prints_its_document(capsys, topic, path):
    assert show(capsys, topic) == (ROOT / path).read_text()


def test_a_findings_section_prints_only_that_section(capsys):
    out = show(capsys, "findings", "38")
    assert out.startswith("## 38.") and "## 39." not in out and "## 37." not in out


def test_an_unknown_section_names_the_range(capsys):
    assert cli.main(["docs", "findings", "999"]) == 2
    assert "§999" in capsys.readouterr().err


def test_a_section_needs_findings(capsys):
    assert cli.main(["docs", "guide", "3"]) == 2


def test_every_topic_is_packed_into_the_wheel():
    """A document read in a checkout but missing from the wheel works in tests and fails on
    every install."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packed = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    for source in docs.TOPICS.values():
        assert packed.get(source) == f"verdict/_docs/{Path(source).name}", source


def test_the_cli_offers_every_topic():
    assert tuple(docs.TOPICS) == cli.TOPIC_NAMES


class _ClosedPipe:
    """stdout after the reader (`head`, `less`) has quit: every write fails."""

    def write(self, text):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


def test_a_closed_pipe_ends_quietly(monkeypatch):
    """`verdict docs findings | head` printed a BrokenPipeError traceback. A reader that stops early
    is normal use, not an error."""
    monkeypatch.setattr("sys.stdout", _ClosedPipe())
    assert cli.main(["docs", "findings"]) == 0
