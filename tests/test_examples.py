"""The examples are documentation that can rot, so they are tested like code: every bank must
validate against the wire schema, every input must parse, and no question may name a field its
inputs lack. No model loads; this checks shape, not answers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verdict import inputs
from verdict.schema import DecideRequest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
BANKS = sorted(EXAMPLES.glob("*/bank.json"))


def test_there_are_examples():
    assert len(BANKS) >= 3


@pytest.mark.parametrize("bank_path", BANKS, ids=lambda p: p.parent.name)
def test_bank_validates_and_inputs_match_it(bank_path):
    bank = json.loads(bank_path.read_text())
    files = sorted(bank_path.parent.glob("*.jsonl"))
    assert files, f"{bank_path.parent.name} has no inputs"
    for inputs_file in files:
        rows = [json.loads(l) for l in inputs_file.read_text().splitlines() if l.strip()]
        assert rows, f"{inputs_file} is empty"
        for row in rows:
            state = row.get("state", row)
            DecideRequest(state=state, questions=bank)
            assert not inputs.missing_fields(state, bank), (inputs_file.name, state)


@pytest.mark.parametrize("bank_path", BANKS, ids=lambda p: p.parent.name)
def test_expected_labels_name_real_questions_with_valid_answers(bank_path):
    """A row's `expected` (verdict examples' ground truth for auto-scoring) must be checkable:
    every key is a real question in the bank, `noul` expects a bool, `choice` expects one of its
    own criteria, `score` expects a real level. A typo here would silently score as always-wrong,
    not fail loudly, without this."""
    bank = json.loads(bank_path.read_text())
    for inputs_file in sorted(bank_path.parent.glob("*.jsonl")):
        for line in inputs_file.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            expected = row.get("expected")
            if not expected:
                continue
            for qid, value in expected.items():
                assert qid in bank, (inputs_file.name, qid)
                q = bank[qid]
                if q["type"] == "noul":
                    assert isinstance(value, bool), (inputs_file.name, qid, value)
                elif q["type"] == "choice":
                    assert value in q["criteria"], (inputs_file.name, qid, value)
                elif q["type"] == "score":
                    assert 0 <= int(value) < len(q["criteria"]), (inputs_file.name, qid, value)


def test_every_example_is_listed_in_the_examples_readme():
    readme = (EXAMPLES / "README.md").read_text()
    for bank in BANKS:
        assert f"{bank.parent.name}/" in readme, bank.parent.name


def test_the_documented_install_line_names_the_current_version():
    """The README and the guide pin a release tag in their install line. A release that bumps the
    version without updating them leaves a copy-paste install of the old version, silently."""
    import tomllib

    root = EXAMPLES.parent
    version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
    for doc in ("README.md", "docs/guide.md"):
        text = (root / doc).read_text()
        pins = set(__import__("re").findall(r"verdict\.git@v(\d+\.\d+\.\d+)", text))
        assert pins == {version}, f"{doc} installs {sorted(pins)}, pyproject says {version}"
