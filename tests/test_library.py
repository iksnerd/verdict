"""The question library and the question linter. The library is the measured questions shipped in
the tool, so they can be used by name instead of copied out of a skill file; the linter is the
failed questions turned into a check. No model loads here."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from verdict import bench, cli, inputs, library
from verdict.schema import DecideRequest

ENTRIES = library.load()
SCORECARDS = Path(__file__).resolve().parents[1] / "bench" / "scorecards"


def test_there_is_a_library():
    assert len(ENTRIES) >= 5


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_entry_is_a_valid_question_with_provenance(entry):
    assert re.fullmatch(r"[a-z][a-z0-9_]*", entry.name)
    DecideRequest(state={"text": "x"}, questions={entry.name: entry.question})
    assert entry.measured and entry.source, f"{entry.name} needs a measured result and a source"
    assert entry.state, f"{entry.name} must say what state it was measured on"


def test_names_are_unique_and_do_not_shadow_presets():
    names = [e.name for e in ENTRIES]
    assert len(names) == len(set(names))
    assert not set(names) & set(inputs.PRESETS)


def test_bench_backed_entries_ask_exactly_the_suite_question():
    suites = {s.name: s for s in bench.load_suites()}
    for e in ENTRIES:
        if e.bench:
            assert e.bench in suites, f"{e.name} names a missing suite {e.bench}"
            assert e.question == suites[e.bench].question


def test_bench_backed_measured_values_match_the_latest_scorecard():
    cards = bench.scorecards(SCORECARDS) if SCORECARDS.is_dir() else []
    if not cards:
        pytest.skip("no scorecard committed yet")
    latest = json.loads(cards[-1].read_text())
    for e in ENTRIES:
        if e.bench and e.bench in latest["suites"]:
            value = latest["suites"][e.bench]["value"]
            assert f"{value:.2f}" in e.measured, (
                f"{e.name} says {e.measured!r}; {cards[-1].name} has {value:.2f}")


def test_a_library_name_loads_as_a_bank():
    bank = inputs.load_questions("is_instruction")
    assert list(bank) == ["is_instruction"]
    assert bank["is_instruction"]["type"] == "noul"


def test_several_names_load_as_one_bank():
    bank = inputs.load_questions("is_instruction,touches_secret")
    assert list(bank) == ["is_instruction", "touches_secret"]


def test_an_unknown_name_in_a_list_is_an_error():
    with pytest.raises(ValueError, match="nope"):
        inputs.load_questions("is_instruction,nope")


def test_questions_lists_every_entry(capsys):
    assert cli.main(["questions"]) == 0
    out = capsys.readouterr().out
    for e in ENTRIES:
        assert e.name in out


def test_questions_name_prints_the_bank(capsys):
    assert cli.main(["questions", "touches_secret"]) == 0
    assert json.loads(capsys.readouterr().out)["touches_secret"]["type"] == "noul"


# ---- the linter: FINDINGS' failed questions, as a check ----------------------------------------

FAILED = [
    "How hard is this for a language model?",                            # §25, AUC 0.48
    "Does this involve money, legal, medical or safety consequences?",   # §25, AUC 0.52
    "Could running `command` cause harm that is hard to undo?",          # §29, AUC 0.53
]
WORKED = [e.question["instructions"] for e in ENTRIES] + [
    "Would `command` delete data?",
    "Does `post` threaten violence, harm or intimidation?",   # harm as content, not consequence
]


@pytest.mark.parametrize("text", FAILED)
def test_lint_flags_questions_about_consequences_or_difficulty(text):
    assert library.lint({"q": {"type": "noul", "instructions": text}})


@pytest.mark.parametrize("text", WORKED)
def test_lint_passes_questions_about_the_surface(text):
    assert library.lint({"q": {"type": "noul", "instructions": text}}) == []


@pytest.mark.parametrize("text", ["Is `text` an ordinary personal message?",   # §33, AUC 0.51
                                  "Is `text` an ordinary request?",            # §33, AUC 0.63
                                  "Is this a normal message?"])
def test_lint_flags_a_yes_no_about_the_absence_of_a_property(text):
    warnings = library.lint({"q": {"type": "noul", "instructions": text}})
    assert warnings and "for the property itself" in warnings[0]


def test_ordinary_as_a_choice_option_is_fine():
    """The same word worked as a named choice option (§33, 0.94 to 0.98): only a yes/no fails."""
    q = {"type": "choice", "instructions": "Is `text` spam?",
         "criteria": {"spam": "advertising", "ham": "an ordinary personal message"}}
    assert library.lint({"q": q}) == []


def test_refusal_names_the_opt_out_of_the_front_door():
    assert "--allow-unmeasured" in library.refusal(["q is bad"])
    assert "?allow_unmeasured=true" in library.refusal(["q is bad"], opt_out="?allow_unmeasured=true")
