"""The CLI's help is the first thing an agent reads, so its examples must not teach a question
shape FINDINGS measured as failing. `calibrate`'s examples used "Could `command` cause harm?", the
wording §29 put at chance (0.53), after the linter existed to catch it."""
from __future__ import annotations

import re

import pytest

from verdict import cli, library

HELP = {name: getattr(cli, name) for name in dir(cli)
        if name.endswith(("EPILOG", "HELP")) and isinstance(getattr(cli, name), str)}

#: `verdict ask "<text>" "<question>"` and `"instructions": "<question>"` in the help text.
_ASKED = re.compile(r'verdict ask \S+ "([^"]+)"|verdict ask "[^"]*" "([^"]+)"|"instructions": "([^"]+)"')


def questions():
    for name, text in HELP.items():
        for m in _ASKED.finditer(text):
            yield name, next(g for g in m.groups() if g)


def test_the_help_has_examples_to_check():
    assert len(list(questions())) >= 8


@pytest.mark.parametrize("where,question", list(questions()))
def test_no_help_example_asks_a_question_shape_that_failed(where, question):
    kind = "choice" if " -o " in HELP[where] and question.startswith("What") else "noul"
    assert library.lint({"q": {"type": kind, "instructions": question}}) == [], (where, question)


def test_decide_help_names_the_library():
    text = cli.DECIDE_EPILOG
    assert "verdict questions" in text and "is_instruction" in text


def test_ask_help_says_a_new_yes_no_is_asked_as_a_no_yes_choice():
    assert "§38" in cli.ASK_HELP and "--yesno" in cli.ASK_HELP


#: `-q NAME` or `--questions NAME` in an example, where NAME is a preset or library question.
_NAMED = re.compile(r"(?:-q|--questions) ([a-z_,]+)\b")


def named_banks():
    from verdict import inputs

    known = set(inputs.PRESETS) | {e.name for e in library.load()}
    for name, text in HELP.items():
        for m in _NAMED.finditer(text):
            if set(m.group(1).split(",")) <= known:
                yield name, m.group(1)


@pytest.mark.parametrize("where,bank", list(named_banks()))
def test_every_bank_a_help_example_names_is_accepted_as_is(where, bank, capsys):
    """An example that `decide` refuses teaches the refused thing: `-q guard` stayed in the help
    after guard's `harm_severity` (what harm the text could cause) began to be refused."""
    pytest.importorskip("laya_mlx")
    assert cli.main(["validate", "-q", bank]) == 0, (where, bank, capsys.readouterr().err)


@pytest.mark.parametrize("text", ["EPILOG", "ASK_HELP", "DECIDE_HELP"])
def test_the_help_says_bad_shapes_are_refused_not_warned_about(text):
    body = getattr(cli, text)
    assert "--allow-unmeasured" in body and "refuse" in body, text


def test_serve_help_mentions_the_jev_protocol():
    assert "/v1/systemone" in cli.SERVE_EPILOG and "TYPESAFE_BASE_URL" in cli.SERVE_EPILOG


@pytest.mark.parametrize("command", ["validate", "ask", "decide", "rank", "calibrate", "bench",
                                     "update", "route"])
def test_every_command_a_script_can_branch_on_states_its_exit_status(command, capsys):
    """`bench --verify` gates releases on its exit code and its help did not say so."""
    with pytest.raises(SystemExit):
        cli.main([command, "--help"])
    assert "Exit status" in capsys.readouterr().out, command


COMMANDS = ["validate", "ask", "decide", "rank", "docs", "questions", "presets", "calibrate",
            "bench", "examples", "route", "cases", "serve", "update", "config", "init"]


def rendered(command, capsys):
    with pytest.raises(SystemExit):
        cli.main([command, "--help"] if command else ["--help"])
    return capsys.readouterr().out


@pytest.mark.parametrize("command", [None, *COMMANDS])
def test_no_prose_line_runs_past_100_columns(command, capsys):
    """The raw formatter keeps a description's own line breaks, so one written as a single string
    printed as one 400-column line on eight commands. Example command lines may run long."""
    long = [line for line in rendered(command, capsys).splitlines()
            if len(line) > 100 and not line.lstrip().startswith(("verdict ", "echo ", "git "))]
    assert not long, long


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_shows_an_example_and_its_exit_status(command, capsys):
    out = rendered(command, capsys)
    assert "Example" in out and "Exit status" in out, command


def test_the_top_level_help_states_the_limits(capsys):
    """An agent reads this first; §41 and §46 are where it should not reach for verdict."""
    out = rendered(None, capsys)
    assert "§46" in out and "compare numbers" in out


def test_route_help_says_json_exits_zero(capsys):
    assert "--json" in cli.ROUTE_EPILOG.split("Exit status")[1]
