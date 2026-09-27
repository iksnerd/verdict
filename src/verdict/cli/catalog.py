"""Reading the question and preset catalog: `verdict validate`, `verdict questions` and
`verdict presets`. No model is loaded."""
from __future__ import annotations

import argparse
import json
import sys

from . import inference, support
from ..engine import maybe_truncated
from ..errors import QuestionError

#: What this project has measured about a preset, shown where the preset is listed.
PRESET_NOTES = {
    "router": "difficulty and is_sensitive are at chance on real agent traffic here "
              "(FINDINGS §25); a surface question such as \"Is this an instruction?\" is not",
}


def _validate_cmd(args: argparse.Namespace) -> int:
    """Check and normalize a bank without settings, network calls or model loading."""
    from .. import inputs, library
    from ..schema import laya_questions

    try:
        questions = laya_questions(json.load(sys.stdin) if args.questions == "-"
                                   else inputs.load_questions(args.questions))
        problems = library.lint(questions)
        if problems and not args.allow_unmeasured:
            raise QuestionError(inference._refusal(problems))
        warnings = problems + maybe_truncated(questions)
    except (ValueError, OSError) as exc:
        if args.json:
            print(json.dumps({"valid": False, "error": str(exc)}))
            return 2
        return support._fail(exc)
    if args.json:
        print(json.dumps({"valid": True, "questions": questions, "warnings": warnings}, indent=2))
    else:
        print(f"Valid: {len(questions)} question(s)")
        for warning in warnings:
            print(f"verdict: {warning}", file=sys.stderr)
    return 0


def _questions_cmd(args: argparse.Namespace) -> int:
    """The measured-question library: what each asks, on what state, and how it did."""
    from .. import library

    entries = library.load()
    if args.json and not args.name:
        print(json.dumps({e.name: {"question": e.question, "state": e.state,
                                   "measured": e.measured, "source": e.source}
                          for e in entries}, indent=2))
        return 0
    if args.name:
        entry = next((e for e in entries if e.name == args.name), None)
        if entry is None:
            names = [e.name for e in entries]
            return support._fail(f"no library question {args.name!r}"
                                 f"{support._suggest(args.name, names)} There are: "
                                 f"{', '.join(names)}")
        print(json.dumps({entry.name: entry.question}, indent=2))
        return 0
    for e in entries:
        print(f"{e.name:17s} {e.question['type']:6s} {e.question['instructions']}")
        print(f"{'':17s} state {e.state}")
        print(f"{'':17s} {e.measured} ({e.source})")
        print()
    print("Use one or several by name:  verdict decide STATE -q is_instruction,touches_secret")
    print("Numbers are for the state they were measured on; on your own data, rank or run "
          "`verdict calibrate` before gating.")
    return 0


def _presets_cmd(args: argparse.Namespace) -> int:
    from .. import inputs

    try:
        if args.json and not args.name:
            print(json.dumps({name: inputs.preset(name) for name in inputs.PRESETS}, indent=2))
            return 0
        if args.name:
            print(json.dumps(inputs.preset(args.name), indent=2))
            return 0
        for name in inputs.PRESETS:
            bank = inputs.preset(name)
            fields = ", ".join(f"`{f}`" for f in inputs.fields_named(bank)) or "any"
            print(f"{name:11s} state field {fields}")
            if name in PRESET_NOTES:
                print(f"  note: {PRESET_NOTES[name]}")
            for qid, q in bank.items():
                print(f"  {qid:20s} {q['type']:6s} {q['instructions']}")
            print()
    except ValueError as exc:
        return support._fail(exc)
    print("Print one as JSON to edit it:  verdict presets guard > my-bank.json")
    return 0
