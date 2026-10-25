"""A bank of typed questions about a state: `verdict decide` and `verdict ask`."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import inference, support
from ..backends.engine import maybe_truncated


def _read_fits(path: str | None) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    return data.get("questions", data)


def _decide_cmd(args: argparse.Namespace) -> int:
    """State plus a bank of typed questions in, one answer per question out, in one pass."""
    from ..judge import calibrate
    from .. import inputs
    from ..schema import laya_questions

    if args.jsonl and not args.questions:
        return support._fail("--jsonl reads states from stdin, so --questions is required")
    if args.jsonl and args.state is not None:
        return support._fail("--jsonl reads states from stdin, one per line; drop the state "
                             "argument, or drop --jsonl to ask about this one state")
    if args.jsonl and getattr(sys.stdin, "isatty", lambda: False)():
        return support._fail("--jsonl reads states from stdin, and stdin is a terminal; pipe or "
                             "redirect a file: verdict decide --jsonl -q BANK < states.jsonl")
    try:
        if args.questions:
            questions = inputs.load_questions(args.questions)
        else:
            bank = sys.stdin.read()
            if not bank.strip():
                raise ValueError("no --questions, and no bank on stdin; pass -q PRESET|NAMES|"
                                 "JSON|FILE (see `verdict questions`)")
            questions = json.loads(bank)
        questions = laya_questions(questions)
        fits = _read_fits(args.calibration)
        if args.jsonl:
            # A generator, not a list: a live pipe never reaches EOF, and each line is answered
            # as it arrives. Each state keeps its input line number, blank lines included.
            states = _jsonl_states(sys.stdin, warn=lambda line: print(f"verdict: {line}",
                                                                      file=sys.stderr))
        elif args.state is not None:
            states = [(None, inputs.read_state(args.state))]
        else:
            return support._fail("give a state, or --questions and --jsonl with states on stdin")
    except ValueError as exc:
        return support._fail(exc)

    ask = inference._Asker(args)
    # A long criterion is cut mid-sentence with no error. The character check is free; --check
    # pays about 1.3 s for the tokenizer to count exactly.
    warnings = (inference.load(ask.main_path).option_overflow(questions) if args.check
                else maybe_truncated(questions))
    for line in warnings:
        ask.warn(line)

    from .. import client

    # Per yes/no question: count, low, high. Running, not a list: a live pipe never ends.
    seen: dict[str, tuple[int, float, float]] = {}
    index, answered = None, 0
    try:
        for i, (index, state) in enumerate(states):
            if i and args.pause:
                time.sleep(args.pause)
            result = ask(state, questions)
            answered += 1
            if fits:
                result["answers"] = calibrate.apply(result["answers"], fits)
            for qid, a in result["answers"].items():
                if a.get("type") == "noul":
                    n, lo, hi = seen.get(qid, (0, a["noul"], a["noul"]))
                    seen[qid] = (n + 1, min(lo, a["noul"]), max(hi, a["noul"]))
            if args.jsonl:
                print(json.dumps({"index": index, "state": state, **result}), flush=True)
            else:
                print(json.dumps(result, indent=2))
    except (ValueError, client.NoServer, client.ServerError, client.ServerTimeout) as exc:
        # In --jsonl, name the line that failed so a caller can resume after it.
        failed = getattr(exc, "index", index) if args.jsonl else None
        return support._fail(exc, **({"index": failed} if failed is not None else {}))
    if args.jsonl and not answered:
        return support._fail("no states on stdin; pipe one JSON state (or text) per line")
    if args.jsonl:
        for line in _spread_report(seen):
            print(f"verdict: {line}", file=sys.stderr)
    return 0


def _jsonl_states(lines, warn):
    """(index, state) per non-blank line; `index` is the 0-based input line, so output rows
    pair with input lines even across skipped blanks. A line that starts like JSON must parse
    (ValueError names the line); plain text is sent as text, with one warning."""
    from .. import inputs

    warned = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            state = inputs.parse_state(line)
        except ValueError as exc:
            err = ValueError(f"line {index + 1}: {exc}")
            err.index = index
            raise err from exc
        if isinstance(state, str) and not warned:
            warned = True
            warn(f"line {index + 1} is not JSON; sending it, and any later such line, as text")
        yield index, state


#: Below this range across a batch, a yes/no question ranks almost nothing. A status triage's
#: `blocked` question scored 0.58 to 0.65 over 137 rooms while it was reading linter
#: boilerplate, and read 0.12 to 0.63 once the inputs were fixed (FINDINGS §30).
NARROW_SPREAD = 0.1
SPREAD_MIN_STATES = 10


def _spread_report(seen: dict[str, tuple[int, float, float]]) -> list[str]:
    """One line per yes/no question: its range over the batch, flagged when too narrow to rank.
    `seen` maps each question to `(count, low, high)`.

    The scores are uncalibrated, so a batch is only useful for its ordering, and an ordering
    squeezed into a few hundredths is noise. Usually the inputs are at fault (all near-identical,
    or all boilerplate), not the model, so the fix is to look at them.
    """
    out = []
    for qid, (n, lo, hi) in seen.items():
        if n < SPREAD_MIN_STATES:
            continue
        line = f"spread {qid}: {lo:.2f} to {hi:.2f} over {n} states"
        if hi - lo < NARROW_SPREAD:
            line += (" -- narrow range, it ranks almost nothing; check the inputs "
                     "(near-identical? boilerplate?) before trusting the order")
        out.append(line)
    return out


def _ask_cmd(args: argparse.Namespace) -> int:
    """One question, inline. Prints one line; `--json` for everything."""
    from ..judge import calibrate
    from .. import inputs

    try:
        question = inputs.inline_question(args.question, args.options, args.levels,
                                          args.true, args.false)
        cut, fit = None, None
        if args.cut is not None and (args.options or args.levels):
            try:
                float(args.cut)
            except ValueError:
                pass  # a calibrate file: for a choice it applies the fitted temperature
            else:
                raise ValueError("a numeric --cut applies to a yes/no; a choice or score has no "
                                 "single cut (a `verdict calibrate` file still applies)")
        if args.cut is not None:
            try:
                cut = float(args.cut)
            except ValueError:
                if not Path(args.cut).is_file():
                    raise ValueError(f"--cut {args.cut} is neither a number between 0 and 1 nor a "
                                     "file from `verdict calibrate`") from None
                fits = _read_fits(args.cut)
                fit = fits.get("q") or (next(iter(fits.values())) if len(fits) == 1 else None)
                if fit is None:
                    raise ValueError(f"{args.cut} holds more than one fit and none named 'q'")
                cut = fit.get("cut")
            if cut is not None and not 0 <= cut <= 1:
                raise ValueError(f"--cut {args.cut}: a cut is a probability, between 0 and 1")
        state = inputs.read_state(args.text)
    except (ValueError, OSError) as exc:
        return support._fail(exc)

    result = inference._Asker(args)(state, {"q": question})
    if fit:
        result["answers"] = calibrate.apply(result["answers"], {"q": fit})
    answer = result["answers"]["q"]

    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    if answer["type"] == "choice":
        # The chosen option's probability, calibrated when a fit is given. Not laya's `confidence`,
        # which for a choice is 1 - H(p)/log(k): a 0.50/0.50 split would print "A 0.00".
        p = answer.get("calibrated", answer["probabilities"])[answer["choice"]]
        print(f"{answer['choice']} {p:.2f}")
        return 0
    if answer["type"] == "score":
        top = max(answer["probabilities"], key=answer["probabilities"].get)
        print(f"{answer['score']:.2f}/{len(answer['legend']) - 1} {answer['legend'][top]}")
        return 0
    score = answer["noul"]
    if cut is None:
        print(f"{score:.2f}")
        return 0
    yes = score >= cut
    shown = f"{score:.2f}"
    if (float(shown) >= cut) != yes:
        # Rounded, the score would read as the other side of the cut ("no 0.50" at 0.5).
        shown = f"{score:.4f}"
    print(f"{'yes' if yes else 'no'} {shown}")
    return 0 if yes else 1


def _classify_cmd(args: argparse.Namespace) -> int:
    """A text into one leaf of a two-level tree, beyond the 20-option flat default."""
    from ..judge import hierarchy
    from .. import inputs

    try:
        tree = json.loads(Path(args.tree).read_text())
        state = inputs.read_state(args.text)
    except json.JSONDecodeError as exc:
        return support._fail(f"{args.tree} is not valid JSON: {exc}")
    except (ValueError, OSError) as exc:
        return support._fail(exc)

    result = hierarchy.classify(inference._Asker(args), state, tree, beam=args.beam)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return 0
    print(f"{result.label} {result.probability:.2f}  ({' > '.join(result.path)})")
    return 0


def _cascade_cmd(args: argparse.Namespace) -> int:
    """Laya answers every state; the answers it is least sure of are asked again of a second server.

    The second server sees only the escalated questions of the escalated states, and the summary says
    how many that was. `--check` measures, on labelled examples, whether that helps for this task."""
    from ..judge import calibrate, cascade
    from .. import client, config, inputs
    from ..schema import laya_questions

    if args.share is not None and args.below is not None:
        return support._fail("give --share or --below, not both: --share escalates that fraction of "
                             "each question's answers (the least sure), --below escalates the "
                             "answers less sure than a fixed confidence")
    share, below = args.share, args.below
    if share is None and below is None:
        share = 0.3
    try:
        questions = laya_questions(inputs.load_questions(args.questions))
        second_url = config.check_url(args.escalate, "--escalate")
    except ValueError as exc:
        return support._fail(exc)

    if args.fit and not args.check:
        return support._fail("--fit fits a cut on labelled examples, so it needs --check LABELLED.jsonl")
    if args.out and not (args.check and args.fit):
        return support._fail("--out saves the cuts that --check --fit finds; add both")
    if args.calibration and below is not None:
        return support._fail("a fitted cut measures the unsure as a distance in logit units, not a "
                             "probability: use --share with --calibration, not --below")
    try:
        fits = _read_fits(args.calibration)
    except (OSError, ValueError) as exc:
        return support._fail(f"{args.calibration}: {exc}")

    primary = inference._Asker(args)
    flags = {"allow_unmeasured": True} if args.allow_unmeasured else {}
    errors = (client.NoServer, client.ServerError, client.ServerTimeout)

    def secondary(state, qs):
        return client.decide(state, qs, second_url, **flags)

    if args.check:
        try:
            rows = [json.loads(line) for line in Path(args.check).read_text().splitlines() if line.strip()]
            states = [r["state"] for r in rows]
            labels = []
            for i, r in enumerate(rows):
                lab = r.get("labels") or ({next(iter(questions)): r["label"]}
                                         if "label" in r and len(questions) == 1 else None)
                if not isinstance(lab, dict):
                    raise ValueError(f"example {i + 1} needs \"labels\": {{question: value}}")
                labels.append(lab)
        except (KeyError, ValueError, OSError) as exc:
            return support._fail(f"{args.check}: {exc}")
        report = cascade.check(states, questions, labels, primary, secondary, share=share,
                               below=below, fits=fits, holdout=0.3 if args.fit else None,
                               errors=errors)
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"questions": {q: r["fit"] for q, r in report.items() if r["fit"]}}, indent=2) + "\n")
            print(f"verdict: wrote the fitted cuts to {args.out}; use them with --calibration",
                  file=sys.stderr)
        if args.json:
            print(json.dumps(report, indent=2))
            return 0
        held = " (held out)" if args.fit else ""
        for qid, r in report.items():
            at = (f"primary {r['primary']:.3f} at cut {r['cut']:.3f} (at 0.5: {r['primary_raw']:.3f})"
                  if r["cut"] is not None else f"primary {r['primary']:.3f} at 0.5")
            print(f"{qid}{held}: {at}  cascade {r['cascade']:.3f} "
                  f"({r['escalated']} of {r['n']} escalated)  second {r['secondary']:.3f}")
        if args.fit:
            print("fitted on 70% of the examples; the numbers above are the 30% held out")
        no = [q for q, r in report.items() if not r["helps"]]
        if no:
            print(f"escalating would not help for: {', '.join(no)}. The second backend does not beat "
                  "the primary on these examples, so sending text out buys nothing (FINDINGS §52).")
        else:
            print("escalating helps for every question checked.")
        return 0

    if getattr(sys.stdin, "isatty", lambda: False)():
        return support._fail("cascade reads states from stdin, one JSON state per line, and stdin "
                             "is a terminal; pipe or redirect a file: verdict cascade -q BANK "
                             "--escalate URL < states.jsonl")
    try:
        numbered = list(_jsonl_states(sys.stdin, warn=lambda line: print(f"verdict: {line}",
                                                                         file=sys.stderr)))
    except ValueError as exc:
        return support._fail(exc)
    if not numbered:
        return support._fail("no states on stdin; pipe one JSON state (or text) per line")
    result = cascade.run([state for _, state in numbered], questions, primary, secondary,
                         share=share, below=below, fits=fits, errors=errors)
    for (index, state), row in zip(numbered, result.rows):
        print(json.dumps({"index": index, "state": state, **row}), flush=True)
    kept = len(numbered) - result.states_sent
    print(f"verdict: escalated {result.escalated} of {result.answers} answers to {second_url}: "
          f"{result.states_sent} of {len(numbered)} states were sent there, {kept} stayed on the "
          "primary", file=sys.stderr)
    if result.failed:
        print(f"verdict: {len(result.failed)} state(s) could not be escalated and kept the primary "
              f"answer (first: {result.failed[0][1]})", file=sys.stderr)
        return 1
    return 0
