"""Measuring answer quality: `verdict bench` (gates a release), `verdict rank` and
`verdict calibrate`."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import inference, support


def _bench_cmd(args: argparse.Namespace) -> int:
    """Score the pinned public suites, or with --verify compare a scorecard with the previous one."""
    from importlib.metadata import version

    from .. import bench

    # A suite is scored exactly as pinned: `sst2-yesno` exists to track the plain yes/no that the
    # everyday rewrite works around, so rewriting it here would hide the thing it measures.
    args.yesno = args.allow_unmeasured = True
    # Suites are clipped on purpose, as users' states are; the warning is for a user's own input.
    args.warn_clipped = False
    if args.verify:
        path = Path(args.verify)
        current = json.loads(path.read_text())
        prev_path = Path(args.against) if args.against else bench.previous(path.parent,
                                                                             path.stem.lstrip("v"))
        prev = json.loads(prev_path.read_text()) if prev_path else None
        problems = bench.verify(current, prev)
        for line in problems:
            print(f"verdict bench: {line}", file=sys.stderr)
        against = prev_path.name if prev_path else "nothing (first scorecard)"
        print(f"{path.name} against {against}: "
              f"{'REGRESSED' if problems else 'ok'}", file=sys.stderr)
        return 1 if problems else 0

    try:
        suites = bench.load_suites(bench.SUITES_FILE)
        if args.suite:
            unknown = set(args.suite) - {s.name for s in suites}
            if unknown:
                raise ValueError(f"no suite {', '.join(sorted(unknown))}; there are: "
                                 f"{', '.join(s.name for s in suites)}")
            suites = [s for s in suites if s.name in args.suite]
        if args.per_class:
            for s in suites:
                s.per_class = args.per_class
    except ValueError as exc:
        return support._fail(str(exc))

    if args.systemone:
        from urllib.parse import urlparse

        host = urlparse(args.systemone).hostname or args.systemone
        key = os.environ.get("TYPESAFE_API_KEY") or (
            "local" if host in ("127.0.0.1", "localhost") else None)
        if key is None:
            return support._fail(f"--systemone {args.systemone} needs TYPESAFE_API_KEY; the datasets are "
                         "public, but each item is sent to that service and billed")
        ask = bench.systemone_asker(args.systemone, args.systemone_model, key)
        model_name = f"systemone:{args.systemone_model}@{host}"
        budget = None
        bits = None  # not ours to know
    else:
        asker = inference._Asker(args)
        ask = asker
        model_name = Path(str(asker.main_path)).name
        budget = asker.settings.prompt_token_budget
        bits = asker.settings.bits
    calls = 0
    askers = {}

    def paced(state, questions, lang=None):
        nonlocal calls
        if calls and args.pause:
            time.sleep(args.pause)
        calls += 1
        if lang and not args.systemone:
            # A suite in another language names its checkpoint; the rest use the served one.
            if lang not in askers:
                askers[lang] = inference._Asker(argparse.Namespace(**{**vars(args), "lang": lang}))
            return askers[lang](state, questions)
        return ask(state, questions)

    results = {}
    for s in suites:
        try:
            items = bench.sample(s, bench.fetch(s))
        except ValueError as exc:
            return support._fail(str(exc))
        print(f"{s.name}: asking {len(items)} items...", file=sys.stderr)
        try:
            r = results[s.name] = bench.score(
                s, items, lambda st, q, _lang=s.lang: paced(st, q, _lang))
        except ValueError as exc:
            return support._fail(f"{s.name}: {exc}")
        extra = (f"  target over 0.5: {r['target_over_half']:.2f}  cut {r['fit']['cut']:.4f}"
                 if r["metric"] == "auc" else f"  chance {r['chance']:.2f}"
                 + (f"  auc {r['auc']:.3f}" if "auc" in r else ""))
        print(f"{s.name:<12} {r['metric']} {r['value']:.3f} "
              f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  n={r['n']}{extra}")
    doc = {"version": version("verdict"), "model": model_name, "budget": budget, "bits": bits,
           "date": time.strftime("%Y-%m-%d"), "suites": results}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _rank_cmd(args: argparse.Namespace) -> int:
    """Rank a `decide --jsonl` run by a weighted sum of its named, rescaled answers."""
    from .. import rank

    if args.k < 0:
        return support._fail(f"-k {args.k}: give a count, or 0 for all")
    try:
        results = rank.rank(rank.load(args.scored), rank.parse_weights(args.weights))
    except ValueError as exc:
        return support._fail(str(exc))
    for r in results[:args.k] if args.k else results:
        if args.json:
            print(json.dumps(r))
            continue
        state = r["state"]
        text = next(iter(state.values()), "") if isinstance(state, dict) else state
        why = "  ".join(f"{d} {c:+.2f}" for d, c in
                        sorted(r["contributions"].items(), key=lambda kv: -abs(kv[1])))
        print(f"{r['score']:5.2f}  {why}  | {str(text)[:70]}")
    return 0


def _calibrate_cmd(args: argparse.Namespace) -> int:
    """Ask a bank of every labelled state, fit a cut or temperature per question, report held-out."""
    from .. import calibrate, inputs

    try:
        questions = inputs.load_questions(args.questions)
    except ValueError as exc:
        return support._fail(str(exc))
    rows = [json.loads(line) for line in Path(args.examples).read_text().splitlines() if line.strip()]
    if args.limit and len(rows) > args.limit:
        import random

        rows = random.Random(args.seed).sample(rows, args.limit)
    if len(rows) < 10:
        return support._fail(f"{len(rows)} labelled examples is too few to fit anything; aim for 100+")

    def labels_of(row: dict) -> dict:
        if "labels" in row:
            return row["labels"]
        if "label" in row and len(questions) == 1:
            return {next(iter(questions)): row["label"]}
        raise ValueError("each line needs \"labels\": {question: value}, or \"label\" when the "
                         "bank has a single question")

    ask = inference._Asker(args)
    # Calibrating is how an unmeasured question gets measured (§29 found harm at chance this way),
    # so the shapes `ask` and `decide` refuse are only warned about here.
    ask.allow_unmeasured = True
    answers = []
    print(f"asking {len(questions)} question(s) of {len(rows)} states "
          f"({args.pause:.2f} s between calls)...", file=sys.stderr)
    for i, row in enumerate(rows):
        if i and args.pause:
            time.sleep(args.pause)
        answers.append(ask(row["state"], questions)["answers"])

    fits = {}
    try:
        for qid, q in questions.items():
            pairs = [(a[qid], labels_of(r).get(qid)) for a, r in zip(answers, rows)]
            pairs = [(a, y) for a, y in pairs if y is not None]
            if q["type"] == "noul":
                fits[qid] = calibrate.fit_noul([a["noul"] for a, _ in pairs],
                                               [calibrate.as_bool(y) for _, y in pairs],
                                               args.heldout, args.seed)
            elif q["type"] == "choice":
                fits[qid] = calibrate.fit_choice([a["probabilities"] for a, _ in pairs],
                                                 [str(y) for _, y in pairs], args.heldout, args.seed)
            else:
                ask.warn(f"{qid} is a score question; there is no single cut to fit, skipped")
    except ValueError as exc:
        return support._fail(str(exc))

    for qid, fit in fits.items():
        held = fit["heldout"]
        if fit["type"] == "noul":
            # Printed rounded for a person; the --out file keeps the exact cut that was scored.
            print(f"{qid}: cut {fit['cut']:.4f}  held-out AUC {held['auc']}  balanced accuracy "
                  f"{held['balanced_accuracy']}  (n={fit['n']}, {fit['positives']} positive)")
        else:
            print(f"{qid}: temperature {fit['temperature']}  held-out accuracy {held['accuracy']}  "
                  f"NLL {held['nll_raw']} -> {held['nll_fitted']}  (n={fit['n']})")
            print("  recall by option (all examples): " + "  ".join(
                f"{k} {v['recall']:.2f}" for k, v in
                sorted(fit["per_option"].items(), key=lambda kv: -kv[1]["recall"])))
            if fit["confusions"]:
                print("  most confused: " + ", ".join(
                    f"{c['true']}->{c['predicted']} x{c['n']}" for c in fit["confusions"][:3]))
    doc = {"model": answers and ask.main_path, "examples": args.examples, "questions": fits}
    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {args.out}; apply it with --calibration {args.out} (decide) "
              f"or --cut {args.out} (ask)", file=sys.stderr)
    else:
        print(json.dumps(doc, indent=2))
    return 0
