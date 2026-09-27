"""`verdict examples`: runs one or more examples/ banks against a backend and prints each row's
answers -- the same real inputs examples/README.md's own numbers came from, so a new backend
(--systemone URL) can be checked against them instead of one-off questions."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import inference, support


def _examples_root() -> Path:
    """The wheel packs examples/ at `verdict/_examples` (force-include in pyproject.toml); a
    checkout reads it in place, so `uv run verdict examples` shows what is being edited."""
    packed = Path(__file__).resolve().parents[1] / "_examples"
    if packed.is_dir():
        return packed
    checkout = Path(__file__).resolve().parents[3] / "examples"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("examples/ is not in this install; run from a checkout of "
                            "https://github.com/iksnerd/verdict")


def _names(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if (p / "bank.json").is_file())


def _rows(example_dir: Path):
    for f in sorted(example_dir.glob("*.jsonl")):
        if "labelled" in f.name:
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                yield json.loads(line)


def _format(answers: dict) -> str:
    parts = []
    for qid, a in answers.items():
        t = a.get("type")
        if t == "noul":
            parts.append(f"{qid}={a['noul']:.2f}")
        elif t == "choice":
            # The chosen option's probability, labelled: laya's `confidence` for a choice is
            # 1 - H(p)/log(k), which read beside the option looks like one and is not.
            parts.append(f"{qid}={a['choice']}(p={a['probabilities'][a['choice']]:.2f})")
        elif t == "score":
            parts.append(f"{qid}={a['score']:.2f}")
        else:
            parts.append(f"{qid}=?")
    return " ".join(parts)


def _correct(answer: dict, expected) -> bool:
    """Whether an answer matches a row's recorded `expected` label. `noul` reads `>= 0.5` as the
    same convention `examples/README.md` and `verdict rank` already use for a yes/no; `choice`
    compares the chosen option; `score` compares the most likely level. A question type this
    doesn't recognize (a future answer type) never counts as correct, on purpose: silently
    skipping it would inflate the score of whatever the CLI does understand."""
    t = answer.get("type")
    if t == "noul":
        return (answer["noul"] >= 0.5) == bool(expected)
    if t == "choice":
        return answer["choice"] == expected
    if t == "score":
        return max(answer["probabilities"], key=answer["probabilities"].get) == str(expected)
    return False


def _systemone_ask(args: argparse.Namespace):
    from urllib.parse import urlparse

    from .. import bench

    host = urlparse(args.systemone).hostname or args.systemone
    key = os.environ.get("TYPESAFE_API_KEY") or ("local" if host in ("127.0.0.1", "localhost")
                                                 else None)
    if key is None:
        raise ValueError(f"--systemone {args.systemone} needs TYPESAFE_API_KEY; the examples are "
                         "invented, but each row is sent to that service and billed")
    return bench.systemone_asker(args.systemone, args.systemone_model, key)


def _examples_cmd(args: argparse.Namespace) -> int:
    """Score examples/ banks with a backend, printing each row's answers."""
    try:
        root = _examples_root()
    except FileNotFoundError as exc:
        return support._fail(exc)

    available = _names(root)
    names = args.name or available
    unknown = [n for n in names if n not in available]
    if unknown:
        hint = support._suggest(unknown[0], available) if len(unknown) == 1 else "."
        return support._fail(f"no example {', '.join(unknown)}{hint} There are: "
                             f"{', '.join(available)}")
    if args.path:
        for name in args.name or []:
            print(root / name)
        return 0 if args.name else support._fail("--path needs an example NAME")

    try:
        ask = _systemone_ask(args) if args.systemone else inference._Asker(args)
    except ValueError as exc:
        return support._fail(exc)

    try:
        calls = 0
        for name in names:
            bank = json.loads((root / name / "bank.json").read_text())
            print(f"=== {name} ===")
            correct = scored = 0
            latencies_ms: list[float] = []
            for i, row in enumerate(_rows(root / name)):
                if calls and args.pause:
                    time.sleep(args.pause)
                calls += 1
                state = row.get("state", row)
                started = time.perf_counter()
                try:
                    result = ask(state, bank)
                except ValueError as exc:
                    return support._fail(f"{name} row {i}: {exc}")
                latencies_ms.append((time.perf_counter() - started) * 1000)
                print(_format(result["answers"]), "|", json.dumps(state)[:90])
                for qid, expected in (row.get("expected") or {}).items():
                    answer = result["answers"].get(qid)
                    if answer is None:
                        continue
                    scored += 1
                    correct += _correct(answer, expected)
            if scored:
                print(f"{name}: {correct}/{scored} match the recorded expectation")
            print(f"{name}: {sum(latencies_ms) / len(latencies_ms):.0f} ms/call over "
                  f"{len(latencies_ms)} calls ({min(latencies_ms):.0f} to {max(latencies_ms):.0f} ms)")
        return 0
    finally:
        if args.systemone:
            ask.close()
