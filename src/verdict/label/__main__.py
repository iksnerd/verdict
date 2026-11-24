"""Label synthetic states with Gemini teachers and report teacher agreement (the accuracy ceiling).

    uv run python -m verdict.label --states data/a.jsonl data/b.jsonl \\
        --key-file .env --max-cost 1.0
"""
import argparse
import itertools
import json
import os
from collections import Counter
from pathlib import Path

from .bank import BANK
from .teacher import DEFAULT_CHEAP_MODELS, DEFAULT_REFERENCE_MODEL, LabelError, Teacher, UnpricedModelError, agreement, gemini_transport, load_key, mean_labels, price_of

CHEAP = list(DEFAULT_CHEAP_MODELS)
REFERENCE = DEFAULT_REFERENCE_MODEL


def load_states(paths):
    seen, out = set(), []
    for p in paths:
        for line in open(p):
            c = json.loads(line)
            if c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)
    return out


def mean_agree(pairs):
    per_q = {qid: [] for qid in BANK}
    for a, b in pairs:
        for qid, v in agreement(a, b).items():
            per_q[qid].append(v)
    return {qid: (sum(v) / len(v) if v else float("nan")) for qid, v in per_q.items()}


def fmt(d):
    return "  ".join(f"{k}={v:.2f}" for k, v in d.items()) + f"   | mean={sum(d.values()) / len(d):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", required=True, type=Path)
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--models", nargs="+", default=CHEAP)
    ap.add_argument("--reference", default=REFERENCE)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-cost", type=float, default=1.0, help="hard USD cap; stops before exceeding it")
    ap.add_argument("--out", type=Path, default=Path("data/labels_pilot.jsonl"))
    a = ap.parse_args()

    transport = gemini_transport(load_key(a.key_file))
    teachers = {m: Teacher(m, transport) for m in [*a.models, a.reference]}
    try:
        for m in teachers:
            price_of(m)  # an unpriced model would count as $0 and the cap could never trip
    except UnpricedModelError as e:
        raise SystemExit(str(e)) from None
    states = load_states(a.states)
    print(f"{len(states)} states; teachers {list(teachers)}; repeats {a.repeats}; cap ${a.max_cost}")

    rows, failures = [], Counter()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    # One line per state, written as it is finished: an interrupted run keeps every label paid for.
    with a.out.open("w") as out:
        for i, case in enumerate(states):
            row = {"id": case["id"], "intended_type": case["intended_type"], "labels": {}}
            try:
                for m, t in teachers.items():
                    reps = 1 if m == a.reference else a.repeats
                    for r in range(reps):
                        if sum(x.cost_usd() for x in teachers.values()) > a.max_cost:
                            raise SystemExit(f"cost cap ${a.max_cost} reached at state {i}; stopping")
                        try:
                            row["labels"].setdefault(m, []).append(t.label(case["state"], BANK))
                        except LabelError as e:
                            failures[m] += 1
                            print(f"  [{m}] state {i}: {e}")
            finally:
                out.write(json.dumps(row) + "\n")
                out.flush()
                os.fsync(out.fileno())
            rows.append(row)
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{len(states)}  spent ${sum(x.cost_usd() for x in teachers.values()):.3f}", flush=True)

    print("\n== self-agreement (run 1 vs run 2, same model) ==")
    for m in a.models:
        pairs = [(r["labels"][m][0], r["labels"][m][1]) for r in rows if len(r["labels"].get(m, [])) >= 2]
        print(f"{m:26s} n={len(pairs)}  {fmt(mean_agree(pairs))}")

    print("\n== cross-model agreement (first run of each) ==")
    for m1, m2 in itertools.combinations(a.models, 2):
        pairs = [(r["labels"][m1][0], r["labels"][m2][0]) for r in rows if r["labels"].get(m1) and r["labels"].get(m2)]
        print(f"{m1} vs {m2}  n={len(pairs)}  {fmt(mean_agree(pairs))}")

    print(f"\n== each cheap teacher (mean of runs) vs reference {a.reference} ==")
    for m in a.models:
        pairs = [
            (mean_labels(r["labels"][m]), r["labels"][a.reference][0])
            for r in rows
            if r["labels"].get(m) and r["labels"].get(a.reference)
        ]
        print(f"{m:26s} n={len(pairs)}  {fmt(mean_agree(pairs))}")

    print("\n== `kind` argmax vs the type the generator was ASKED to write (weak, generator is unreliable) ==")
    for m in [*a.models, a.reference]:
        hits = [
            max(mean_labels(r["labels"][m])["kind"].items(), key=lambda kv: kv[1])[0] == r["intended_type"]
            for r in rows
            if r["labels"].get(m)
        ]
        print(f"{m:26s} matches intended type {sum(hits)}/{len(hits)}")

    print("\n== spend ==")
    for m, t in teachers.items():
        print(f"{m:26s} calls={t.calls} tokens in/out={t.tokens_in}/{t.tokens_out}  ${t.cost_usd():.4f}")
    print(f"total ${sum(t.cost_usd() for t in teachers.values()):.4f}   failures: {dict(failures) or 'none'}")


main()
