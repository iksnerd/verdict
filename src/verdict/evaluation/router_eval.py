"""Scoring the big-vs-small switch against `evals/router_prompts.jsonl`.

This lives in the package so every caller (the refit command below, tests, any scorecard) runs the same
code: a score script imports the project's metric and never reimplements it, so a scorecard and
the repo cannot disagree about the number.

Two things here are deliberate and are the reason §12's 83.3% was worthless:

- **Thresholds are fitted on `train` and reported on `test`.** The split is written into the eval
  file rather than derived, so it cannot drift between runs or machines.
- **The fit minimises expected cost, not accuracy.** A hard prompt sent to the small model counts
  `COST_RATIO` times an easy one sent to the big model, the same asymmetry the base checkpoint
  carries in `rl_agent_config.json`. Fitting on accuracy scored four points higher on held-out
  prompts while making 5 expensive mistakes instead of 3 (FINDINGS §15).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .metrics import auc, bootstrap_ci
from ..judge.router import BIG, ROUTER_BANK, SMALL, T_DIFFICULTY, T_SENSITIVE, route

EVALS = Path("evals/router_prompts.jsonl")
#: Model outputs, keyed by model then prompt id. Derived and gitignored: the eval file is the
#: input, this is a replay cache so a threshold sweep or a scorecard rerun costs nothing and
#: returns the same numbers. Delete it to re-measure.
CACHE = Path("data/router_probs.json")
#: Base Laya, what an install runs. The private fine-tune is opt-in with --model; a default that
#: names weights most users do not have makes the refit fail for them.
DEFAULT_MODEL = "aac6fef/laya-mlx"

#: `difficulty` runs 0 to 3; `is_sensitive` is a probability.
D_GRID = [i / 20 for i in range(0, 61)]
S_GRID = [i / 50 for i in range(0, 51)]

#: How much worse a hard prompt sent to the small model is than an easy one sent to the big model.
#: Matches `rl_agent_config.json`: `act_costs.escalate` 0.5 against `cost_wrong_act` 3.0.
COST_RATIO = 3.0


def load_prompts(path: Path = EVALS) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def measure(rows: Sequence[dict], model: str = DEFAULT_MODEL, cache: Path = CACHE) -> tuple[dict, list[float]]:
    """Model outputs per prompt id, measuring only what the cache is missing."""
    store = json.loads(Path(cache).read_text()) if Path(cache).exists() else {}
    bucket = store.setdefault(model, {})
    todo = [r for r in rows if r["id"] not in bucket]
    times: list[float] = []
    if todo:
        import time

        from ..backends.engine import load

        eng = load(model)
        for r in todo:
            t0 = time.perf_counter()
            ans = eng.predict(eng.clip(r["prompt"]), ROUTER_BANK)
            times.append((time.perf_counter() - t0) * 1000)
            bucket[r["id"]] = {"difficulty": ans["difficulty"]["score"],
                               "is_sensitive": ans["is_sensitive"]["noul"]}
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(store, indent=1, sort_keys=True))
    return bucket, times


def as_answers(p: dict[str, float]) -> dict[str, Any]:
    """Cached model outputs back into the answer shape `route` reads.

    `probabilities` is filled with a non-flat placeholder on purpose: the router's `is_uniform`
    guard reads distributions for flatness, and a cache stores only the two scalars the policy
    uses, so a flat placeholder would make every replayed prompt look like "no model behind this".
    """
    return {
        "difficulty": {"type": "score", "score": p["difficulty"],
                       "probabilities": {"0": 0.4, "1": 0.3, "2": 0.2, "3": 0.1}},
        "is_sensitive": {"type": "noul", "noul": p["is_sensitive"]},
    }


def score(rows: Iterable[dict], probs: dict, td: float = T_DIFFICULTY, ts: float = T_SENSITIVE) -> dict:
    rows = list(rows)
    right = hard_to_small = easy_to_big = 0
    wrong: list[str] = []
    for r in rows:
        got = route(as_answers(probs[r["id"]]), t_difficulty=td, t_sensitive=ts).name
        if got == r["want"]:
            right += 1
        else:
            wrong.append(f"{r['id']}: wanted {r['want']}, got {got} :: {r['prompt'][:52]}")
            hard_to_small += r["want"] == BIG
            easy_to_big += r["want"] == SMALL
    n = len(rows)
    return {
        "n": n, "right": right, "accuracy": right / n if n else float("nan"),
        "hard_to_small": hard_to_small, "easy_to_big": easy_to_big,
        "cost": (hard_to_small * COST_RATIO + easy_to_big) / n if n else float("nan"),
        "wrong": wrong,
    }


def cost(rows: Iterable[dict], probs: dict, td: float, ts: float, ratio: float = COST_RATIO) -> float:
    """Expected cost per prompt. This, not accuracy, is what `fit` minimises."""
    s = score(rows, probs, td, ts)
    return (s["hard_to_small"] * ratio + s["easy_to_big"]) / s["n"]


def fit(rows: Sequence[dict], probs: dict, ratio: float = COST_RATIO) -> tuple[float, float, float]:
    """Lowest-cost (cost, t_difficulty, t_sensitive). Ties break toward the cautious pair."""
    best = (float("inf"), T_DIFFICULTY, T_SENSITIVE)
    for td in D_GRID:
        for ts in S_GRID:
            c = cost(rows, probs, td, ts, ratio)
            if (c, td, ts) < best:
                best = (c, td, ts)
    return best


def evaluate(rows: Sequence[dict], probs: dict, td: float = T_DIFFICULTY, ts: float = T_SENSITIVE) -> dict:
    """The full scorecard payload. `score` at the top level is what a scorecard reads."""
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    held_out = score(test, probs, td, ts)
    lo, hi = bootstrap_ci(test, lambda sample: score(sample, probs, td, ts)["accuracy"])
    labels = [1 if r["want"] == SMALL else 0 for r in rows]
    return {
        # Held-out accuracy as a share out of 100, so a scorecard has an integer perfect score.
        "score": round(100 * held_out["accuracy"], 2),
        "n_train": len(train), "n_test": len(test),
        "held_out": {k: v for k, v in held_out.items() if k != "wrong"},
        "held_out_ci_95": [round(100 * lo, 2), round(100 * hi, 2)],
        "train_accuracy": round(100 * score(train, probs, td, ts)["accuracy"], 2),
        "thresholds": {"difficulty": td, "is_sensitive": ts},
        "auc": {
            q: round(1 - auc([probs[r["id"]][q] for r in rows], labels), 4)
            for q in ROUTER_BANK
        },
        "mistakes": held_out["wrong"],
        "chance": 50.0,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """`python -m verdict.evaluation.router_eval [--refit] [--json FILE]`: score the switch on the held-out
    split, and with --refit print thresholds to paste into router.py."""
    import argparse
    import statistics as st

    ap = argparse.ArgumentParser(prog="python -m verdict.evaluation.router_eval")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--json", dest="json_out")
    args = ap.parse_args(argv)

    rows = load_prompts()
    probs, times = measure(rows, args.model)
    train = [r for r in rows if r["split"] == "train"]
    result = evaluate(rows, probs)

    print(f"model={args.model}  prompts={len(rows)} (train {result['n_train']}, test {result['n_test']})")
    if times:
        print(f"measured {len(times)} prompts, {st.median(times):.0f} ms p50, {len(ROUTER_BANK)} questions")
    else:
        print(f"all {len(rows)} prompts served from {CACHE} (delete it to re-measure)")

    ho, (lo, hi) = result["held_out"], result["held_out_ci_95"]
    print(f"\n=== shipped thresholds (difficulty {T_DIFFICULTY}, sensitive {T_SENSITIVE}) ===")
    print(f"  train      {result['train_accuracy']:.1f}%")
    print(f"  HELD-OUT   {ho['right']}/{ho['n']} = {result['score']:.1f}%   "
          f"95% CI {lo:.1f}% to {hi:.1f}%   (chance {result['chance']:.0f}%)")
    print(f"  hard->small {ho['hard_to_small']}   easy->big {ho['easy_to_big']}   "
          f"cost {ho['cost']:.3f}/prompt")

    print(f"\n=== rank signal, all {len(rows)} prompts (threshold-free) ===")
    for q, v in result["auc"].items():
        print(f"  {q:14s} AUC {v:.2f}")

    if result["mistakes"]:
        print(f"\n=== held-out mistakes ({len(result['mistakes'])}) ===")
        for w in result["mistakes"]:
            print(f"  {w}")

    if args.refit:
        c, td, ts = fit(train, probs)
        refit = evaluate(rows, probs, td, ts)
        rlo, rhi = refit["held_out_ci_95"]
        print(f"\n=== refit on train, minimising cost (hard->small counts 3x) ===")
        print(f"  best on train   cost {c:.3f}/prompt at difficulty {td}, sensitive {ts}")
        print(f"  that on test    {refit['score']:.1f}%  95% CI {rlo:.1f}% to {rhi:.1f}%   "
              f"hard->small {refit['held_out']['hard_to_small']}  "
              f"easy->big {refit['held_out']['easy_to_big']}")
        print(f"  paste into router.py:  T_DIFFICULTY = {td}   T_SENSITIVE = {ts}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=1))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
