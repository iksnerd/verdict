"""Pipeline stages. Each one is resumable (rerun it and only the missing work happens), writes
crash-safe files, and ends by writing `<stage>.manifest.json`: config hash, git commit, sha256
of every input and output, counts and spend. The manifest is what a scorecard pins."""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..gen.domains import Domain
from ..gen.generate import generate_cases
from ..label.teacher import LabelError, Teacher, cost_of
from .audit import split_stats, text_of, topic_of
from .config import PipelineConfig
from .store import JsonlStore, sha256_file, write_json_atomic


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
    except Exception:
        return None


def write_manifest(run_dir: Path, stage: str, cfg: PipelineConfig, inputs: list[str], outputs: list[str], counts: dict, **extra):
    manifest = {
        "stage": stage,
        "config_digest": cfg.digest(),
        "config": cfg.model_dump(),
        "git_commit": _git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {n: sha256_file(run_dir / n) for n in inputs if (run_dir / n).exists()},
        "outputs": {n: sha256_file(run_dir / n) for n in outputs if (run_dir / n).exists()},
        "counts": counts,
        **extra,
    }
    write_json_atomic(run_dir / f"{stage}.manifest.json", manifest)
    return manifest


def gen_stage(cfg: PipelineConfig, run_dir: Path, client, domain: Domain) -> int:
    run_dir = Path(run_dir)
    store = JsonlStore(run_dir / "states.jsonl", key=lambda r: r["id"])
    have = len(store.rows())
    need = cfg.gen.n - have
    if need > 0:
        seen = [text_of(r) for r in store.rows()]
        # seed by how many already exist, so a resumed run does not replay the same seed plan
        generate_cases(
            domain, client, n=need, rng=random.Random(f"{cfg.gen.seed}:{have}"), seen=seen,
            on_case=lambda c: store.append(c.__dict__), workers=cfg.gen.workers,
        )
        if len(store.rows()) == have:
            # Short of `n` is normal (the length cap drops texts). Nothing at all is not: every
            # call failed or was rejected, and the later stages would run on an empty or stale set.
            raise SystemExit(f"gen: no state was accepted out of {need} asked for; check the "
                             "generator (key, model, server) before running the other stages")
    rows = store.rows()
    write_manifest(run_dir, "gen", cfg, [], ["states.jsonl"], {"states": len(rows), "requested": cfg.gen.n},
                   generator=cfg.gen.model)
    return len(rows)


def label_stage(cfg: PipelineConfig, run_dir: Path, transport, bank: dict, preamble: str | None = None) -> dict:
    run_dir = Path(run_dir)
    states = JsonlStore(run_dir / "states.jsonl", key=lambda r: r["id"]).rows()
    labels = JsonlStore(run_dir / "labels.jsonl", key=lambda r: f'{r["id"]}|{r["model"]}|{r["repeat"]}')
    lc = cfg.label
    plan = [(m, lc.repeats) for m in lc.models] + ([(lc.reference, 1)] if lc.reference else [])
    teachers = {m: Teacher(m, transport, preamble=preamble) for m, _ in plan}

    def spent() -> float:
        return sum(cost_of(r["model"], r["tokens_in"], r["tokens_out"]) for r in labels.rows())

    failures = 0
    # A failed call is billed for the tries that answered, but writes no row, so `spent()` cannot
    # see it. It is counted here, for this run only: nothing on disk records what the failed calls
    # of an earlier run cost, so a resumed run starts this at zero and the cap is per run for them.
    failed_spend = 0.0
    for s in states:
        for model, reps in plan:
            for rep in range(reps):
                key = f'{s["id"]}|{model}|{rep}'
                if key in labels.done:
                    continue
                total = spent() + failed_spend
                if total >= lc.max_cost_usd:
                    raise SystemExit(f"cost cap ${lc.max_cost_usd} reached (spent ${total:.4f}, "
                                     f"${failed_spend:.4f} of it on failed calls); rerun to resume")
                t = teachers[model]
                try:
                    out = t.label(s["state"], bank)
                except LabelError:
                    failures += 1
                    failed_spend += cost_of(model, t.last_in, t.last_out)
                    continue
                labels.append({"id": s["id"], "model": model, "repeat": rep, "labels": out,
                               "tokens_in": t.last_in, "tokens_out": t.last_out})
    rows = labels.rows()
    counts = {"states": len(states), "label_rows": len(rows), "failures_this_run": failures,
              "failed_spend_usd": round(failed_spend, 6)}
    # `spend_usd` stays what the rows on disk cost, every run included; `failed_spend_usd` is this run's only.
    write_manifest(run_dir, "label", cfg, ["states.jsonl"], ["labels.jsonl"], counts,
                   spend_usd=round(spent(), 6), failed_spend_usd=round(failed_spend, 6),
                   teachers=[m for m, _ in plan])
    return counts


def _bucket(topic: str, seed: int) -> float:
    return int(hashlib.sha1(f"{seed}|{topic}".encode()).hexdigest()[:8], 16) / 2**32


def _assign(states: list[dict], sp, seed: int) -> dict[str, list[str]]:
    parts: dict[str, list[str]] = {"train": [], "holdout": [], "test": []}
    for s in states:
        h = _bucket(topic_of(s), seed)
        name = "test" if h < sp.test_frac else "holdout" if h < sp.test_frac + sp.holdout_frac else "train"
        parts[name].append(s["id"])
    return parts


def split_stage(cfg: PipelineConfig, run_dir: Path) -> dict:
    """Split by room topic, not by row: near-identical states about one topic must not straddle
    splits. Assignment is a hash of the topic, so adding states never moves an old one.
    [0, test_frac) is test, the next holdout_frac is holdout, the rest is train.

    With `require_type_coverage`, seeds are tried in order until every intended type appears in both
    test and holdout (a lesson from an earlier pipeline: aggregate counts hid a source missing from most splits).
    The seed used is recorded in `split.json`, and a rerun reuses it instead of searching again:
    with more states the first covering seed can be a different one, and a different seed moves
    every topic, test set included. A rerun whose seed or fractions differ from the recorded ones
    is refused for the same reason; delete `split.json` to split afresh on purpose."""
    run_dir = Path(run_dir)
    states = JsonlStore(run_dir / "states.jsonl", key=lambda r: r["id"]).rows()
    sp = cfg.split
    seed = sp.seed
    recorded = run_dir / "split.json"
    prior = json.loads(recorded.read_text()) if recorded.exists() else None
    if prior is not None:
        was = (prior.get("seed"), prior.get("test_frac"), prior.get("holdout_frac"))
        now = (sp.seed, sp.test_frac, sp.holdout_frac)
        if was != now or "seed_used" not in prior:
            raise SystemExit(f"{recorded} was made with seed, test_frac, holdout_frac = {was} and this "
                             f"config has {now}: splitting again would move the test set. Delete "
                             "split.json to split afresh on purpose.")
        seed = prior["seed_used"]
    parts = _assign(states, sp, seed)
    if sp.require_type_coverage and prior is None:
        for seed in range(sp.seed, sp.seed + 500):
            parts = _assign(states, sp, seed)
            st = split_stats(states, parts)
            if not st["missing_in_test"] and not st["missing_in_holdout"]:
                break
        else:
            raise SystemExit("no seed in 500 gives every intended type in both test and holdout; add states or topics")
    out = {**{k: sorted(v) for k, v in parts.items()}, "test_frac": sp.test_frac, "holdout_frac": sp.holdout_frac,
           "seed": sp.seed, "seed_used": seed}
    write_json_atomic(run_dir / "split.json", out)
    write_manifest(run_dir, "split", cfg, ["states.jsonl"], ["split.json"], {k: len(v) for k, v in parts.items()}, seed_used=seed)
    return out
