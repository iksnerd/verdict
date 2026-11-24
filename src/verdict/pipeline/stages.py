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
from ..label.teacher import DEFAULT_PREAMBLE, LabelError, Teacher, cost_of, price_of
from .audit import split_stats, text_of, topic_of
from .config import PipelineConfig
from .store import JsonlStore, run_lock, sha256_file, write_json_atomic


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


def label_digest(cfg: PipelineConfig, bank: dict, preamble: str | None = None) -> str:
    """What a label row was made for: the domain, the question bank and the preamble. It is stored
    in each new row, so a row made under another bank is not mixed into an export."""
    payload = json.dumps({"domain": cfg.domain, "bank": bank, "preamble": (preamble or DEFAULT_PREAMBLE).strip()},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


#: (stage whose manifest is read, "inputs" or "outputs", file) for each stage's freshness check.
_RECORDS = {
    "label": [("gen", "outputs", "states.jsonl")],
    "split": [("gen", "outputs", "states.jsonl")],
    "export": [("gen", "outputs", "states.jsonl"), ("label", "outputs", "labels.jsonl"),
               ("label", "inputs", "states.jsonl"), ("split", "outputs", "split.json"),
               ("split", "inputs", "states.jsonl")],
}
_RECORDS["compile"] = _RECORDS["export"]


def require_fresh(run_dir: Path, consumer: str) -> None:
    """Refuse to run `consumer` on a file that no longer matches what an earlier stage recorded.

    Each stage's manifest holds the sha256 of its inputs and outputs. If `labels.jsonl` was
    appended to after the label manifest was written (a killed run, a hand edit), or `split.json`
    was made from other states than the ones on disk, the stage that follows would read a mix.
    A stage with no manifest (a run made by hand, or before manifests) is not checked."""
    run_dir = Path(run_dir)
    for stage, section, name in _RECORDS[consumer]:
        mpath = run_dir / f"{stage}.manifest.json"
        path = run_dir / name
        if not mpath.exists() or not path.exists():
            continue
        recorded = json.loads(mpath.read_text()).get(section, {}).get(name)
        if recorded is None or recorded == sha256_file(path):
            continue
        what = "made" if section == "outputs" else "read"
        raise SystemExit(
            f"{path} is stale for {consumer}: it no longer matches the copy the {stage} stage {what} "
            f"(recorded {recorded[:12]}, now {sha256_file(path)[:12]}). Rerun `python -m verdict.pipeline "
            f"--config <config> --run-dir {run_dir} --stage {stage}` first.")


def gen_stage(cfg: PipelineConfig, run_dir: Path, client, domain: Domain) -> int:
    run_dir = Path(run_dir)
    with run_lock(run_dir, "gen"):
        return _gen_stage(cfg, run_dir, client, domain)


def _gen_stage(cfg: PipelineConfig, run_dir: Path, client, domain: Domain) -> int:
    store = JsonlStore(run_dir / "states.jsonl", key=lambda r: r["id"])
    have = len(store.latest())
    need = cfg.gen.n - have
    # The generator's tokens are billed whether or not the text is kept. They are in the client's
    # memory only, so each accepted state carries what was spent since the one before it.
    seen_tokens = [getattr(client, "tokens_in", None), getattr(client, "tokens_out", None)]

    def on_case(c):
        row = dict(c.__dict__)
        tin, tout = getattr(client, "tokens_in", None), getattr(client, "tokens_out", None)
        if tin is not None and tout is not None and seen_tokens[0] is not None:
            row["gen_tokens"] = {"in": tin - seen_tokens[0], "out": tout - seen_tokens[1]}
            seen_tokens[:] = [tin, tout]
        store.append(row)

    if need > 0:
        seen = [text_of(r) for r in store.rows()]
        # The type of seed i is a function of (seed, i), counted from the states already there, so a
        # run killed and resumed over and over continues one cycle instead of reshuffling it.
        order = list(domain.types)
        random.Random(f"{cfg.gen.seed}:types").shuffle(order)
        generate_cases(
            domain, client, n=need, rng=random.Random(f"{cfg.gen.seed}:{have}"), seen=seen,
            on_case=on_case, workers=cfg.gen.workers, type_order=order, start=have,
        )
        if len(store.latest()) == have:
            # Short of `n` is normal (the length cap drops texts). Nothing at all is not: every
            # call failed or was rejected, and the later stages would run on an empty or stale set.
            raise SystemExit(f"gen: no state was accepted out of {need} asked for; check the "
                             "generator (key, model, server) before running the other stages")
    rows = store.latest()
    extra = {}
    if seen_tokens[0] is not None:
        # Rows hold what was spent up to each accepted state; calls after the last one (rejected
        # texts at the end of a run) are only in this figure, and a run killed before this line loses them.
        # There is no cost cap on gen: `gen.*` has no cost key, so this is reported, not enforced.
        row_in = sum(r.get("gen_tokens", {}).get("in", 0) for r in store.rows())
        row_out = sum(r.get("gen_tokens", {}).get("out", 0) for r in store.rows())
        extra = {"generator_tokens_in": row_in + client.tokens_in - seen_tokens[0],
                 "generator_tokens_out": row_out + client.tokens_out - seen_tokens[1]}
    write_manifest(run_dir, "gen", cfg, [], ["states.jsonl"], {"states": len(rows), "requested": cfg.gen.n},
                   generator=cfg.gen.model, **extra)
    return len(rows)


def label_stage(cfg: PipelineConfig, run_dir: Path, transport, bank: dict, preamble: str | None = None) -> dict:
    run_dir = Path(run_dir)
    lc = cfg.label
    plan = [(m, lc.repeats) for m in lc.models] + ([(lc.reference, 1)] if lc.reference else [])
    for m, _ in plan:
        price_of(m, lc.prices)  # an unpriced model would cost $0 and the cap could never trip
    with run_lock(run_dir, "label"):
        return _label_stage(cfg, run_dir, transport, bank, preamble, plan)


def _label_stage(cfg, run_dir, transport, bank, preamble, plan) -> dict:
    require_fresh(run_dir, "label")
    states = JsonlStore(run_dir / "states.jsonl", key=lambda r: r["id"]).latest()
    labels = JsonlStore(run_dir / "labels.jsonl", key=lambda r: f'{r["id"]}|{r["model"]}|{r["repeat"]}')
    lc = cfg.label
    digest = label_digest(cfg, bank, preamble)
    # A row made under another bank, preamble or domain is not done: it is asked again, and the
    # newer row wins on read. Old rows without a digest are kept.
    redo = {labels.key(r) for r in labels.rows() if r.get("label_digest", digest) != digest}
    teachers = {m: Teacher(m, transport, preamble=preamble, prices=lc.prices) for m, _ in plan}

    def spent() -> float:
        return sum(cost_of(r["model"], r["tokens_in"], r["tokens_out"], lc.prices) for r in labels.rows())

    failures = 0
    # A failed call is billed for the tries that answered, but writes no row, so `spent()` cannot
    # see it. It is counted here, for this run only: nothing on disk records what the failed calls
    # of an earlier run cost, so a resumed run starts this at zero and the cap is per run for them.
    failed_spend = 0.0
    try:
        for s in states:
            for model, reps in plan:
                for rep in range(reps):
                    key = f'{s["id"]}|{model}|{rep}'
                    if key in labels.done and key not in redo:
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
                        failed_spend += cost_of(model, t.last_in, t.last_out, lc.prices)
                        continue
                    labels.append({"id": s["id"], "model": model, "repeat": rep, "labels": out,
                                   "tokens_in": t.last_in, "tokens_out": t.last_out, "label_digest": digest})
                    redo.discard(key)
    finally:
        # Also when the cap or a Ctrl-C ends the run: the manifest describes the labels.jsonl that
        # is on disk now, not the one the previous run left.
        rows = labels.latest()
        counts = {"states": len(states), "label_rows": len(rows), "failures_this_run": failures,
                  "failed_spend_usd": round(failed_spend, 6)}
        # `spend_usd` stays what the rows on disk cost, every run included; `failed_spend_usd` is this run's only.
        write_manifest(run_dir, "label", cfg, ["states.jsonl"], ["labels.jsonl"], counts,
                       spend_usd=round(spent(), 6), failed_spend_usd=round(failed_spend, 6),
                       teachers=[m for m, _ in plan], label_digest=digest)
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
    require_fresh(run_dir, "split")
    states = JsonlStore(run_dir / "states.jsonl", key=lambda r: r["id"]).latest()
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
