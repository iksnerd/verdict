"""Every run's synthetic data in one file: a row per state per run, with its split and each
teacher's raw labels. Lossless on purpose, so a later mean, filter or reweighting needs no rerun.

    uv run python -m verdict.pipeline.compile runs/v1 runs/issues_pilot --out data/compiled/synthetic.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .stages import require_fresh
from .store import JsonlStore

_STATE_KEYS = ("domain", "intended_type", "ambiguous", "confusable", "generator", "state")


def compile_runs(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in map(Path, run_dirs):
        require_fresh(run_dir, "compile")
        split_path = run_dir / "split.json"
        split = json.loads(split_path.read_text()) if split_path.exists() else {}
        where = {i: name for name in ("train", "holdout", "test") for i in split.get(name, [])}
        teachers = defaultdict(list)
        labels_path = run_dir / "labels.jsonl"
        if labels_path.exists():
            for r in JsonlStore(labels_path, key=lambda r: f'{r["id"]}|{r["model"]}|{r["repeat"]}').latest():
                teachers[r["id"]].append({"model": r["model"], "repeat": r["repeat"], "labels": r["labels"]})
        for s in JsonlStore(run_dir / "states.jsonl", key=lambda r: r["id"]).latest():
            rows.append({"run": run_dir.name, "id": s["id"], **{k: s.get(k) for k in _STATE_KEYS},
                         "split": where.get(s["id"]),
                         "teachers": sorted(teachers[s["id"]], key=lambda t: (t["model"], t["repeat"]))})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = compile_runs(a.runs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    by = Counter((r["run"], r["domain"]) for r in rows)
    labelled = Counter((r["run"], r["domain"]) for r in rows if r["teachers"])
    for (run, domain), n in sorted(by.items()):
        print(f"{run:15} {domain:22} {n:5} states  {labelled[run, domain]:5} labelled")
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
