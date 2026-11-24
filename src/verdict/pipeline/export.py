"""Export labelled states in the `LocalLLaMA/typed-decisions` schema that Laya's fine-tune
notebook reads: one row per case with `state`, `questions` and `gold` as JSON strings, where
`gold[qid]["probabilities"]` is the target distribution.

Targets are the mean of the cheap teachers' distributions. The reference teacher is left out on
purpose so it stays an independent judge. Rules carried over from an earlier corpus pipeline:
the test split is written apart from the training files and never named as one, states that leak
into held-out splits are excluded, and the write order is a salted hash of the id so it is
uncorrelated with generation order (a positional validation split would otherwise be one topic)."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from ..answers import noul_probs
from ..label.teacher import mean_labels
from .audit import SPLITS, held_out_leak, split_of
from .config import PipelineConfig
from .stages import label_digest, require_fresh, write_manifest
from .store import jsonl_text, labels_store, states_store, write_text_atomic


def _gold(bank: dict, mean: dict) -> dict:
    out = {}
    for qid, q in bank.items():
        v = mean[qid]
        out[qid] = {"probabilities": v if q["type"] != "noul" else noul_probs(v)}
    return out


def export_stage(cfg: PipelineConfig, run_dir: Path, bank: dict, salt: str = "0", preamble: str | None = None) -> dict:
    """`preamble` is the teachers' preamble the labels were made with (`label_stage`'s argument): rows
    made under another domain, bank or preamble are refused rather than averaged in."""
    run_dir = Path(run_dir)
    require_fresh(run_dir, "export")
    # One row per key, the last winning: a relabelled row replaces the old one, and a key written
    # twice by two runs is not weighted double.
    states = states_store(run_dir).latest()
    split = json.loads((run_dir / "split.json").read_text())
    digest = label_digest(cfg, bank, preamble)
    per: dict = defaultdict(lambda: defaultdict(list))
    wrong = 0
    label_rows = labels_store(run_dir).latest()
    for r in sorted(label_rows, key=lambda r: r["repeat"]):
        # the reference is excluded on purpose; a repeat past `repeats` is left over from a larger setting
        if r["model"] in cfg.label.models and r["repeat"] < cfg.label.repeats:
            if r.get("label_digest", digest) != digest:  # rows without the key are from before it existed
                wrong += 1
                continue
            per[r["id"]][r["model"]].append(r["labels"])
    if wrong:
        raise SystemExit(f"{wrong} label rows in {run_dir / 'labels.jsonl'} were made under a different domain, "
                         "bank or preamble than this export has. Rerun `--stage label` (it asks those again) "
                         "or pass the preamble the labels were made with.")
    assigned = set(split_of(split))
    unassigned = sorted(s["id"] for s in states if s["id"] not in assigned)
    if unassigned:
        print(f"export: {len(unassigned)} states in states.jsonl are in no split and are left out "
              f"(first {unassigned[0]}); rerun `--stage split`", file=sys.stderr)
    leaked = set(held_out_leak(states, split))
    questions = json.dumps(bank, ensure_ascii=False)
    by_id = {s["id"]: s for s in states}
    counts, excluded = {}, {"leaked": len(leaked), "unlabelled": 0, "unassigned": len(unassigned)}
    texts = {}
    for name in SPLITS:
        rows = []
        for i in split[name]:
            if (name == "train" and i in leaked):
                continue
            teachers = per.get(i, {})
            if not all(m in teachers for m in cfg.label.models) or not teachers:
                excluded["unlabelled"] += 1
                continue
            mean = mean_labels([lab for runs in teachers.values() for lab in runs])
            rows.append({"id": i, "workflow": cfg.domain, "state": json.dumps(by_id[i]["state"], ensure_ascii=False),
                         "questions": questions, "gold": json.dumps(_gold(bank, mean))})
        rows.sort(key=lambda r: hashlib.sha256(f"{salt}|{r['id']}".encode()).hexdigest())
        texts[name] = jsonl_text(rows)
        counts[name] = len(rows)
    # Every file whole or not at all, and the manifest after all three: a kill in between leaves a
    # manifest that does not describe the new files, which `require_fresh` and the sha256s expose.
    for name, text in texts.items():
        write_text_atomic(run_dir / "export" / f"{name}.jsonl", text)
    write_manifest(run_dir, "export", cfg, ["states.jsonl", "labels.jsonl", "split.json"],
                   [f"export/{n}.jsonl" for n in SPLITS], counts, excluded=excluded, salt=salt)
    return counts
