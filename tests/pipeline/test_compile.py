"""Compiling every run's synthetic data into one file: a lossless row per state per run."""
import json

from verdict.pipeline.compile import compile_runs


def _run(d, states, labels, split=None):
    d.mkdir()
    (d / "states.jsonl").write_text("".join(json.dumps(s) + "\n" for s in states))
    (d / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in labels))
    if split is not None:
        (d / "split.json").write_text(json.dumps(split))
    return d


def test_one_row_per_state_with_split_and_every_teacher(tmp_path):
    s = {"id": "a", "domain": "github_issue_typing", "intended_type": "bug", "ambiguous": False,
         "confusable": None, "generator": "g", "state": {"issue": "Crash\nbody"}}
    labels = [{"id": "a", "model": m, "repeat": 0, "labels": {"type": {"bug": 1.0}}, "tokens_in": 1, "tokens_out": 1}
              for m in ("t1", "t2")]
    r1 = _run(tmp_path / "r1", [s], labels, {"train": ["a"], "holdout": [], "test": [], "seed_used": 0})
    r2 = _run(tmp_path / "r2", [s, {**s, "id": "b"}], [])  # no split, no labels
    rows = compile_runs([r1, r2])
    assert [(r["run"], r["id"], r["split"]) for r in rows] == [("r1", "a", "train"), ("r2", "a", None), ("r2", "b", None)]
    first = rows[0]
    assert first["state"] == {"issue": "Crash\nbody"} and first["intended_type"] == "bug"
    assert [(t["model"], t["labels"]) for t in first["teachers"]] == [("t1", {"type": {"bug": 1.0}}), ("t2", {"type": {"bug": 1.0}})]
    assert rows[1]["teachers"] == []
