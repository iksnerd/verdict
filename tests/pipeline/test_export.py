import json
import random

from verdict.gen.domains import MESSAGE_TYPING
from verdict.label.bank import BANK
from verdict.pipeline.audit import held_out_leak, split_stats
from verdict.pipeline.config import PipelineConfig
from verdict.pipeline.export import export_stage
from verdict.pipeline.stages import gen_stage, label_stage, split_stage
from verdict.pipeline.store import JsonlStore


class FakeGen:
    model = "fake"

    def __init__(self):
        self.n = 0

    def generate(self, prompt, seed):
        self.n += 1
        return f"Item {self.n} shipped for module {seed} after review and full verification of edge cases {seed}."


GOOD = {
    "kind": {"action": 0.7, "synthesis": 0.1, "decision": 0.05, "thought": 0.05, "draft": 0.05, "note": 0.03, "other": 0.02},
    "records_decision": {"true": 0.1, "false": 0.9},
    "reports_shipped_work": {"true": 0.9, "false": 0.1},
    "leaves_open_question": {"true": 0.1, "false": 0.9},
    "is_proposal": {"true": 0.1, "false": 0.9},
}


def transport(model, prompt):
    return json.dumps(GOOD), {"promptTokenCount": 10, "candidatesTokenCount": 5}


def cfg():
    return PipelineConfig.model_validate({"name": "t", "gen": {"n": 60}, "label": {"models": ["m1", "m2"], "reference": None, "repeats": 2, "max_cost_usd": 5},
                                          "split": {"test_frac": 0.2, "holdout_frac": 0.2}})


def built(tmp_path):
    c = cfg()
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    label_stage(c, tmp_path, transport, BANK)
    split_stage(c, tmp_path)
    return c


def test_split_stats_crosstabs_type_by_split(tmp_path):
    built(tmp_path)
    states = JsonlStore(tmp_path / "states.jsonl", key=lambda r: r["id"]).rows()
    stats = split_stats(states, json.loads((tmp_path / "split.json").read_text()))
    assert set(stats["intended_type"]) <= set(MESSAGE_TYPING.types)
    total = sum(sum(v.values()) for v in stats["intended_type"].values())
    assert total == len(states)
    assert "missing_in_test" in stats  # types with zero test presence are called out


def test_held_out_leak_finds_exact_and_near_duplicates_across_splits():
    states = [
        {"id": "a", "state": {"message": "Shipped the retry logic for the sync worker and added a regression test"}},
        {"id": "b", "state": {"message": "shipped the retry logic for the sync worker and added a regression test today"}},
        {"id": "c", "state": {"message": "Should we move the queue to Postgres or keep Redis for now?"}},
    ]
    split = {"train": ["a", "c"], "holdout": [], "test": ["b"]}
    assert held_out_leak(states, split) == ["a"]


def test_export_writes_typed_decisions_schema_and_averages_teachers(tmp_path):
    c = built(tmp_path)
    export_stage(c, tmp_path, BANK)
    row = json.loads((tmp_path / "export/train.jsonl").read_text().splitlines()[0])
    assert set(row) == {"id", "workflow", "state", "questions", "gold"}
    gold = json.loads(row["gold"])
    assert abs(sum(gold["kind"]["probabilities"].values()) - 1.0) < 1e-6
    assert gold["records_decision"]["probabilities"].keys() == {"true", "false"}
    assert json.loads(row["questions"])["kind"]["type"] == "choice"


def test_test_split_is_structurally_absent_from_training_files(tmp_path):
    c = built(tmp_path)
    export_stage(c, tmp_path, BANK)
    split = json.loads((tmp_path / "split.json").read_text())
    trained = {json.loads(l)["id"] for f in ("train", "holdout") for l in (tmp_path / f"export/{f}.jsonl").read_text().splitlines()}
    assert not trained & set(split["test"])
    assert (tmp_path / "export/test.jsonl").exists()  # kept apart, never named as a training file


def test_export_order_is_shuffled_by_salted_hash_not_by_generation_order(tmp_path):
    c = built(tmp_path)
    export_stage(c, tmp_path, BANK, salt="a")
    a = [json.loads(l)["id"] for l in (tmp_path / "export/train.jsonl").read_text().splitlines()]
    export_stage(c, tmp_path, BANK, salt="b")
    b = [json.loads(l)["id"] for l in (tmp_path / "export/train.jsonl").read_text().splitlines()]
    states = [r["id"] for r in JsonlStore(tmp_path / "states.jsonl", key=lambda r: r["id"]).rows()]
    in_gen_order = [i for i in states if i in set(a)]
    assert a != in_gen_order and a != b and sorted(a) == sorted(b)


def test_held_out_leak_reads_issue_states():
    # issue states keep their text under "issue"; read as empty, every one "duplicated" another
    states = [
        {"id": "a", "state": {"issue": "Crash on startup with 2.1\nTraceback in loader.py when the cache is empty"}},
        {"id": "b", "state": {"issue": "How do I set a custom cache path?\nThe docs mention an option I cannot find"}},
        {"id": "c", "state": {"issue": "Add a --dry-run flag to migrate\nIt would help to preview changes first"}},
    ]
    assert held_out_leak(states, {"train": ["a", "c"], "holdout": ["b"], "test": []}) == []
