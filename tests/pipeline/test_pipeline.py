import json
import random

import pytest

from verdict.gen.domains import MESSAGE_TYPING
from verdict.label.bank import BANK
from verdict.pipeline.config import PipelineConfig
from verdict.pipeline.stages import gen_stage, label_stage, split_stage
from verdict.pipeline.store import JsonlStore, sha256_file, write_json_atomic


def cfg(**over):
    base = {"name": "t", "gen": {"n": 6, "seed": 0}, "label": {"models": ["m1", "m2"], "reference": "ref", "repeats": 2, "max_cost_usd": 5.0}, "split": {"test_frac": 0.3, "seed": 0}}
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return PipelineConfig.model_validate(base)


class FakeGen:
    model = "fake-gen"

    def __init__(self):
        self.n = 0

    def generate(self, prompt, seed):
        self.n += 1
        return f"Finished item {self.n} in module {seed} and verified it with tests {seed}."


GOOD = {
    "kind": {"action": 0.7, "synthesis": 0.1, "decision": 0.05, "thought": 0.05, "draft": 0.05, "note": 0.03, "other": 0.02},
    "records_decision": {"true": 0.1, "false": 0.9},
    "reports_shipped_work": {"true": 0.9, "false": 0.1},
    "leaves_open_question": {"true": 0.1, "false": 0.9},
    "is_proposal": {"true": 0.1, "false": 0.9},
}


class FakeTransport:
    def __init__(self):
        self.calls = 0

    def __call__(self, model, prompt):
        self.calls += 1
        return json.dumps(GOOD), {"promptTokenCount": 1000, "candidatesTokenCount": 500}


def test_store_appends_and_reports_done_keys(tmp_path):
    s = JsonlStore(tmp_path / "a.jsonl", key=lambda r: r["id"])
    s.append({"id": "x", "v": 1})
    s2 = JsonlStore(tmp_path / "a.jsonl", key=lambda r: r["id"])  # reopened: resumable
    assert "x" in s2.done and len(s2.rows()) == 1


def test_store_skips_a_torn_last_line(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"id": "x"}\n{"id": "y", "tru')
    s = JsonlStore(p, key=lambda r: r["id"])
    assert s.done == {"x"}


def test_write_json_atomic_leaves_no_partial_file(tmp_path):
    write_json_atomic(tmp_path / "m.json", {"a": 1})
    assert json.loads((tmp_path / "m.json").read_text()) == {"a": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_gen_stage_is_resumable(tmp_path):
    g = FakeGen()
    gen_stage(cfg(), tmp_path, g, MESSAGE_TYPING)
    first = g.n
    assert len(JsonlStore(tmp_path / "states.jsonl", key=lambda r: r["id"]).rows()) == 6
    gen_stage(cfg(), tmp_path, g, MESSAGE_TYPING)
    assert g.n == first  # nothing regenerated


def test_label_stage_resumes_and_records_cost(tmp_path):
    gen_stage(cfg(), tmp_path, FakeGen(), MESSAGE_TYPING)
    t = FakeTransport()
    label_stage(cfg(), tmp_path, t, BANK)
    n_calls = 6 * (2 * 2 + 1)  # 6 states x (2 models x 2 repeats + 1 reference)
    assert t.calls == n_calls
    label_stage(cfg(), tmp_path, t, BANK)
    assert t.calls == n_calls  # resumed: zero new calls
    manifest = json.loads((tmp_path / "label.manifest.json").read_text())
    assert manifest["counts"]["label_rows"] == n_calls
    assert manifest["outputs"]["labels.jsonl"] == sha256_file(tmp_path / "labels.jsonl")


def test_label_stage_stops_at_the_cost_cap(tmp_path):
    gen_stage(cfg(), tmp_path, FakeGen(), MESSAGE_TYPING)
    t = FakeTransport()
    with pytest.raises(SystemExit):
        label_stage(cfg(label={"max_cost_usd": 0.0}), tmp_path, t, BANK)
    assert t.calls == 0


def test_split_is_deterministic_and_topic_disjoint(tmp_path):
    gen_stage(cfg(gen={"n": 40}), tmp_path, FakeGen(), MESSAGE_TYPING)
    split_stage(cfg(gen={"n": 40}), tmp_path)
    a = json.loads((tmp_path / "split.json").read_text())
    split_stage(cfg(gen={"n": 40}), tmp_path)
    assert a == json.loads((tmp_path / "split.json").read_text())
    states = {r["id"]: r for r in JsonlStore(tmp_path / "states.jsonl", key=lambda r: r["id"]).rows()}
    train_topics = {states[i]["state"]["room_topic"] for i in a["train"]}
    test_topics = {states[i]["state"]["room_topic"] for i in a["test"]}
    assert not (train_topics & test_topics) and a["test"] and a["train"]
    assert not set(a["train"]) & set(a["test"])


def test_three_way_split_is_topic_disjoint_and_covers_everything(tmp_path):
    c = cfg(gen={"n": 60}, split={"test_frac": 0.2, "holdout_frac": 0.2})
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    out = split_stage(c, tmp_path)
    states = {r["id"]: r for r in JsonlStore(tmp_path / "states.jsonl", key=lambda r: r["id"]).rows()}
    parts = {k: {states[i]["state"]["room_topic"] for i in out[k]} for k in ("train", "holdout", "test")}
    assert set(out["train"]) | set(out["holdout"]) | set(out["test"]) == set(states)
    assert not (parts["train"] & parts["holdout"]) and not (parts["train"] & parts["test"]) and not (parts["holdout"] & parts["test"])
    assert out["holdout"] and out["test"]


def test_holdout_defaults_to_empty(tmp_path):
    gen_stage(cfg(), tmp_path, FakeGen(), MESSAGE_TYPING)
    assert split_stage(cfg(), tmp_path)["holdout"] == []


def test_require_type_coverage_searches_seeds_until_every_type_is_in_test_and_holdout(tmp_path):
    from verdict.pipeline.audit import split_stats

    c = cfg(gen={"n": 150}, split={"test_frac": 0.25, "holdout_frac": 0.2, "require_type_coverage": True})
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    out = split_stage(c, tmp_path)
    states = JsonlStore(tmp_path / "states.jsonl", key=lambda r: r["id"]).rows()
    st = split_stats(states, out)
    assert st["missing_in_test"] == [] and st["missing_in_holdout"] == []
    assert "seed_used" in out


def test_coverage_search_gives_up_loudly_when_impossible(tmp_path):
    c = cfg(gen={"n": 6}, split={"test_frac": 0.3, "holdout_frac": 0.2, "require_type_coverage": True})
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    with pytest.raises(SystemExit):
        split_stage(c, tmp_path)
