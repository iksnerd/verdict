"""Stage freshness, label provenance, atomic export, deterministic gen, run locks (audit 2). Invented data only."""
import json

import pytest

from verdict.gen.domains import MESSAGE_TYPING
from verdict.label.bank import BANK
from verdict.label.teacher import UnpricedModelError
from verdict.pipeline import export as export_mod
from verdict.pipeline.compile import compile_runs
from verdict.pipeline.config import PipelineConfig
from verdict.pipeline.export import export_stage
from verdict.pipeline.stages import gen_stage, label_stage, split_stage
from verdict.pipeline.store import JsonlStore, jsonl_text, labels_store, run_lock, sha256_file, states_store

from pfakes import AlternatingTransport, build_run

PRICES = {"m1": [0.1, 0.4], "m2": [0.1, 0.4]}


def cfg(n=6, repeats=2, cap=5.0):
    return PipelineConfig.model_validate({
        "name": "t", "gen": {"n": n}, "split": {"test_frac": 0.3, "holdout_frac": 0.2},
        "label": {"models": ["m1", "m2"], "reference": None, "repeats": repeats, "max_cost_usd": cap, "prices": PRICES}})


class FakeGen:
    model = "fake-gen"

    def __init__(self):
        self.n = 0
        self.tokens_in = self.tokens_out = 0

    def generate(self, prompt, seed):
        self.n += 1
        self.tokens_in += 10
        self.tokens_out += 3
        return f"Finished item {self.n} in module {seed} and verified it with tests {seed}."


def built(tmp_path, **kw):
    return build_run(cfg(**kw), tmp_path, FakeGen(), AlternatingTransport())


def append_state(tmp_path, sid="zz"):
    with open(tmp_path / "states.jsonl", "a") as f:
        f.write(json.dumps({"id": sid, "intended_type": "note", "state": {"room_topic": "t", "author": "a", "message": "x" * 30}}) + "\n")


# --- 1. freshness -------------------------------------------------------------------------------

def test_split_refuses_states_changed_since_gen_wrote_its_manifest(tmp_path):
    c = cfg()
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    append_state(tmp_path)
    with pytest.raises(SystemExit, match=r"states\.jsonl.*stale.*--stage gen"):
        split_stage(c, tmp_path)


def test_export_refuses_labels_changed_since_the_label_manifest(tmp_path):
    c = built(tmp_path)
    with open(tmp_path / "labels.jsonl", "a") as f:
        f.write("\n")  # any change at all
        f.write(json.dumps({"id": "x", "model": "m1", "repeat": 0, "labels": {}, "tokens_in": 0, "tokens_out": 0}) + "\n")
    with pytest.raises(SystemExit, match=r"labels\.jsonl.*stale.*--stage label"):
        export_stage(c, tmp_path, BANK)


def test_export_refuses_a_split_made_from_other_states(tmp_path):
    c = built(tmp_path)
    c2 = cfg(n=12)
    gen_stage(c2, tmp_path, FakeGen(), MESSAGE_TYPING)
    label_stage(c2, tmp_path, AlternatingTransport(), BANK)
    with pytest.raises(SystemExit, match=r"split\.manifest|split stage|--stage split"):
        export_stage(c2, tmp_path, BANK)
    split_stage(c2, tmp_path)
    export_stage(c2, tmp_path, BANK)  # fresh again


def test_export_reports_states_in_no_split(tmp_path):
    c = built(tmp_path)
    for stage in ("gen", "label", "split"):  # no manifests: only the report is under test
        (tmp_path / f"{stage}.manifest.json").unlink()
    append_state(tmp_path)
    export_stage(c, tmp_path, BANK)
    assert json.loads((tmp_path / "export.manifest.json").read_text())["excluded"]["unassigned"] == 1


def test_a_stage_with_no_manifest_is_not_checked(tmp_path):
    c = built(tmp_path)
    (tmp_path / "label.manifest.json").unlink()
    with open(tmp_path / "labels.jsonl", "a") as f:
        f.write("")
    export_stage(c, tmp_path, BANK)


def test_compile_refuses_stale_labels(tmp_path):
    built(tmp_path)
    with open(tmp_path / "labels.jsonl", "a") as f:
        f.write(json.dumps({"id": "x", "model": "m1", "repeat": 0, "labels": {}}) + "\n")
    with pytest.raises(SystemExit, match="stale"):
        compile_runs([tmp_path])


def test_label_manifest_describes_the_file_when_the_cost_cap_ends_the_run(tmp_path):
    gen_stage(cfg(), tmp_path, FakeGen(), MESSAGE_TYPING)
    with pytest.raises(SystemExit, match="cost cap"):
        label_stage(cfg(cap=0.0007), tmp_path, AlternatingTransport(), BANK)  # $0.0003 a call: three calls
    m = json.loads((tmp_path / "label.manifest.json").read_text())
    assert m["outputs"]["labels.jsonl"] == sha256_file(tmp_path / "labels.jsonl")
    assert m["counts"]["label_rows"] == 3


def test_rows_carry_the_bank_and_preamble_digest_and_a_different_one_is_refused_at_export(tmp_path):
    c = built(tmp_path)
    rows = labels_store(tmp_path).rows()
    assert all(r["label_digest"] for r in rows)
    with pytest.raises(SystemExit, match="different domain, bank or preamble"):
        export_stage(c, tmp_path, BANK, preamble="another preamble")


def test_label_asks_again_for_rows_made_under_another_preamble_and_export_then_uses_the_new_ones(tmp_path):
    c = built(tmp_path)
    t = AlternatingTransport()
    label_stage(c, tmp_path, t, BANK, preamble="a new preamble")
    assert sum(t.calls.values()) == 6 * 4  # every row again
    export_stage(c, tmp_path, BANK, preamble="a new preamble")


def test_rows_from_before_the_digest_still_load(tmp_path):
    c = built(tmp_path)
    p = tmp_path / "labels.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    for r in rows:
        r.pop("label_digest")
    p.write_text(jsonl_text(rows))
    (tmp_path / "label.manifest.json").unlink()
    export_stage(c, tmp_path, BANK)
    assert sum(1 for _ in (tmp_path / "export/train.jsonl").read_text().splitlines()) > 0


def test_lowering_repeats_takes_effect_at_export(tmp_path):
    c = built(tmp_path)
    def mean_action(cc):
        export_stage(cc, tmp_path, BANK)
        row = json.loads((tmp_path / "export/train.jsonl").read_text().splitlines()[0])
        return json.loads(row["gold"])["kind"]["probabilities"]["action"]
    assert mean_action(c) == pytest.approx(0.5)  # (0.8 + 0.2) / 2
    assert mean_action(cfg(repeats=1)) == pytest.approx(0.8)


def test_export_writes_each_file_atomically_and_the_manifest_last(tmp_path, monkeypatch):
    c = built(tmp_path)
    order = []
    real_atomic, real_manifest = export_mod.write_text_atomic, export_mod.write_manifest
    monkeypatch.setattr(export_mod, "write_text_atomic", lambda p, t: (order.append(p.name), real_atomic(p, t)))
    monkeypatch.setattr(export_mod, "write_manifest", lambda *a, **k: (order.append("manifest"), real_manifest(*a, **k)))
    export_stage(c, tmp_path, BANK)
    assert order == ["train.jsonl", "holdout.jsonl", "test.jsonl", "manifest"]
    assert not list((tmp_path / "export").glob("*.tmp"))


# --- 2. prices ----------------------------------------------------------------------------------

def test_an_unpriced_label_model_is_refused_before_any_paid_call(tmp_path):
    c = PipelineConfig.model_validate({"name": "t", "gen": {"n": 3}, "label": {"models": ["mystery-model"], "reference": None}})
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    t = AlternatingTransport()
    with pytest.raises(UnpricedModelError, match="mystery-model.*PRICES"):
        label_stage(c, tmp_path, t, BANK)
    assert t.calls == {}


# --- 5. gen ---------------------------------------------------------------------------------------

def test_a_resumed_gen_continues_the_same_type_cycle(tmp_path):
    whole, resumed = tmp_path / "whole", tmp_path / "resumed"
    gen_stage(cfg(n=6), whole, FakeGen(), MESSAGE_TYPING)
    for n in (2, 4, 6):  # killed and resumed twice
        gen_stage(cfg(n=n), resumed, FakeGen(), MESSAGE_TYPING)
    types = lambda d: [r["intended_type"] for r in states_store(d).rows()]
    assert len(types(resumed)) == 6 and types(resumed) == types(whole)
    assert len(set(types(resumed))) == 6  # six different types, none repeated by a reshuffle


def test_gen_persists_the_generators_token_counts(tmp_path):
    client = FakeGen()
    gen_stage(cfg(n=4), tmp_path, client, MESSAGE_TYPING)
    rows = states_store(tmp_path).rows()
    assert sum(r["gen_tokens"]["in"] for r in rows) == client.tokens_in == 40
    m = json.loads((tmp_path / "gen.manifest.json").read_text())
    assert (m["generator_tokens_in"], m["generator_tokens_out"]) == (40, 12)


# --- 6. locks and duplicates ----------------------------------------------------------------------

def test_a_second_run_on_one_directory_is_refused(tmp_path):
    with run_lock(tmp_path, "label"):
        with pytest.raises(SystemExit, match="in use by another run"):
            gen_stage(cfg(), tmp_path, FakeGen(), MESSAGE_TYPING)
        with pytest.raises(SystemExit, match="in use by another run"):
            label_stage(cfg(), tmp_path, AlternatingTransport(), BANK)
    gen_stage(cfg(), tmp_path, FakeGen(), MESSAGE_TYPING)  # released


def test_latest_keeps_the_last_row_of_a_duplicated_key(tmp_path):
    s = JsonlStore(tmp_path / "a.jsonl", key=lambda r: r["id"])
    for r in ({"id": "x", "v": 1}, {"id": "y", "v": 1}, {"id": "x", "v": 2}):
        s.append(r)
    assert [(r["id"], r["v"]) for r in s.latest()] == [("x", 2), ("y", 1)]


def test_export_and_compile_do_not_double_weight_a_duplicated_label_row(tmp_path):
    c = built(tmp_path)
    p = tmp_path / "labels.jsonl"
    first = json.loads(p.read_text().splitlines()[0])
    with open(p, "a") as f:
        f.write(json.dumps(first) + "\n")
    (tmp_path / "label.manifest.json").unlink()
    export_stage(c, tmp_path, BANK)
    rows = compile_runs([tmp_path])
    assert sum(len(r["teachers"]) for r in rows) == 6 * 4
