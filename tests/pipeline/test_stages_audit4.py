"""Audit 4: relabel idempotence, label manifest completeness, atomic temp names, stage locks. Invented data only."""
import json
import os

import pytest

from verdict.gen.domains import MESSAGE_TYPING
from verdict.label.bank import BANK
from verdict.pipeline.compile import compile_runs
from verdict.pipeline.config import PipelineConfig
from verdict.pipeline.export import export_stage
from verdict.pipeline.stages import gen_stage, label_stage, split_stage
from verdict.pipeline.store import run_lock, write_text_atomic

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


def reply(a):
    rest = (1 - a) / 6
    return {"kind": {"action": a, "synthesis": rest, "decision": rest, "thought": rest,
                     "draft": rest, "note": rest, "other": rest},
            "records_decision": {"true": 0.1, "false": 0.9}, "reports_shipped_work": {"true": 0.9, "false": 0.1},
            "leaves_open_question": {"true": 0.1, "false": 0.9}, "is_proposal": {"true": 0.1, "false": 0.9}}


class Transport:
    def __init__(self):
        self.calls = {}

    def __call__(self, model, prompt):
        n = self.calls[model] = self.calls.get(model, 0) + 1
        return json.dumps(reply(0.8 if (n - 1) % 2 == 0 else 0.2)), {"promptTokenCount": 1000, "candidatesTokenCount": 500}


def nrows(p):
    return len((p / "labels.jsonl").read_text().splitlines())


def test_a_relabelled_key_is_not_asked_again_on_the_next_run(tmp_path):
    c = cfg()
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    label_stage(c, tmp_path, Transport(), BANK)
    label_stage(c, tmp_path, Transport(), BANK, preamble="a new preamble")
    before = nrows(tmp_path)
    t = Transport()
    label_stage(c, tmp_path, t, BANK, preamble="a new preamble")
    assert t.calls == {}
    assert nrows(tmp_path) == before


def test_label_manifest_says_it_is_incomplete_after_a_cap_stop_and_export_warns(tmp_path, capsys):
    c = cfg()
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    with pytest.raises(SystemExit, match="cost cap"):
        label_stage(cfg(cap=0.0007), tmp_path, Transport(), BANK)
    m = json.loads((tmp_path / "label.manifest.json").read_text())
    assert m["complete"] is False
    assert m["unlabelled_states"] == 6
    split_stage(c, tmp_path)
    capsys.readouterr()
    export_stage(c, tmp_path, BANK)
    err = capsys.readouterr().err
    assert "unlabelled" in err and "label" in err
    assert json.loads((tmp_path / "export.manifest.json").read_text())["excluded"]["unlabelled"] > 0


def test_a_finished_label_run_is_complete_and_export_is_quiet(tmp_path, capsys):
    c = cfg()
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    label_stage(c, tmp_path, Transport(), BANK)
    m = json.loads((tmp_path / "label.manifest.json").read_text())
    assert m["complete"] is True and m["unlabelled_states"] == 0
    split_stage(c, tmp_path)
    capsys.readouterr()
    export_stage(c, tmp_path, BANK)
    assert capsys.readouterr().err == ""


def test_an_old_label_manifest_without_the_new_keys_does_not_warn(tmp_path, capsys):
    c = cfg()
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    label_stage(c, tmp_path, Transport(), BANK)
    split_stage(c, tmp_path)
    p = tmp_path / "label.manifest.json"
    m = json.loads(p.read_text())
    m.pop("complete"), m.pop("unlabelled_states")
    p.write_text(json.dumps(m))
    capsys.readouterr()
    export_stage(c, tmp_path, BANK)
    assert capsys.readouterr().err == ""


def test_a_row_without_token_counts_does_not_mask_the_original_stop(tmp_path):
    c = cfg()
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    label_stage(c, tmp_path, Transport(), BANK)
    p = tmp_path / "labels.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    rows[0].pop("tokens_in")
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "label.manifest.json").unlink()
    label_stage(cfg(cap=0.0), tmp_path, Transport(), BANK)  # nothing left to ask: must not raise KeyError


def test_atomic_writes_do_not_share_a_temp_name(tmp_path, monkeypatch):
    seen = []
    real = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (seen.append(str(a)), real(a, b)))
    write_text_atomic(tmp_path / "f.txt", "a")
    write_text_atomic(tmp_path / "f.txt", "b")
    assert len(set(seen)) == 2 and all(s != str(tmp_path / "f.txt.tmp") for s in seen)
    assert not list(tmp_path.glob("*.tmp*")) and (tmp_path / "f.txt").read_text() == "b"


def test_atomic_write_cleans_its_temp_on_failure(tmp_path, monkeypatch):
    def boom(a, b):
        raise OSError("no")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        write_text_atomic(tmp_path / "f.txt", "a")
    assert list(tmp_path.iterdir()) == []


def _built(tmp_path):
    c = cfg()
    gen_stage(c, tmp_path, FakeGen(), MESSAGE_TYPING)
    label_stage(c, tmp_path, Transport(), BANK)
    split_stage(c, tmp_path)
    return c


def test_split_export_and_compile_refuse_a_locked_run(tmp_path):
    c = _built(tmp_path)
    import subprocess, sys
    holder = subprocess.Popen([sys.executable, "-c",
        "import sys,time;from verdict.pipeline.store import run_lock\n"
        "with run_lock(sys.argv[1]):\n print('x',flush=True);time.sleep(30)", str(tmp_path)],
        stdout=subprocess.PIPE, text=True)
    try:
        holder.stdout.readline()
        with pytest.raises(SystemExit, match="in use"):
            split_stage(c, tmp_path)
        with pytest.raises(SystemExit, match="in use"):
            export_stage(c, tmp_path, BANK)
        with pytest.raises(SystemExit, match="in use"):
            compile_runs([tmp_path])
    finally:
        holder.kill()
        holder.wait()


def test_stages_release_the_lock_so_they_run_in_sequence(tmp_path):
    c = _built(tmp_path)
    export_stage(c, tmp_path, BANK)
    compile_runs([tmp_path])
    split_stage(c, tmp_path)
    with run_lock(tmp_path):
        pass
