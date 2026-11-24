import json

from verdict.evaluation.metrics import bootstrap_ci, mean_ci
from verdict.label.teacher import GEMINI_URL
from verdict.pipeline.audit import SPLITS, split_of
from verdict.pipeline.store import (JsonlStore, jsonl_text, label_key, labels_store,
                                    states_store, write_json_atomic, write_text_atomic)


def test_label_key_and_stores(tmp_path):
    assert label_key("a", "m", 2) == "a|m|2"
    s = states_store(tmp_path)
    s.append({"id": "x"})
    assert states_store(tmp_path).done == {"x"}
    lab = labels_store(tmp_path)
    lab.append({"id": "x", "model": "m", "repeat": 0})
    assert labels_store(tmp_path).done == {"x|m|0"}
    other = tmp_path / "other.jsonl"
    assert labels_store(tmp_path, other).path == other


def test_split_of():
    assert SPLITS == ("train", "holdout", "test")
    assert split_of({"train": ["a"], "test": ["b"]}) == {"a": "train", "b": "test"}


def test_atomic_bytes(tmp_path):
    p = tmp_path / "d" / "o.json"
    write_json_atomic(p, {"b": 1, "a": "é"})
    assert p.read_bytes() == json.dumps({"b": 1, "a": "é"}, indent=1, sort_keys=True).encode()
    assert jsonl_text([{"a": "é"}, {"b": 2}]) == '{"a": "é"}\n{"b": 2}\n'
    write_text_atomic(p, "x")
    assert p.read_text() == "x"


def test_torn_tail_rewrite(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text('{"id": "a"}\n{"id": "b')
    JsonlStore(p, key=lambda r: r["id"])
    assert p.read_text() == '{"id": "a"}\n'


def test_mean_ci_matches_bootstrap():
    hits = [True, False, True, True, False, True, True, True]
    assert mean_ci(hits, iters=1000) == bootstrap_ci(hits, lambda h: sum(h) / len(h), iters=1000)


def test_gemini_url():
    assert GEMINI_URL.format(model="m").endswith("/models/m:generateContent")
