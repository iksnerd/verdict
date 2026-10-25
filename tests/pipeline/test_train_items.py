import json
from pathlib import Path

import pytest

pytest.importorskip("laya")
from huggingface_hub import snapshot_download  # noqa: E402

from verdict.label.bank import BANK  # noqa: E402
from verdict.train.items import build_items  # noqa: E402


@pytest.fixture(scope="module")
def model_dir():
    try:
        return Path(snapshot_download("convaiinnovations/laya", allow_patterns=["tokenizer/*", "rl_agent_config.json", "encoder/*"], local_files_only=True))
    except Exception:
        pytest.skip("laya tokenizer not in the local HF cache")


def row(**over):
    gold = {"kind": {"probabilities": {"action": 0.6, "synthesis": 0.4, "decision": 0, "thought": 0, "draft": 0, "note": 0, "other": 0}}}
    for q in ("records_decision", "reports_shipped_work", "leaves_open_question", "is_proposal"):
        gold[q] = {"probabilities": {"true": 0.9, "false": 0.1}}
    return {"id": "x", "workflow": "t", "state": json.dumps({"message": "Shipped the fix and added a test."}),
            "questions": json.dumps(BANK), "gold": json.dumps(gold), **over}


def test_one_item_per_question_with_normalised_targets(tmp_path, model_dir):
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(row()) + "\n")
    items, dropped = build_items(p, model_dir)
    assert len(items) == 5 and dropped == 0
    assert all(abs(sum(i["target"]) - 1) < 1e-6 for i in items)
    assert all(len(i["markers"]) == len(i["target"]) for i in items)
    assert items[0]["label"] == 0  # 'action' has the highest mass


def test_a_question_missing_from_gold_is_skipped(tmp_path, model_dir):
    r = row()
    g = json.loads(r["gold"])
    g.pop("is_proposal")
    r["gold"] = json.dumps(g)
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps(r) + "\n")
    assert len(build_items(p, model_dir)[0]) == 4
