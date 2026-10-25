import json

import pytest

from verdict.label.bank import BANK
from verdict.label.prompt_eval import score_preamble, SpendLedger
from verdict.label.teacher import Teacher

GOOD = {
    "kind": {"action": 0.7, "synthesis": 0.1, "decision": 0.05, "thought": 0.05, "draft": 0.05, "note": 0.03, "other": 0.02},
    "records_decision": {"true": 0.1, "false": 0.9},
    "reports_shipped_work": {"true": 0.9, "false": 0.1},
    "leaves_open_question": {"true": 0.1, "false": 0.9},
    "is_proposal": {"true": 0.1, "false": 0.9},
}
REF = {"kind": {"action": 0.6, "synthesis": 0.2, "decision": 0.05, "thought": 0.05, "draft": 0.05, "note": 0.03, "other": 0.02},
       "records_decision": 0.05, "reports_shipped_work": 0.95, "leaves_open_question": 0.2, "is_proposal": 0.1}


class T:
    def __init__(self):
        self.calls = 0

    def __call__(self, model, prompt):
        self.calls += 1
        return json.dumps(GOOD), {"promptTokenCount": 1000, "candidatesTokenCount": 500}


def states(n=3):
    return [{"id": f"s{i}", "state": {"message": f"m{i}"}} for i in range(n)]


def refs(n=3):
    return {f"s{i}": REF for i in range(n)}


def test_perfect_agreement_scores_100(tmp_path):
    t = T()
    r = score_preamble(states(), refs(), Teacher("gemini-2.5-flash-lite", t, preamble="P"), BANK,
                       cache_path=tmp_path / "c.jsonl", ledger=SpendLedger(tmp_path / "l.json", cap=5))
    assert r["score"] == 100.0 and r["n"] == 3 and t.calls == 3


def test_identical_preamble_is_served_from_cache_and_costs_nothing(tmp_path):
    t = T()
    kw = dict(cache_path=tmp_path / "c.jsonl", ledger=SpendLedger(tmp_path / "l.json", cap=5))
    score_preamble(states(), refs(), Teacher("gemini-2.5-flash-lite", t, preamble="P"), BANK, **kw)
    first = kw["ledger"].total()
    r = score_preamble(states(), refs(), Teacher("gemini-2.5-flash-lite", t, preamble="P"), BANK, **kw)
    assert t.calls == 3 and kw["ledger"].total() == first and r["cached"] == 3


def test_a_different_preamble_is_not_a_cache_hit(tmp_path):
    t = T()
    kw = dict(cache_path=tmp_path / "c.jsonl", ledger=SpendLedger(tmp_path / "l.json", cap=5))
    score_preamble(states(), refs(), Teacher("gemini-2.5-flash-lite", t, preamble="P1"), BANK, **kw)
    score_preamble(states(), refs(), Teacher("gemini-2.5-flash-lite", t, preamble="P2"), BANK, **kw)
    assert t.calls == 6


def test_ledger_persists_and_the_cap_stops_spending(tmp_path):
    ledger = SpendLedger(tmp_path / "l.json", cap=0.0)
    with pytest.raises(SystemExit):
        score_preamble(states(), refs(), Teacher("gemini-2.5-flash-lite", T(), preamble="P"), BANK,
                       cache_path=tmp_path / "c.jsonl", ledger=ledger)
    ledger2 = SpendLedger(tmp_path / "l2.json", cap=5)
    ledger2.add(1.25)
    assert SpendLedger(tmp_path / "l2.json", cap=5).total() == 1.25
