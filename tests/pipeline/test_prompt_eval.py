import pytest

from verdict.label.bank import BANK
from verdict.label.prompt_eval import score_preamble, SpendLedger
from verdict.label.teacher import Teacher

from fakes import PaidGarbage
from pfakes import GoodTransport as T

REF = {"kind": {"action": 0.6, "synthesis": 0.2, "decision": 0.05, "thought": 0.05, "draft": 0.05, "note": 0.03, "other": 0.02},
       "records_decision": 0.05, "reports_shipped_work": 0.95, "leaves_open_question": 0.2, "is_proposal": 0.1}


def states(n=3):
    return [{"id": f"s{i}", "state": {"message": f"m{i}"}} for i in range(n)]


def refs(n=3):
    return {f"s{i}": REF for i in range(n)}


def run_score(tmp_path, t, preamble="P", model="gemini-2.5-flash-lite", cap=5, n_states=3, n_refs=None,
              bank=BANK, prices=None, ledger=None, cache_path=None):
    """score_preamble on invented states. A test that reuses the cache or ledger across calls passes them in."""
    teacher = Teacher(model, t, preamble=preamble, **({} if prices is None else {"prices": prices}))
    return score_preamble(states(n_states), refs(n_states if n_refs is None else n_refs), teacher, bank,
                          cache_path=cache_path if cache_path is not None else tmp_path / "c.jsonl",
                          ledger=ledger if ledger is not None else SpendLedger(tmp_path / "l.json", cap=cap))


def test_perfect_agreement_scores_100(tmp_path):
    t = T()
    r = run_score(tmp_path, t)
    assert r["score"] == 100.0 and r["n"] == 3 and t.calls == 3


def test_identical_preamble_is_served_from_cache_and_costs_nothing(tmp_path):
    t = T()
    kw = dict(cache_path=tmp_path / "c.jsonl", ledger=SpendLedger(tmp_path / "l.json", cap=5))
    run_score(tmp_path, t, **kw)
    first = kw["ledger"].total()
    r = run_score(tmp_path, t, **kw)
    assert t.calls == 3 and kw["ledger"].total() == first and r["cached"] == 3


def test_a_different_preamble_is_not_a_cache_hit(tmp_path):
    t = T()
    kw = dict(cache_path=tmp_path / "c.jsonl", ledger=SpendLedger(tmp_path / "l.json", cap=5))
    run_score(tmp_path, t, preamble="P1", **kw)
    run_score(tmp_path, t, preamble="P2", **kw)
    assert t.calls == 6


def test_ledger_persists_and_the_cap_stops_spending(tmp_path):
    ledger = SpendLedger(tmp_path / "l.json", cap=0.0)
    with pytest.raises(SystemExit):
        run_score(tmp_path, T(), ledger=ledger)
    ledger2 = SpendLedger(tmp_path / "l2.json", cap=5)
    ledger2.add(1.25)
    assert SpendLedger(tmp_path / "l2.json", cap=5).total() == 1.25


# gemini-2.5-flash-lite: (1000 * 0.10 + 500 * 0.40) / 1e6 = $0.0003 a try, two tries a failed label
FAILED_LABEL_USD = 0.0006


def test_a_failed_label_is_charged_to_the_ledger(tmp_path):
    ledger = SpendLedger(tmp_path / "l.json", cap=5)
    r = run_score(tmp_path, PaidGarbage(), ledger=ledger)
    assert r["failures"] == 3 and r["n"] == 0
    assert ledger.total() == pytest.approx(3 * FAILED_LABEL_USD)
    assert r["spend_usd_this_call"] == pytest.approx(3 * FAILED_LABEL_USD)
    assert SpendLedger(tmp_path / "l.json", cap=5).total() == pytest.approx(3 * FAILED_LABEL_USD)


def test_failed_labels_alone_can_reach_the_spend_cap(tmp_path):
    t = PaidGarbage()
    with pytest.raises(SystemExit):
        run_score(tmp_path, t, cap=0.001)
    assert t.calls == 4  # two failed labels ($0.0012) cross the cap; the third state is never asked
