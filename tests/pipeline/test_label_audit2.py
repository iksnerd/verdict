"""Spend ledger, cache key, reference check, incremental label output (audit 2). Invented data only."""
import json
import runpy
import sys

import pytest

from verdict.label import teacher as teacher_mod
from verdict.label.bank import BANK
from verdict.label.prompt_eval import SpendLedger, score_preamble
from verdict.label.teacher import Teacher, UnpricedModelError, cost_of
from fakes import GOOD_LABELS as GOOD  # noqa: E402
from test_prompt_eval import REF, T, refs, states  # noqa: E402  (same directory, invented data)


def test_an_unknown_model_has_no_price_unless_the_config_states_one():
    with pytest.raises(UnpricedModelError, match="mystery.*PRICES"):
        cost_of("mystery", 1000, 1000)
    assert cost_of("mystery", 1_000_000, 0, {"mystery": [2.0, 3.0]}) == pytest.approx(2.0)
    with pytest.raises(UnpricedModelError):
        Teacher("mystery", lambda m, p: ("", {})).cost_usd()


def test_score_preamble_refuses_an_unpriced_teacher_before_any_call(tmp_path):
    t = T()
    with pytest.raises(UnpricedModelError):
        score_preamble(states(), refs(), Teacher("mystery", t, preamble="P"), BANK,
                       cache_path=tmp_path / "c.jsonl", ledger=SpendLedger(tmp_path / "l.json", cap=5))
    assert t.calls == 0


def test_a_priced_in_config_teacher_charges_the_ledger(tmp_path):
    ledger = SpendLedger(tmp_path / "l.json", cap=5)
    score_preamble(states(), refs(), Teacher("mystery", T(), preamble="P", prices={"mystery": [1.0, 1.0]}), BANK,
                   cache_path=tmp_path / "c.jsonl", ledger=ledger)
    assert ledger.total() == pytest.approx(3 * 1500 / 1e6)


def test_two_ledger_objects_on_one_file_add_up(tmp_path):
    a, b = SpendLedger(tmp_path / "l.json", cap=5), SpendLedger(tmp_path / "l.json", cap=5)
    a.add(1.0)
    b.add(2.0)
    assert a.total() == b.total() == 3.0


def test_the_cap_sees_what_another_run_spent(tmp_path):
    a, b = SpendLedger(tmp_path / "l.json", cap=1.5), SpendLedger(tmp_path / "l.json", cap=1.5)
    a.check()
    b.add(2.0)
    with pytest.raises(SystemExit, match="spend cap"):
        a.check()


def test_a_changed_bank_is_asked_again(tmp_path):
    t = T()
    kw = dict(cache_path=tmp_path / "c.jsonl", ledger=SpendLedger(tmp_path / "l.json", cap=5))
    score_preamble(states(), refs(), Teacher("gemini-2.5-flash-lite", t, preamble="P"), BANK, **kw)
    assert t.calls == 3
    other = {**BANK, "kind": {**BANK["kind"], "instructions": "A different question about `message`?"}}
    score_preamble(states(), refs(), Teacher("gemini-2.5-flash-lite", t, preamble="P"), other, **kw)
    assert t.calls == 6


def test_a_missing_reference_is_a_clear_error_before_any_paid_call(tmp_path):
    t = T()
    with pytest.raises(KeyError, match="no reference label"):
        score_preamble(states(3), refs(2), Teacher("gemini-2.5-flash-lite", t, preamble="P"), BANK,
                       cache_path=tmp_path / "c.jsonl", ledger=SpendLedger(tmp_path / "l.json", cap=5))
    assert t.calls == 0


def run_label_main(monkeypatch, tmp_path, transport, extra=()):
    sp = tmp_path / "s.jsonl"
    sp.write_text("".join(json.dumps({"id": f"s{i}", "intended_type": "note", "state": {"message": f"m{i}"}}) + "\n"
                          for i in range(3)))
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(teacher_mod, "load_key", lambda *a: "k")
    monkeypatch.setattr(teacher_mod, "gemini_transport", lambda key: transport)
    monkeypatch.setattr(sys, "argv", ["verdict.label", "--states", str(sp), "--models", "gemini-2.5-flash-lite",
                                      "--reference", "gemini-3.1-pro-preview", "--repeats", "1", "--out", str(out), *extra])
    runpy.run_module("verdict.label", run_name="__main__")
    return out


def test_label_main_keeps_every_paid_label_when_interrupted(monkeypatch, tmp_path):
    calls = []

    def transport(model, prompt):
        calls.append(model)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return json.dumps(GOOD), {"promptTokenCount": 10, "candidatesTokenCount": 5}

    with pytest.raises(KeyboardInterrupt):
        run_label_main(monkeypatch, tmp_path, transport)
    first = json.loads((tmp_path / "out.jsonl").read_text().splitlines()[0])
    assert first["id"] == "s0" and set(first["labels"]) == {"gemini-2.5-flash-lite", "gemini-3.1-pro-preview"}


def test_label_main_refuses_an_unpriced_model(monkeypatch, tmp_path, capsys):
    sp = tmp_path / "s.jsonl"
    sp.write_text("")
    monkeypatch.setattr(teacher_mod, "load_key", lambda *a: "k")
    monkeypatch.setattr(teacher_mod, "gemini_transport", lambda key: (lambda m, p: pytest.fail("no call")))
    monkeypatch.setattr(sys, "argv", ["verdict.label", "--states", str(sp), "--models", "mystery", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="no price for model 'mystery'"):
        runpy.run_module("verdict.label", run_name="__main__")
