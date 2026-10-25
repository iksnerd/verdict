import json
import math

import pytest

from verdict.label.bank import BANK
from verdict.label.teacher import (
    LabelError,
    Teacher,
    agreement,
    build_prompt,
    load_key,
    mean_labels,
    parse_labels,
)


def good():
    return {
        "kind": {"action": 0.6, "synthesis": 0.1, "decision": 0.1, "thought": 0.05, "draft": 0.05, "note": 0.05, "other": 0.05},
        "records_decision": {"true": 0.2, "false": 0.8},
        "reports_shipped_work": {"true": 0.9, "false": 0.1},
        "leaves_open_question": {"true": 0.1, "false": 0.9},
        "is_proposal": {"true": 0.05, "false": 0.95},
    }


def test_bank_covers_all_question_types():
    assert {q["type"] for q in BANK.values()} == {"choice", "noul"}
    assert "kind" in BANK


def test_prompt_contains_state_and_every_question():
    p = build_prompt({"message": "shipped it"}, BANK)
    assert "shipped it" in p
    assert all(qid in p for qid in BANK)


def test_parse_normalises_and_maps_noul():
    raw = json.dumps({**good(), "kind": {k: v * 2 for k, v in good()["kind"].items()}})
    out = parse_labels(raw, BANK)
    assert math.isclose(sum(out["kind"].values()), 1.0, abs_tol=1e-6)
    assert 0 <= out["records_decision"] <= 1  # noul collapses to P(true)


def test_parse_strips_code_fences():
    out = parse_labels("```json\n" + json.dumps(good()) + "\n```", BANK)
    assert set(out) == set(BANK)


@pytest.mark.parametrize("mutate", [lambda d: d.pop("kind"), lambda d: d["kind"].pop("action"), lambda d: d.update(kind={"action": 0, "note": 0})])
def test_parse_rejects_incomplete_or_zero_mass(mutate):
    d = good()
    mutate(d)
    with pytest.raises(LabelError):
        parse_labels(json.dumps(d), BANK)


def test_mean_labels_averages_distributions():
    a = {"kind": {"x": 1.0, "y": 0.0}, "n": 0.2}
    b = {"kind": {"x": 0.0, "y": 1.0}, "n": 0.6}
    m = mean_labels([a, b])
    assert m["kind"] == {"x": 0.5, "y": 0.5} and math.isclose(m["n"], 0.4)


def test_agreement_argmax_and_noul_threshold():
    a = {"kind": {"x": 0.9, "y": 0.1}, "n": 0.8}
    b = {"kind": {"x": 0.6, "y": 0.4}, "n": 0.7}
    c = {"kind": {"x": 0.2, "y": 0.8}, "n": 0.1}
    assert agreement(a, b) == {"kind": 1.0, "n": 1.0}
    assert agreement(a, c) == {"kind": 0.0, "n": 0.0}


class FakeTransport:
    def __init__(self, texts):
        self.texts, self.calls = list(texts), 0

    def __call__(self, model, prompt):
        self.calls += 1
        return self.texts.pop(0), {"promptTokenCount": 100, "candidatesTokenCount": 50}


def test_teacher_retries_once_on_bad_json_then_succeeds():
    t = FakeTransport(["not json", json.dumps(good())])
    teacher = Teacher("gemini-2.5-flash-lite", t)
    out = teacher.label({"message": "x"}, BANK)
    assert t.calls == 2 and set(out) == set(BANK)
    assert teacher.tokens_in == 200 and teacher.tokens_out == 100


def test_teacher_raises_after_two_failures():
    with pytest.raises(LabelError):
        Teacher("m", FakeTransport(["bad", "worse"])).label({"message": "x"}, BANK)


def test_load_key_reads_only_the_named_variable_from_a_file(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    f = tmp_path / "env"
    f.write_text("HF_TOKEN=nope\nGOOGLE_API_KEY=abc123\nOTHER=zzz\n")
    assert load_key(str(f)) == "abc123"


def test_load_key_errors_clearly_when_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    f = tmp_path / "env"
    f.write_text("HF_TOKEN=nope\n")
    with pytest.raises(LabelError):
        load_key(str(f))


def test_custom_preamble_replaces_the_default_instructions():
    p = build_prompt({"message": "x"}, BANK, preamble="CUSTOM RULES HERE")
    assert "CUSTOM RULES HERE" in p and "honest" not in p
    assert "ENTRY:" in p and "QUESTIONS:" in p  # the fixed frame is not editable


def test_teacher_uses_its_preamble():
    seen = {}

    def transport(model, prompt):
        seen["p"] = prompt
        return json.dumps(good()), {}

    Teacher("m", transport, preamble="ONLY THIS").label({"message": "x"}, BANK)
    assert "ONLY THIS" in seen["p"]


def test_a_network_timeout_is_a_label_error_so_the_stage_carries_on(monkeypatch):
    # a ReadTimeout escaping the transport ended two label runs early, still exiting 0
    import httpx

    from verdict.label.teacher import LabelError, gemini_transport

    def slow(*a, **k):
        raise httpx.ReadTimeout("The read operation timed out")

    monkeypatch.setattr(httpx, "post", slow)
    with pytest.raises(LabelError, match="timed out"):
        gemini_transport("k")("gemini-2.5-flash-lite", "prompt")
