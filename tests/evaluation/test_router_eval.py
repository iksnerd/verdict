"""`router_eval` is what both the CLI and any scorecard call, so its arithmetic is tested
here rather than trusted. No model: the scoring takes cached outputs, which is the same path a
scorecard replay uses."""
from __future__ import annotations

import pytest

from verdict.judge.router import BIG, SMALL, T_DIFFICULTY, T_SENSITIVE
from verdict.evaluation.router_eval import COST_RATIO, as_answers, cost, evaluate, fit, load_prompts, score

from efakes import engine  # noqa: F401  (fixture)


def rows_and_probs():
    """Four prompts: two easy, two hard, with outputs chosen so the shipped cuts get them right."""
    rows = [
        {"id": "s1", "prompt": "easy one", "want": SMALL, "split": "train"},
        {"id": "s2", "prompt": "easy two", "want": SMALL, "split": "test"},
        {"id": "b1", "prompt": "hard one", "want": BIG, "split": "train"},
        {"id": "b2", "prompt": "hard two", "want": BIG, "split": "test"},
    ]
    probs = {
        "s1": {"difficulty": 0.5, "is_sensitive": 0.1},
        "s2": {"difficulty": 0.6, "is_sensitive": 0.1},
        "b1": {"difficulty": 2.5, "is_sensitive": 0.1},
        "b2": {"difficulty": 2.6, "is_sensitive": 0.1},
    }
    return rows, probs


def test_a_perfect_router_scores_one_hundred_and_costs_nothing():
    rows, probs = rows_and_probs()
    s = score(rows, probs)
    assert s["accuracy"] == 1.0 and s["cost"] == 0.0
    assert s["hard_to_small"] == 0 and s["easy_to_big"] == 0 and not s["wrong"]


def test_the_two_mistakes_are_counted_separately():
    rows, probs = rows_and_probs()
    probs["b1"]["difficulty"] = 0.1      # a hard prompt that looks easy -> small
    probs["s1"]["difficulty"] = 2.9      # an easy prompt that looks hard -> big
    s = score(rows, probs)
    assert s["hard_to_small"] == 1 and s["easy_to_big"] == 1
    assert len(s["wrong"]) == 2


def test_cost_weights_the_expensive_mistake_more():
    """The whole reason `fit` minimises cost: these two are not equally bad."""
    rows, probs = rows_and_probs()
    hard_to_small = dict(probs, b1={"difficulty": 0.1, "is_sensitive": 0.1})
    easy_to_big = dict(probs, s1={"difficulty": 2.9, "is_sensitive": 0.1})
    c_expensive = cost(rows, hard_to_small, T_DIFFICULTY, T_SENSITIVE)
    c_cheap = cost(rows, easy_to_big, T_DIFFICULTY, T_SENSITIVE)
    assert c_expensive == pytest.approx(c_cheap * COST_RATIO)


def test_as_answers_is_never_mistaken_for_a_uniform_backend():
    """A flat placeholder here would make every replayed prompt read as 'no model behind this',
    which the router turns into BIG, silently scoring the cache instead of the model."""
    from verdict.judge.router import is_uniform

    assert not is_uniform(as_answers({"difficulty": 1.5, "is_sensitive": 0.6}))


def test_fit_returns_thresholds_inside_the_question_ranges():
    rows, probs = rows_and_probs()
    c, td, ts = fit(rows, probs)
    assert 0.0 <= td <= 3.0 and 0.0 <= ts <= 1.0 and c >= 0.0


def test_fit_separates_a_separable_set():
    rows, probs = rows_and_probs()
    c, td, ts = fit(rows, probs)
    assert c == 0.0
    assert score(rows, probs, td, ts)["accuracy"] == 1.0


def test_evaluate_reports_the_held_out_split_not_the_whole_set():
    rows, probs = rows_and_probs()
    r = evaluate(rows, probs)
    assert r["n_train"] == 2 and r["n_test"] == 2
    assert r["held_out"]["n"] == 2          # not 4
    assert r["score"] == 100.0
    assert r["chance"] == 50.0


def test_evaluate_puts_the_score_where_the_scorecard_reads_it():
    """A scorecard reads `score` as a share out of 100."""
    rows, probs = rows_and_probs()
    r = evaluate(rows, probs)
    assert isinstance(r["score"], float) and 0.0 <= r["score"] <= 100.0
    lo, hi = r["held_out_ci_95"]
    assert 0.0 <= lo <= r["score"] <= hi <= 100.0


def test_evaluate_reports_auc_per_question():
    rows, probs = rows_and_probs()
    r = evaluate(rows, probs)
    assert set(r["auc"]) == {"difficulty", "is_sensitive"}
    assert r["auc"]["difficulty"] == 1.0    # perfectly separating, and lower means smaller


def test_the_real_eval_set_loads_and_is_split():
    rows = load_prompts()
    assert len(rows) == 72
    assert {r["split"] for r in rows} == {"train", "test"}


def test_an_empty_eval_file_is_an_error_not_a_score_of_zero(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_prompts(p)


def test_the_refit_runs_as_a_module(capsys):
    """`python -m verdict.evaluation.router_eval --refit` is how the router's thresholds are refitted."""
    import pytest

    from verdict.evaluation import router_eval

    with pytest.raises(SystemExit) as exit_:
        router_eval.main(["--help"])
    assert exit_.value.code == 0
    assert "--refit" in capsys.readouterr().out


# --- the replay cache: an entry is only as good as what it was measured for ---

def test_a_full_matching_cache_loads_no_model(tmp_path, engine):
    from verdict.evaluation.router_eval import measure

    rows, _ = rows_and_probs()
    cache = tmp_path / "probs.json"
    first, times = measure(rows, "m", cache)
    assert engine.loads == 1 and len(times) == 4

    again, times = measure(rows, "m", cache)
    assert engine.loads == 1 and times == []
    assert {r["id"]: again[r["id"]]["difficulty"] for r in rows} == \
           {r["id"]: first[r["id"]]["difficulty"] for r in rows}


def test_an_edited_prompt_under_the_same_id_is_measured_again(tmp_path, engine):
    from verdict.evaluation.router_eval import measure

    rows, _ = rows_and_probs()
    cache = tmp_path / "probs.json"
    measure(rows, "m", cache)
    rows[0] = dict(rows[0], prompt="a much longer and different prompt")
    engine.asked.clear()

    probs, times = measure(rows, "m", cache)
    assert engine.asked == ["a much longer and different prompt"]
    assert len(times) == 1
    assert probs["s1"]["difficulty"] == float(len("a much longer and different prompt"))


def test_a_changed_question_bank_misses_every_entry(tmp_path, engine, monkeypatch):
    from verdict.evaluation import router_eval

    rows, _ = rows_and_probs()
    cache = tmp_path / "probs.json"
    router_eval.measure(rows, "m", cache)
    bank = {q: dict(spec) for q, spec in router_eval.ROUTER_BANK.items()}
    bank["difficulty"]["instructions"] = "A reworded question."
    monkeypatch.setattr(router_eval, "ROUTER_BANK", bank)
    engine.asked.clear()

    _, times = router_eval.measure(rows, "m", cache)
    assert len(engine.asked) == 4 and len(times) == 4


def test_a_changed_token_budget_misses_every_entry(tmp_path, engine, monkeypatch):
    from verdict.evaluation import router_eval

    rows, _ = rows_and_probs()
    cache = tmp_path / "probs.json"
    router_eval.measure(rows, "m", cache)
    monkeypatch.setattr(router_eval, "PROMPT_TOKEN_BUDGET", router_eval.PROMPT_TOKEN_BUDGET * 2)
    engine.asked.clear()
    engine.budgets.clear()

    router_eval.measure(rows, "m", cache)
    assert len(engine.asked) == 4
    # The budget in the fingerprint is the one the prompt was actually clipped to.
    assert set(engine.budgets) == {router_eval.PROMPT_TOKEN_BUDGET}


def test_an_entry_with_no_fingerprint_is_measured_again_and_the_rest_is_left_alone(tmp_path, engine):
    import json

    from verdict.evaluation.router_eval import measure

    rows, _ = rows_and_probs()
    cache = tmp_path / "probs.json"
    old = {"difficulty": 9.0, "is_sensitive": 0.9}
    cache.write_text(json.dumps({"m": {"s1": old, "gone": old}, "other-model": {"s1": old}}))

    probs, times = measure(rows, "m", cache)
    assert len(times) == 4 and probs["s1"]["difficulty"] == float(len("easy one"))
    stored = json.loads(cache.read_text())
    assert stored["m"]["gone"] == old            # not asked for in this run: untouched
    assert stored["other-model"] == {"s1": old}
    assert stored["m"]["s1"]["difficulty"] == float(len("easy one"))


def test_an_interrupted_run_keeps_what_it_measured(tmp_path, engine):
    import json

    from verdict.evaluation.router_eval import measure

    rows, _ = rows_and_probs()
    cache = tmp_path / "probs.json"
    engine.die_after = 2
    with pytest.raises(KeyboardInterrupt):
        measure(rows, "m", cache)
    assert set(json.loads(cache.read_text())["m"]) == {"s1", "s2"}
    assert [p.name for p in tmp_path.iterdir()] == ["probs.json"]   # no temp file left behind

    engine.die_after = None
    engine.asked.clear()
    probs, times = measure(rows, "m", cache)
    assert engine.asked == ["hard one", "hard two"] and len(times) == 2
    assert set(probs) >= {"s1", "s2", "b1", "b2"}
