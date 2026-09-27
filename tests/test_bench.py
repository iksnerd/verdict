"""`verdict bench`: the answer-quality scorecard. No model and no network here: the dataset loader
and the model are both faked, so these check the bookkeeping, which is what a gate must get right."""
from __future__ import annotations

import json

import pytest

from verdict import bench


def suite(**over):
    base = {
        "name": "toy", "dataset": "someone/toy", "revision": "abc123", "split": "test",
        "text": "sentence", "label": "label", "classes": {"0": "negative", "1": "positive"},
        "question": {"type": "noul", "instructions": "Is `text` positive?"}, "target": "positive",
        "per_class": 3, "seed": 0,
    }
    base.update(over)
    return bench.Suite.from_dict(base)


ROWS = [{"sentence": f"good {i}", "label": 1} for i in range(10)] + \
       [{"sentence": f"bad {i}", "label": 0} for i in range(10)] + \
       [{"sentence": "  ", "label": 1}]


def test_sample_is_balanced_seeded_and_skips_blank_text():
    s = suite()
    a, b = bench.sample(s, ROWS), bench.sample(s, ROWS)
    assert a == b
    assert sorted(c for _, c in a) == ["negative"] * 3 + ["positive"] * 3
    assert all(t.strip() for t, _ in a)


def test_a_different_seed_draws_a_different_sample():
    assert bench.sample(suite(seed=0), ROWS) != bench.sample(suite(seed=1), ROWS)


def test_unknown_label_value_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError, match="label 7"):
        bench.sample(suite(), ROWS + [{"sentence": "x", "label": 7}])


def fake_noul(state, questions):
    """Positive text scores high, so AUC is 1.0 by construction."""
    good = state["text"].startswith("good")
    return {"answers": {"q": {"type": "noul", "noul": 0.2 if good else 0.01, "confidence": 0.9}}}


def test_noul_suite_reports_auc_interval_and_a_fitted_cut():
    s = suite(per_class=6)
    result = bench.score(s, bench.sample(s, ROWS), fake_noul)
    assert result["metric"] == "auc"
    assert result["value"] == 1.0
    assert result["ci"][0] <= 1.0 <= result["ci"][1]
    assert result["n"] == 12
    # The lean is visible: every positive is under 0.5, and the fitted cut says where the line is.
    assert result["target_over_half"] == 0.0
    assert 0.01 < result["fit"]["cut"] < 0.2


def fake_choice(state, questions):
    guess = "positive" if state["text"].startswith("good") else "negative"
    other = "negative" if guess == "positive" else "positive"
    return {"answers": {"q": {"type": "choice", "choice": guess, "confidence": 0.5,
                              "probabilities": {guess: 0.8, other: 0.2}}}}


def test_choice_suite_reports_accuracy_and_per_option_recall():
    s = suite(question={"type": "choice", "instructions": "Sentiment of `text`?",
                        "criteria": ["positive", "negative"]}, target=None)
    result = bench.score(s, bench.sample(s, ROWS), fake_choice)
    assert result["metric"] == "accuracy"
    assert result["value"] == 1.0
    assert result["fit"]["per_option"]["negative"]["recall"] == 1.0


def test_a_two_option_choice_also_reports_auc_so_it_compares_with_the_yes_no():
    """Accuracy at the argmax mixes ranking with where the model puts the line; the yes/no suites
    report AUC, so a binary choice carries it too (injection: accuracy 0.83, AUC 0.95)."""
    s = suite(question={"type": "choice", "instructions": "Sentiment of `text`?",
                        "criteria": ["positive", "negative"]}, target=None)
    result = bench.score(s, bench.sample(s, ROWS), fake_choice)
    assert result["auc"] == 1.0


def test_a_many_way_choice_has_no_auc():
    s = suite(question={"type": "choice", "instructions": "?", "criteria": ["a", "b", "c"]},
              classes={"0": "a", "1": "b", "2": "c"}, target=None)
    rows = [{"sentence": f"x{i}", "label": i % 3} for i in range(30)]

    def ask(state, questions):
        return {"answers": {"q": {"type": "choice", "choice": "a", "confidence": 0.1,
                                  "probabilities": {"a": 0.5, "b": 0.3, "c": 0.2}}}}
    assert "auc" not in bench.score(s, bench.sample(s, rows), ask)


def test_choice_criteria_must_cover_every_class():
    with pytest.raises(ValueError, match="neutral"):
        suite(question={"type": "choice", "instructions": "?", "criteria": ["positive", "negative"]},
              classes={"0": "negative", "1": "positive", "2": "neutral"}, target=None)


def test_noul_suite_needs_a_target_class():
    with pytest.raises(ValueError, match="target"):
        suite(target=None)


def card(value, lo, hi, model="m1"):
    return {"model": model, "suites": {"toy": {"metric": "auc", "value": value, "ci": [lo, hi]}}}


def test_verify_passes_when_within_the_previous_interval():
    assert bench.verify(card(0.80, 0.74, 0.86), card(0.82, 0.76, 0.88)) == []


def test_verify_fails_when_below_the_previous_interval():
    problems = bench.verify(card(0.70, 0.64, 0.76), card(0.82, 0.76, 0.88))
    assert len(problems) == 1 and "toy" in problems[0] and "0.70" in problems[0]


def test_verify_fails_when_a_suite_disappears():
    current = {"model": "m1", "suites": {}}
    assert "toy" in bench.verify(current, card(0.82, 0.76, 0.88))[0]


def test_verify_with_no_previous_scorecard_passes():
    assert bench.verify(card(0.5, 0.4, 0.6), None) == []


def test_scorecards_are_ordered_by_version_not_by_name(tmp_path):
    for v in ("0.9.0", "0.10.0", "0.4.0"):
        (tmp_path / f"v{v}.json").write_text(json.dumps(card(0.8, 0.7, 0.9)))
    assert [p.stem for p in bench.scorecards(tmp_path)] == ["v0.4.0", "v0.9.0", "v0.10.0"]


def test_previous_scorecard_is_the_newest_older_version(tmp_path):
    for v in ("0.4.0", "0.5.0", "0.6.0"):
        (tmp_path / f"v{v}.json").write_text(json.dumps(card(0.8, 0.7, 0.9)))
    assert bench.previous(tmp_path, "0.6.0").stem == "v0.5.0"
    assert bench.previous(tmp_path, "0.4.0") is None


def test_shipped_suites_all_parse_and_pin_a_revision():
    suites = bench.load_suites()
    assert len(suites) >= 4
    for s in suites:
        assert len(s.revision) == 40, f"{s.name} must pin a full commit sha"


# ---- the command -------------------------------------------------------------------------------

from verdict import cli, client  # noqa: E402


@pytest.fixture
def toy_bench(monkeypatch, tmp_path):
    suites = tmp_path / "suites.json"
    suites.write_text(json.dumps({"suites": [{
        "name": "toy", "dataset": "someone/toy", "revision": "a" * 40, "split": "test",
        "text": "sentence", "label": "label", "classes": {"0": "negative", "1": "positive"},
        "question": {"type": "noul", "instructions": "Is `text` positive?"},
        "target": "positive", "per_class": 5, "seed": 0}]}))
    monkeypatch.setattr(bench, "SUITES_FILE", suites)
    monkeypatch.setattr(bench, "fetch", lambda s: ROWS)
    monkeypatch.setattr(client, "decide",
                        lambda state, questions, url=None, model=None, **flags: fake_noul(state, questions))
    return tmp_path


def test_bench_writes_a_scorecard(toy_bench, capsys):
    out = toy_bench / "v0.5.0.json"
    assert cli.main(["bench", "--out", str(out), "--pause", "0"]) == 0
    doc = json.loads(out.read_text())
    assert doc["suites"]["toy"]["value"] == 1.0
    assert doc["version"]
    # A quantized run must not pass for a full-precision one (FINDINGS §36).
    assert doc["bits"] in (16, 8)
    assert "toy" in capsys.readouterr().out


def test_bench_verify_needs_no_model_and_fails_on_a_regression(toy_bench, monkeypatch, capsys):
    monkeypatch.setattr(client, "decide", lambda *a, **k: pytest.fail("verify must not ask"))
    (toy_bench / "v0.4.0.json").write_text(json.dumps(card(0.90, 0.85, 0.95)))
    (toy_bench / "v0.5.0.json").write_text(json.dumps(card(0.70, 0.60, 0.80)))
    assert cli.main(["bench", "--verify", str(toy_bench / "v0.5.0.json")]) == 1
    assert "below the previous interval" in capsys.readouterr().err


def test_bench_verify_passes_against_itself_when_first(toy_bench):
    (toy_bench / "v0.5.0.json").write_text(json.dumps(card(0.70, 0.60, 0.80)))
    assert cli.main(["bench", "--verify", str(toy_bench / "v0.5.0.json")]) == 0


def test_bench_unknown_suite_is_an_error(toy_bench):
    assert cli.main(["bench", "--suite", "nope", "--pause", "0"]) == 2


def test_english_only_drops_what_the_detector_flags(monkeypatch):
    from verdict import inputs

    monkeypatch.setattr(inputs, "language",
                        lambda state: {"is_english": not state["text"].startswith("bad 1")})
    s = suite(per_class=20, english_only=True)
    texts = [t for t, _ in bench.sample(s, ROWS)]
    assert "bad 1" not in texts and "bad 2" in texts


# ---- a /v1/systemone contestant (TypeSafe's Jev, or verdict itself) ------------------------------

import httpx  # noqa: E402


def jev_transport(calls, fail_first=0):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) <= fail_first:
            return httpx.Response(529, json={"detail": "overloaded"})
        body = json.loads(request.content)
        good = body["state"]["text"].startswith("good")
        return httpx.Response(200, json={
            "model": "jev-1.13", "usage": {"input_tokens": 9, "output_tokens": 0},
            "answers": {"q": {"type": "noul", "noul": 0.9 if good else 0.1}}})
    return httpx.MockTransport(handler)


def test_systemone_asker_speaks_the_jev_protocol():
    calls = []
    ask = bench.systemone_asker("https://api.example", "jev-latest", "k3y",
                                transport=jev_transport(calls))
    out = ask({"text": "good 1"}, {"q": {"type": "noul", "instructions": "?"}})
    assert out["answers"]["q"]["noul"] == 0.9
    req = calls[0]
    assert req.url.path == "/v1/systemone"
    assert req.headers["authorization"] == "Bearer k3y"
    assert json.loads(req.content)["model"] == "jev-latest"


def test_systemone_asker_retries_overload_and_rate_limits():
    calls = []
    ask = bench.systemone_asker("https://api.example", "jev-latest", "k", backoff=0,
                                transport=jev_transport(calls, fail_first=2))
    assert ask({"text": "good"}, {"q": {"type": "noul"}})["answers"]["q"]["noul"] == 0.9
    assert len(calls) == 3


def test_systemone_asker_gives_up_with_the_status_in_the_error():
    ask = bench.systemone_asker("https://api.example", "jev-latest", "k", backoff=0, attempts=2,
                                transport=jev_transport([], fail_first=9))
    with pytest.raises(ValueError, match="529"):
        ask({"text": "x"}, {"q": {"type": "noul"}})


def flaky_transport(calls, exc=httpx.TimeoutException, fail_first=0):
    """A server that raises a transport-level error (not an HTTP status) for its first calls --
    a cold-starting model, or one that hasn't accepted a connection yet -- then answers."""
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) <= fail_first:
            raise exc("simulated: not answering yet")
        return httpx.Response(200, json={
            "model": "jev-1.13", "usage": {"input_tokens": 9, "output_tokens": 0},
            "answers": {"q": {"type": "noul", "noul": 0.9}}})
    return httpx.MockTransport(handler)


def test_systemone_asker_retries_a_timeout_and_says_so(capsys):
    calls = []
    ask = bench.systemone_asker("https://api.example", "jev-latest", "k", backoff=0,
                                transport=flaky_transport(calls, fail_first=2))
    assert ask({"text": "x"}, {"q": {"type": "noul"}})["answers"]["q"]["noul"] == 0.9
    assert len(calls) == 3
    err = capsys.readouterr().err
    assert "loading" in err and "api.example" in err


def test_systemone_asker_retries_a_connection_error_the_same_way():
    calls = []
    ask = bench.systemone_asker("https://api.example", "jev-latest", "k", backoff=0,
                                transport=flaky_transport(calls, exc=httpx.ConnectError, fail_first=1))
    assert ask({"text": "x"}, {"q": {"type": "noul"}})["answers"]["q"]["noul"] == 0.9
    assert len(calls) == 2


def test_systemone_asker_gives_up_on_persistent_timeouts_with_a_clear_message():
    ask = bench.systemone_asker("https://api.example", "jev-latest", "k", backoff=0, attempts=2,
                                transport=flaky_transport([], fail_first=99))
    with pytest.raises(ValueError, match="loading"):
        ask({"text": "x"}, {"q": {"type": "noul"}})


def test_bench_scores_a_systemone_endpoint(toy_bench, monkeypatch, capsys):
    monkeypatch.setattr(client, "decide", lambda *a, **k: pytest.fail("must not ask verdict"))
    monkeypatch.setattr(bench, "_default_transport", lambda: jev_transport([]))
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    out = toy_bench / "jev.json"
    assert cli.main(["bench", "--systemone", "https://api.example", "--pause", "0",
                     "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["suites"]["toy"]["value"] == 1.0
    assert doc["model"] == "systemone:jev-latest@api.example"


def test_bench_against_a_remote_systemone_needs_a_key(toy_bench, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert cli.main(["bench", "--systemone", "https://api.example", "--pause", "0"]) == 2


def dead_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated: server never answered")
    return httpx.MockTransport(handler)


def test_bench_against_a_persistently_unreachable_systemone_server_fails_cleanly(
        toy_bench, monkeypatch, capsys):
    """Same failure `verdict examples` guards against: a dead --systemone endpoint must not
    surface as an unhandled traceback from inside `bench.score`."""
    monkeypatch.setattr(bench, "_default_transport", dead_transport)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    code = cli.main(["bench", "--systemone", "https://api.example", "--pause", "0"])
    assert code == 2
    err = capsys.readouterr().err
    assert "loading" in err
    assert "Traceback" not in err


# ---- suites in other languages -------------------------------------------------------------------

MIXED = ROWS + [{"sentence": f"meh {i}", "label": 2} for i in range(5)]


def test_ignored_labels_are_skipped_not_errors():
    """Russian reviews carry a neutral label and Bulgarian ones a mixed one: a positive-versus-
    negative suite names them to skip, so an unexpected label still fails loudly."""
    s = suite(ignore=["2"])
    assert sorted({c for _, c in bench.sample(s, MIXED)}) == ["negative", "positive"]
    with pytest.raises(ValueError, match="label 2"):
        bench.sample(suite(), MIXED)


@pytest.mark.parametrize("lang", ["multi", "en"])
def test_a_suite_can_name_the_checkpoint_that_reads_it(lang):
    assert suite(lang=lang).lang == lang


def test_an_unknown_suite_lang_is_refused():
    with pytest.raises(ValueError, match="lang"):
        suite(lang="bulgarian")


def test_bench_asks_a_multi_suite_with_the_multilingual_checkpoint(toy_bench, monkeypatch):
    data = json.loads(bench.SUITES_FILE.read_text())
    data["suites"][0]["lang"] = "multi"
    bench.SUITES_FILE.write_text(json.dumps(data))
    models = []

    def fake(state, questions, url=None, model=None, **flags):
        models.append(model)
        return fake_noul(state, questions)
    monkeypatch.setattr(client, "decide", fake)
    assert cli.main(["bench", "--pause", "0"]) == 0
    assert set(models) == {"multilingual"}


def test_bench_does_not_warn_about_long_items(toy_bench, monkeypatch, capsys):
    """The bench clips its items on purpose, as users' states are; the long-state warning is for
    a user's own input, and on a bench run it was only noise."""
    long_rows = [{**r, "sentence": r["sentence"] + " filler" * 200} for r in ROWS]
    monkeypatch.setattr(bench, "fetch", lambda s: long_rows)
    assert cli.main(["bench", "--out", str(toy_bench / "v0.5.0.json"), "--pause", "0"]) == 0
    assert "first 128 tokens" not in capsys.readouterr().err
