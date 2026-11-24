"""The bench gate's like-with-like rule and the router cache's key and writes (audit 2)."""
import json

import pytest

from verdict.evaluation import bench, router_eval

from efakes import engine, write_cards  # noqa: F401  (engine is a fixture)


def card(value=0.8, model="m1", **extra):
    suite = {"metric": "auc", "value": value, "ci": [value - 0.05, value + 0.05], **extra}
    return {"model": model, "bits": 16, "budget": 128, "suites": {"toy": suite}}


def test_a_current_card_without_a_measurement_field_is_an_error_not_a_baseline(tmp_path):
    write_cards(tmp_path, {"0.5.0": card()})
    for lost in ("model", "bits", "budget"):
        like = {k: v for k, v in card().items() if k != lost}
        with pytest.raises(ValueError, match=lost):
            bench.previous(tmp_path, "0.6.0", like=like)


def test_an_older_card_without_a_measurement_field_is_an_error(tmp_path):
    old = card()
    del old["budget"]
    write_cards(tmp_path, {"0.5.0": old})
    with pytest.raises(ValueError, match=r"v0\.5\.0\.json has no budget"):
        bench.previous(tmp_path, "0.6.0", like=card())


def test_a_changed_model_bits_or_budget_is_still_a_baseline(tmp_path):
    write_cards(tmp_path, {"0.5.0": card()})
    assert bench.previous(tmp_path, "0.6.0", like={**card(), "bits": 8}) is None


def test_verify_says_so_when_the_previous_card_has_no_sample_fingerprint(capsys):
    cur, prev = card(sample_sha256="a" * 8), card()
    assert bench.verify(cur, prev) == []  # still compared, still passes
    err = capsys.readouterr().err
    assert "toy" in err and "previous" in err and "sample_sha256" in err


def test_verify_is_quiet_when_both_cards_carry_the_fingerprint(capsys):
    assert bench.verify(card(sample_sha256="a"), card(sample_sha256="a")) == []
    assert capsys.readouterr().err == ""


# --- router cache -------------------------------------------------------------------------------

ROWS = [{"id": "a", "prompt": "one"}, {"id": "b", "prompt": "two two"}]


def test_a_concurrent_runs_bucket_is_not_dropped(tmp_path, engine):
    cache = tmp_path / "probs.json"

    def other_run_writes(n):
        if n == 2:  # between this run's two prompts another process adds its own model's bucket
            store = json.loads(cache.read_text())
            store["other-model"] = {"z": {"difficulty": 1.0, "is_sensitive": 0.5, "fp": "f"}}
            cache.write_text(json.dumps(store))

    engine.on_predict = other_run_writes
    router_eval.measure(ROWS, "m", cache)
    store = json.loads(cache.read_text())
    assert "other-model" in store and set(store["m"]) == {"a", "b"}


def test_retrained_local_weights_miss_the_cache(tmp_path, engine):
    weights = tmp_path / "model"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"x" * 10)
    cache = tmp_path / "probs.json"
    router_eval.measure(ROWS, str(weights), cache)
    assert engine.n == 2
    router_eval.measure(ROWS, str(weights), cache)
    assert engine.n == 2  # unchanged weights: served from the cache
    (weights / "model.safetensors").write_bytes(b"y" * 20)
    router_eval.measure(ROWS, str(weights), cache)
    assert engine.n == 4  # new weights under the same path: measured again


def test_a_hub_id_keeps_its_plain_key(tmp_path, engine):
    router_eval.measure(ROWS, "owner/some-model", tmp_path / "probs.json")
    assert list(json.loads((tmp_path / "probs.json").read_text())) == ["owner/some-model"]
