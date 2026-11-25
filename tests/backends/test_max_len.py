"""`[model].max_len`: opt-in, how many tokens the checkpoint reads. Default 0 is the checkpoint's own.

laya-mlx reads the limit from `agent.cfg["max_len"]` and applies its own check
(`head_max_len < max_len <= max_position_embeddings`) only at load, never to a changed cfg, and
clips silently above 8192. So verdict validates, and these tests use agent doubles that record
what they were given."""
from __future__ import annotations

import argparse
import types

import pytest

from verdict import cli, config
from verdict.backends import engine
from verdict.backends.mlx import MlxBackend
from verdict.cli import inference
from fakes import Agent, fake_transformers, install_laya


def isolated(monkeypatch, tmp_path, toml=None):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VERDICT_MAX_LEN", raising=False)
    if toml is None:
        monkeypatch.setattr(config, "CONFIG_PATHS", ())
    else:
        f = tmp_path / "verdict.toml"
        f.write_text(toml)
        monkeypatch.setattr(config, "CONFIG_PATHS", (f,))


# ---- the setting ---------------------------------------------------------------------------------

def test_default_is_zero_and_explain_says_default(monkeypatch, tmp_path):
    isolated(monkeypatch, tmp_path)
    assert config.load().max_len == 0
    row = next(r for r in config.explain() if r["key"] == "model.max_len")
    assert row["value"] == 0 and row["source"] == "default"


def test_file_then_env_then_flag(monkeypatch, tmp_path):
    isolated(monkeypatch, tmp_path, "[model]\nmax_len = 1024\n")
    assert config.load().max_len == 1024
    rows = {r["key"]: r for r in config.explain()}
    assert rows["model.max_len"]["source"].endswith("verdict.toml")
    monkeypatch.setenv("VERDICT_MAX_LEN", "2048")
    assert config.load().max_len == 2048
    assert {r["key"]: r for r in config.explain()}["model.max_len"]["source"] == "$VERDICT_MAX_LEN"
    settings = config.load()
    assert inference._max_len(argparse.Namespace(max_len=4096), settings) == 4096
    assert inference._max_len(argparse.Namespace(max_len=None), settings) == 2048
    assert inference._max_len(argparse.Namespace(), settings) == 2048


@pytest.mark.parametrize("bad", ["-1", "abc", "2.5", "8193", "5", "true"])
def test_a_bad_environment_value_is_a_config_error(monkeypatch, tmp_path, bad):
    isolated(monkeypatch, tmp_path)
    monkeypatch.setenv("VERDICT_MAX_LEN", bad)
    with pytest.raises(config.ConfigError, match="max_len"):
        config.load()


@pytest.mark.parametrize("bad", ["true", "2.5", '"abc"', "-1", "8193", "[512]"])
def test_a_bad_file_value_is_a_config_error_naming_the_key(monkeypatch, tmp_path, bad):
    isolated(monkeypatch, tmp_path, f"[model]\nmax_len = {bad}\n")
    with pytest.raises(config.ConfigError, match=r"\[model\]\.max_len"):
        config.load()


def test_zero_is_the_default_and_8192_is_the_ceiling(monkeypatch, tmp_path):
    isolated(monkeypatch, tmp_path, "[model]\nmax_len = 8192\n")
    assert config.load().max_len == 8192
    isolated(monkeypatch, tmp_path, "[model]\nmax_len = 0\n")
    assert config.load().max_len == 0


def test_a_bad_flag_is_a_one_line_error_exit_2(monkeypatch, tmp_path, capsys):
    isolated(monkeypatch, tmp_path)
    assert cli.main(["serve", "--max-len", "9000"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("verdict: ") and "--max-len" in err and "8192" in err
    assert "Traceback" not in err and err.count("\n") == 1


def test_init_writes_no_key_when_unset_and_round_trips_when_set(monkeypatch, tmp_path):
    isolated(monkeypatch, tmp_path)
    text = config.to_toml(config.load())
    assert not any(line.startswith("max_len") for line in text.splitlines())
    s = config.load()
    s.max_len = 2048
    again = tmp_path / "again.toml"
    again.write_text(config.to_toml(s))
    monkeypatch.setattr(config, "CONFIG_PATHS", (again,))
    assert config.load().max_len == 2048


# ---- the engine ----------------------------------------------------------------------------------

def laya_agent(native=512, head=192):
    agent = types.SimpleNamespace(model=object(), cfg={"max_len": native, "head_max_len": head})
    agent._inference = agent.model
    agent.predict = lambda state, questions: {"answers": {}}
    return agent


def fake_load(monkeypatch, agent, loaded=None):
    install_laya(monkeypatch, lambda model_id, **kw: (loaded if loaded is not None else []).append(kw) or agent)


def test_max_len_is_applied_to_the_agent_after_load_and_not_passed_to_laya(monkeypatch):
    agent, loaded = laya_agent(), []
    fake_load(monkeypatch, agent, loaded)
    e = engine.load("m", max_len=2048)
    assert e.agent is agent
    assert agent.cfg["max_len"] == 2048 and e.native_max_len == 512
    assert loaded == [{}], "max_len is ours, never passed to laya_mlx.load"


def test_it_applies_to_the_multilingual_checkpoint_too(monkeypatch):
    agent = laya_agent(native=1024, head=256)
    fake_load(monkeypatch, agent)
    e = engine.load("multi", max_len=4096)
    e.agent
    assert agent.cfg["max_len"] == 4096 and e.native_max_len == 1024


def test_an_injected_agent_gets_it_too():
    agent = laya_agent()
    e = engine.Engine("m", agent=agent, max_len=1024)
    assert agent.cfg["max_len"] == 1024 and e.max_len == 1024


@pytest.mark.parametrize("n", [192, 100])
def test_a_limit_not_above_head_max_len_names_both_numbers(monkeypatch, n):
    fake_load(monkeypatch, laya_agent())
    e = engine.load("m", max_len=n)
    with pytest.raises(ValueError, match=rf"max_len {n}.*head_max_len 192"):
        e.agent
    assert "\n" not in str(pytest.raises(ValueError, lambda: e.agent).value)


def test_a_failed_limit_never_leaves_the_agent_cached(monkeypatch):
    agent = laya_agent()
    fake_load(monkeypatch, agent)
    e = engine.load("m", max_len=100)
    for _ in range(2):
        with pytest.raises(ValueError):
            e.agent
    assert e._agent is None and agent.cfg["max_len"] == 512


def test_a_lazy_multilingual_engine_reports_the_checkpoints_1024(monkeypatch):
    """The bug: a lazily loaded Engine had no agent at construction, so it said 512 for the
    multilingual checkpoint, which reads 1024, and over-reported `truncated`."""
    fake_load(monkeypatch, laya_agent(native=1024, head=256))
    fake_transformers(monkeypatch, [])
    e = engine.Engine("multi")
    text = "x " * 600
    assert e.exceeds_context(text) is False
    assert e.max_len == 1024


def test_a_lazy_english_engine_reports_512(monkeypatch):
    fake_load(monkeypatch, laya_agent())
    fake_transformers(monkeypatch, [])
    e = engine.Engine("main")
    assert e.exceeds_context("x " * 600) is True
    assert e.max_len == 512


def test_the_limit_is_fixed_at_first_use(monkeypatch):
    agent = laya_agent(native=1024, head=256)
    fake_load(monkeypatch, agent)
    e = engine.Engine("multi")
    assert e.max_len == 1024
    agent.cfg["max_len"] = 2048
    assert e.max_len == 1024


def test_a_configured_limit_needs_no_load_to_be_known(monkeypatch):
    fake_load(monkeypatch, laya_agent())
    monkeypatch.setitem(__import__("sys").modules, "laya_mlx", None)   # a load would fail
    assert engine.Engine("m", max_len=2048).max_len == 2048


def test_capabilities_report_the_resolved_limit(monkeypatch):
    agent = laya_agent(native=1024, head=256)
    backend = MlxBackend("multi", agent=agent)
    assert backend.capabilities().max_state_and_question_tokens == 1024
    backend = MlxBackend("multi", agent=laya_agent(), max_len=2048)
    assert backend.capabilities().max_state_and_question_tokens == 2048


def test_the_backend_hands_the_setting_to_every_checkpoint_it_loads(monkeypatch):
    loaded = []
    monkeypatch.setattr("verdict.backends.mlx.load",
                        lambda model_id, **kw: loaded.append((model_id, kw)) or engine.Engine(model_id, agent=Agent()))
    backend = MlxBackend("main", agent=laya_agent(), multilingual_id="multi", max_len=2048)
    backend.engine_for("multilingual")
    assert loaded == [("multi", {"max_len": 2048})]


# ---- the budget warning and the in-process path ----------------------------------------------------

def test_note_when_the_budget_makes_it_moot():
    note = config.max_len_note(2048, 128)
    assert note == "max_len 2048 has no effect while prompt_token_budget is 128"
    assert config.max_len_note(2048, 0) is None          # read whole
    assert config.max_len_note(2048, 4096) is None       # budget leaves room
    assert config.max_len_note(0, 128) is None           # unset
    assert config.max_len_note(512, 128) is None         # not above the default
    assert config.max_len_note(768, 128, native=1024) is None


def test_a_local_load_passes_max_len_and_warns_once(monkeypatch, tmp_path, capsys):
    from fakes import FakeEngine

    isolated(monkeypatch, tmp_path)
    seen = []
    monkeypatch.setattr(inference, "_ENGINES", {})
    monkeypatch.setattr(inference, "load", lambda path, **kw: seen.append(kw) or FakeEngine())
    monkeypatch.setattr(inference.client, "decide",
                        lambda *a, **k: (_ for _ in ()).throw(inference.client.NoServer("none")))
    q = {"q": {"type": "noul", "instructions": "?"}}
    for _ in range(2):
        inference._answer("s", q, "http://127.0.0.1:1", "m", budget=128, max_len=2048)
    assert seen == [{"max_len": 2048}]
    err = capsys.readouterr().err
    assert err.count("max_len 2048 has no effect while prompt_token_budget is 128") == 1
    # unset: no kwarg, so every existing load stays as it was
    seen.clear()
    monkeypatch.setattr(inference, "_ENGINES", {})
    inference._answer("s", q, "http://127.0.0.1:1", "m", budget=128)
    assert seen == [{}] or seen == []


def test_a_fit_made_at_another_length_warns_and_an_old_fit_is_the_default():
    from verdict.cli import decisions

    warned = []
    ask = types.SimpleNamespace(settings=types.SimpleNamespace(bits=16, prompt_token_budget=128, max_len=2048),
                                warn=warned.append)
    decisions._check_fit_identity(ask, {"model": "m", "bits": 16, "budget": 128}, {"model": "m"}, "fit.json")
    assert len(warned) == 1 and "max_len default" in warned[0] and "max_len 2048" in warned[0]
    warned.clear()
    decisions._check_fit_identity(ask, {"model": "m", "max_len": 2048}, {"model": "m"}, "fit.json")
    assert warned == []
    ask.settings.max_len = 0
    decisions._check_fit_identity(ask, {"model": "m", "bits": 16}, {"model": "m"}, "fit.json")
    assert warned == []
