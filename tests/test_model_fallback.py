"""Without the fine-tuned weights, fall back to base laya with a warning instead of failing.

The fine-tuned checkpoint is not in git or the release, so a machine without it could not answer
at all: `verdict serve` and every in-process load failed on the missing path. Base laya downloads
itself and ties on choices, but its yes/no answers are weaker (FINDINGS §35), so the fallback
says so. Only the configured model falls back; a `--model` the caller typed still fails."""
from __future__ import annotations

import argparse

import pytest

from verdict import cli, config
from verdict.cli import inference, support
from verdict.backends.engine import DEFAULT_MODEL


def test_an_existing_checkpoint_is_used_as_is(tmp_path):
    (tmp_path / "ckpt").mkdir()
    assert config.resolve_model(str(tmp_path / "ckpt")) == (str(tmp_path / "ckpt"), None)


@pytest.mark.parametrize("missing", ["/nowhere/verdict-v1-mlx", "./models/gone", "models/verdict-v1-mlx"])
def test_a_missing_local_checkpoint_falls_back_to_base_laya(missing, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # so the old relative default does not exist either
    model, warning = config.resolve_model(missing)
    assert model == DEFAULT_MODEL
    assert missing in warning and "§40" in warning


def test_a_configured_hub_id_is_left_alone(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert config.resolve_model("aac6fef/laya-typed-decisions-mlx") == (
        "aac6fef/laya-typed-decisions-mlx", None)


def test_an_explicit_model_is_never_replaced(tmp_path):
    missing = str(tmp_path / "typo")
    assert config.resolve_model(missing, explicit=True) == (missing, None)


_real_load = config.load


def settings_with(path):
    s = _real_load()
    s.model_path = path
    return s


def test_ask_falls_back_and_warns_once(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(config, "load", lambda: settings_with(str(tmp_path / "gone")))
    ns = argparse.Namespace(url=None, model=None, lang="en")
    asker = inference._Asker(ns)
    assert asker.main_path == DEFAULT_MODEL
    assert capsys.readouterr().err.count("base laya") == 1


def test_serve_falls_back_instead_of_failing(monkeypatch, capsys, tmp_path):
    missing = str(tmp_path / "gone")
    monkeypatch.setattr(support, "_settings", lambda: settings_with(missing))
    loaded = {}

    class FakeBackend:
        def __init__(self, model_id, **kw):
            loaded["model"] = model_id
            self.name = f"fake:{model_id}"
            self.engine = argparse.Namespace(agent=None, tokenizer=None)

    import uvicorn

    import verdict.backends.mlx as backend_mlx
    monkeypatch.setattr(backend_mlx, "MlxBackend", FakeBackend)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    assert cli.main(["serve", "--port", "0"]) == 0
    assert loaded["model"] == DEFAULT_MODEL
    assert "base laya" in capsys.readouterr().err
