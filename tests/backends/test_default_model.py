"""A fresh install answers with base Laya, and nothing offers to download the private fine-tune.

The fine-tune is private (its labels are Gemini's), so for anyone but its author `verdict weights`
could only fail, and `init` and `update` pushed them toward it. With every new yes/no asked as a
no/yes choice, base Laya is within noise of the fine-tune (FINDINGS §40), so it is the default.
Someone with the fine-tune points `model.path` at it. No model loads here."""
from __future__ import annotations

import importlib.util

import pytest

from verdict import cli, config


def test_the_default_model_is_base_laya():
    assert config.DEFAULTS["model"]["path"] == "aac6fef/laya-mlx"


def test_there_is_no_weights_command(capsys):
    with pytest.raises(SystemExit) as exit_:
        cli.main(["weights"])
    assert exit_.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_the_download_module_is_gone():
    assert importlib.util.find_spec("verdict.weights") is None


def test_a_missing_local_model_falls_back_without_pointing_at_a_download(tmp_path):
    model, warning = config.resolve_model(str(tmp_path / "gone"))
    assert model == config.FALLBACK_MODEL
    assert warning and "verdict weights" not in warning


def test_the_default_needs_no_fallback():
    model, warning = config.resolve_model(config.DEFAULTS["model"]["path"])
    assert model == "aac6fef/laya-mlx" and warning is None
