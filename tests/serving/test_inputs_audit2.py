"""Audit 2: a laya whose lang.py cannot be executed is the same as no laya."""
from __future__ import annotations

import pytest

from verdict import inputs


@pytest.mark.parametrize("exc", [ImportError("x"), SyntaxError("x"), ModuleNotFoundError("x")])
def test_an_unreadable_laya_lang_file_returns_none(monkeypatch, exc):
    monkeypatch.setattr(inputs, "_lang", None)

    def boom(name):
        raise exc

    monkeypatch.setattr(inputs, "_laya_file", boom)
    assert inputs.language("hello") is None
