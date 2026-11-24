"""Audit 4: served_checkpoint resolves names as MlxBackend._resolve does."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verdict import library
from verdict.backends import mlx
from verdict.backends.base import UnknownModel


@pytest.mark.parametrize("name", [None, "english", "multilingual", "jev-latest", "ft", "typo"])
def test_served_checkpoint_agrees_with_the_backend(name, monkeypatch):
    monkeypatch.setattr(mlx, "load", lambda cid, **kw: SimpleNamespace(loaded=cid))
    b = mlx.MlxBackend("served/model", agent=object(), multilingual_id="org/multi",
                       extra_checkpoints={"jev-latest": "org/extra-jev", "ft": "org/ft",
                                          "english": "org/extra-english"})
    try:
        engine = b._resolve(name, cache=False)
        expected = b.model_id if engine is b.engine else engine.loaded
    except UnknownModel:
        expected = name
    assert library.served_checkpoint(b, name) == expected


def test_an_unconfigured_jev_alias_is_the_served_checkpoint():
    b = mlx.MlxBackend("served/model", agent=object())
    assert library.served_checkpoint(b, "jev-latest") == "served/model"
