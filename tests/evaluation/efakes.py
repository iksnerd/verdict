"""Shared helpers for tests/evaluation. Invented data only."""
import json

import pytest


def write_cards(directory, cards):
    """Write one scorecard per entry of `{version: card}` as `v<version>.json` in `directory`."""
    for version, card in cards.items():
        (directory / f"v{version}.json").write_text(json.dumps(card))


class CountingEngine:
    """Stands in for a loaded checkpoint: answers from the prompt text, counts what it was asked.

    `on_predict(n)` runs after the n-th question is recorded; `die_after=k` raises KeyboardInterrupt
    on the question after the k-th.
    """

    def __init__(self, on_predict=None, die_after=None):
        self.asked, self.budgets = [], []
        self.on_predict, self.die_after = on_predict, die_after

    @property
    def n(self):
        return len(self.asked)

    def clip(self, prompt, budget=None):
        self.budgets.append(budget)
        return prompt

    def predict(self, state, questions):
        if self.die_after is not None and len(self.asked) >= self.die_after:
            raise KeyboardInterrupt
        self.asked.append(state)
        if self.on_predict:
            self.on_predict(len(self.asked))
        return {"difficulty": {"type": "score", "score": float(len(state))},
                "is_sensitive": {"type": "noul", "noul": 0.25}}


@pytest.fixture
def engine(monkeypatch):
    """`measure` loads through `backends.engine.load`; this hands it a CountingEngine and counts the loads."""
    from verdict.backends import engine as engine_module

    eng = CountingEngine()
    eng.loads = 0

    def fake_load(model, *a, **kw):
        eng.loads += 1
        return eng

    monkeypatch.setattr(engine_module, "load", fake_load)
    return eng
