"""One cache, one paying run at a time (audit 4). Invented data only."""
import fcntl
import os

import pytest

from verdict.label.bank import BANK
from verdict.label.prompt_eval import SpendLedger, score_preamble
from verdict.label.teacher import Teacher
from test_prompt_eval import T, refs, states  # noqa: E402  (same directory, invented data)


def test_a_second_run_on_a_cache_in_use_is_refused_before_any_paid_call(tmp_path):
    cache = tmp_path / "c.jsonl"
    fd = os.open(tmp_path / "c.jsonl.lock", os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # what the first run holds
    try:
        t = T()
        with pytest.raises(SystemExit, match="in use"):
            score_preamble(states(), refs(), Teacher("gemini-2.5-flash-lite", t, preamble="P"), BANK,
                           cache_path=cache, ledger=SpendLedger(tmp_path / "l.json", cap=5))
        assert t.calls == 0
    finally:
        os.close(fd)


def test_the_lock_is_released_when_the_run_ends(tmp_path):
    kw = dict(cache_path=tmp_path / "c.jsonl", ledger=SpendLedger(tmp_path / "l.json", cap=5))
    for _ in range(2):
        score_preamble(states(), refs(), Teacher("gemini-2.5-flash-lite", T(), preamble="P"), BANK, **kw)
