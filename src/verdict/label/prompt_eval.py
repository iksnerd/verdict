"""Score a teacher-prompt preamble: label states with a cheap teacher under that preamble and
measure agreement with the fixed reference labels. This is what an optimizer's judge calls.

Two guards, because an optimizer will happily call a paid API in a loop: identical
(preamble, model, state) triples are cached on disk and never paid for twice, and every call
adds to a persistent spend ledger that stops the run at a hard cap."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

from .teacher import LabelError, Teacher, agreement, cost_of, price_of
from ..pipeline.store import JsonlStore, write_json_atomic


class SpendLedger:
    """The file is the truth: every read re-reads it and every add reads, adds and writes under a
    lock, so two runs on one ledger add up and the cap sees what the other run spent."""

    def __init__(self, path: Path, cap: float):
        self.path, self.cap = Path(path), cap

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path.with_name(self.path.name + ".lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _read(self) -> float:
        return json.loads(self.path.read_text())["usd"] if self.path.exists() else 0.0

    def total(self) -> float:
        return round(self._read(), 6)

    def add(self, usd: float) -> None:
        with self._locked():
            write_json_atomic(self.path, {"usd": round(self._read() + usd, 6), "cap": self.cap})

    def check(self) -> None:
        total = self._read()
        if total >= self.cap:
            raise SystemExit(f"spend cap ${self.cap} reached (${total:.4f}); stopping before another paid call")


def _key(preamble: str, model: str, state_id: str, bank: dict | None = None) -> str:
    """What a cached label was made for: the model, the preamble, the state and the question bank
    (a changed bank is a different question). The reply format and prompt template are fixed text
    in `build_prompt`; the sampling temperature belongs to the transport and is not recorded."""
    bank_digest = hashlib.sha256(json.dumps(bank, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return hashlib.sha256(f"{model}\x00{preamble}\x00{state_id}\x00{bank_digest}".encode()).hexdigest()


def score_preamble(states: list[dict], reference: dict, teacher: Teacher, bank: dict, cache_path: Path, ledger: SpendLedger) -> dict:
    missing = [s["id"] for s in states if s["id"] not in reference]
    if missing:  # before any paid call, not after the first label was bought
        raise KeyError(f"no reference label for {len(missing)} state(s), first {missing[0]!r}: "
                       "`reference` must hold every id in `states`")
    price_of(teacher.model, teacher.prices)  # an unpriced model would cost $0 and the cap could never trip
    cache = JsonlStore(cache_path, key=lambda r: r["k"])
    by_key = {r["k"]: r["labels"] for r in cache.rows()}
    per_q: dict[str, list[float]] = {}
    spent = 0.0
    cached = failures = 0
    for s in states:
        k = _key(teacher.preamble or "", teacher.model, s["id"], bank)
        if k in by_key:
            labels = by_key[k]
            cached += 1
        else:
            ledger.check()
            try:
                labels = teacher.label(s["state"], bank)
            except LabelError:
                failures += 1
                cost = cost_of(teacher.model, teacher.last_in, teacher.last_out, teacher.prices)  # the failed tries were billed too
                spent += cost
                ledger.add(cost)
                continue
            cost = cost_of(teacher.model, teacher.last_in, teacher.last_out, teacher.prices)
            spent += cost
            ledger.add(cost)
            cache.append({"k": k, "labels": labels})
        for qid, v in agreement(labels, reference[s["id"]]).items():
            per_q.setdefault(qid, []).append(v)
    allv = [v for xs in per_q.values() for v in xs]
    return {
        "score": round(100 * sum(allv) / len(allv), 2) if allv else 0.0,
        "n": len(states) - failures,
        "cached": cached,
        "failures": failures,
        "per_question": {q: round(100 * sum(v) / len(v), 2) for q, v in per_q.items()},
        "spend_usd_this_call": round(spent, 6),
        "spend_usd_total": ledger.total(),
    }
