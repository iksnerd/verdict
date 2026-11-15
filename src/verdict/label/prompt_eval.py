"""Score a teacher-prompt preamble: label states with a cheap teacher under that preamble and
measure agreement with the fixed reference labels. This is what an optimizer's judge calls.

Two guards, because an optimizer will happily call a paid API in a loop: identical
(preamble, model, state) triples are cached on disk and never paid for twice, and every call
adds to a persistent spend ledger that stops the run at a hard cap."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .teacher import LabelError, Teacher, agreement, cost_of
from ..pipeline.store import JsonlStore, write_json_atomic


class SpendLedger:
    def __init__(self, path: Path, cap: float):
        self.path, self.cap = Path(path), cap
        self._total = json.loads(self.path.read_text())["usd"] if self.path.exists() else 0.0

    def total(self) -> float:
        return round(self._total, 6)

    def add(self, usd: float) -> None:
        self._total += usd
        write_json_atomic(self.path, {"usd": round(self._total, 6), "cap": self.cap})

    def check(self) -> None:
        if self._total >= self.cap:
            raise SystemExit(f"spend cap ${self.cap} reached (${self._total:.4f}); stopping before another paid call")


def _key(preamble: str, model: str, state_id: str) -> str:
    return hashlib.sha256(f"{model}\x00{preamble}\x00{state_id}".encode()).hexdigest()


def score_preamble(states: list[dict], reference: dict, teacher: Teacher, bank: dict, cache_path: Path, ledger: SpendLedger) -> dict:
    cache = JsonlStore(cache_path, key=lambda r: r["k"])
    by_key = {r["k"]: r["labels"] for r in cache.rows()}
    per_q: dict[str, list[float]] = {}
    spent = 0.0
    cached = failures = 0
    for s in states:
        k = _key(teacher.preamble or "", teacher.model, s["id"])
        if k in by_key:
            labels = by_key[k]
            cached += 1
        else:
            ledger.check()
            try:
                labels = teacher.label(s["state"], bank)
            except LabelError:
                failures += 1
                cost = cost_of(teacher.model, teacher.last_in, teacher.last_out)  # the failed tries were billed too
                spent += cost
                ledger.add(cost)
                continue
            cost = cost_of(teacher.model, teacher.last_in, teacher.last_out)
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
