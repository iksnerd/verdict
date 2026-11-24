"""Checks to run before trusting a split. Both come from a real failure in an earlier corpus
pipeline: a split whose aggregate counts looked fine while one whole source had zero presence in
three of four splits (only the cross-tab showed it), and 46 training documents that had exact
duplicates in a held-out split (only a cross-split check finds those)."""
from __future__ import annotations

import hashlib
import re

from ..gen.generate import is_near_duplicate


def topic_of(s: dict) -> str:
    return s.get("topic") or s["state"].get("room_topic", "")


def author_of(s: dict) -> str:
    return s.get("author") or s["state"].get("author", "")


SPLITS = ("train", "holdout", "test")


def split_of(split: dict) -> dict[str, str]:
    """{state id: split name} for every id a split.json assigns."""
    return {i: name for name in SPLITS for i in split.get(name, [])}


def text_of(s: dict) -> str:
    st = s["state"]
    return st.get("message") or st.get("text") or st.get("issue", "")


def split_stats(states: list[dict], split: dict) -> dict:
    where = split_of(split)
    stats: dict = {"intended_type": {}, "author": {}, "room_topic": {}}
    for s in states:
        name = where.get(s["id"], "unassigned")
        for dim, val in (("intended_type", s["intended_type"]), ("author", author_of(s)), ("room_topic", topic_of(s))):
            stats[dim].setdefault(val, {}).setdefault(name, 0)
            stats[dim][val][name] += 1
    stats["missing_in_test"] = sorted(t for t, c in stats["intended_type"].items() if not c.get("test"))
    stats["missing_in_holdout"] = sorted(t for t, c in stats["intended_type"].items() if not c.get("holdout"))
    return stats


def _norm(text: str) -> str:
    return hashlib.sha256(" ".join(re.findall(r"[a-z0-9]+", text.lower())).encode()).hexdigest()


def held_out_leak(states: list[dict], split: dict, threshold: float = 0.8) -> list[str]:
    """Training ids with an exact or near-duplicate message in holdout or test."""
    by_id = {s["id"]: text_of(s) for s in states}
    held = [by_id[i] for name in ("holdout", "test") for i in split.get(name, []) if i in by_id]
    held_hashes = {_norm(m) for m in held}
    return sorted(i for i in split.get("train", []) if i in by_id and (_norm(by_id[i]) in held_hashes or is_near_duplicate(by_id[i], held, threshold)))
