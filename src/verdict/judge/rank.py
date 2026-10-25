"""Named answers as a vector, ranked by a weighted sum over rescaled dimensions.

Each question in a `decide --jsonl` run is a named dimension: a yes/no is its P(yes), a choice
gives one dimension per option (`intent.bug`), a score its position on the scale from 0 to 1. A
query is a weight per dimension, and every result can say what each dimension contributed.

Two choices here were measured, not assumed (FINDINGS §39):

- **A weighted sum, not cosine.** Cosine ignores magnitude, so an item scoring about 0.13 on every
  dimension points the same way as one scoring 0.9, and ranked among the top five for a query it
  matched on nothing.
- **Each dimension rescaled to its percentile over the collection.** The answers are uncalibrated
  and each question has its own range: one spread 0.12 to 0.65 while another reached 0.92, and
  until they were rescaled the wide one decided every ranking.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def dimensions(answers: dict[str, Any]) -> dict[str, float]:
    """One named number per yes/no, per choice option and per score."""
    out = {}
    for qid, a in answers.items():
        kind = a.get("type")
        if kind == "noul":
            out[qid] = a["noul"]
        elif kind == "choice":
            for option, p in a["probabilities"].items():
                out[f"{qid}.{option}"] = p
        elif kind == "score":
            top = max(len(a.get("legend") or a.get("probabilities") or {}) - 1, 1)
            out[qid] = a["score"] / top
    return out


def percentiles(values: list[float]) -> list[float]:
    """Each value's rank over the list, 0 for the lowest and 1 for the highest; ties share the
    average of their ranks."""
    if len(values) < 2:
        return [0.5] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return [r / (len(values) - 1) for r in ranks]


def parse_weights(specs: list[str]) -> dict[str, float]:
    weights = {}
    for spec in specs:
        name, sep, value = spec.partition("=")
        try:
            if not sep or not name:
                raise ValueError
            weights[name] = float(value)
        except ValueError:
            raise ValueError(f"{spec!r}: give each weight as NAME=WEIGHT, e.g. about_money=1") from None
    return weights


def load(path: str) -> list[dict[str, Any]]:
    """The lines of a `decide --jsonl` run."""
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise ValueError(f"{path}: {exc.strerror}") from None
    rows = []
    for n, l in enumerate(text.splitlines(), 1):
        if not l.strip():
            continue
        try:
            row = json.loads(l)
        except ValueError:
            row = None
        if not isinstance(row, dict):
            raise ValueError(f"{path} line {n} is not a JSON object; rank reads the output of "
                             "`verdict decide --jsonl`")
        rows.append(row)
    if not any(r.get("answers") for r in rows):
        raise ValueError(f"{path} has no answers; rank reads the output of `verdict decide --jsonl`")
    return rows


def rank(rows: list[dict[str, Any]], weights: dict[str, float]) -> list[dict[str, Any]]:
    """Every row with its `score` and per-dimension `contributions`, best first."""
    vectors = [dimensions(r.get("answers", {})) for r in rows]
    known = sorted(set().union(*vectors)) if vectors else []
    unknown = [w for w in weights if w not in known]
    if unknown:
        raise ValueError(f"no dimension {', '.join(unknown)}; the file has {', '.join(known)}")
    scaled = {d: percentiles([v.get(d, 0.0) for v in vectors]) for d in weights}
    out = []
    for i, row in enumerate(rows):
        contributions = {d: w * scaled[d][i] for d, w in weights.items()}
        out.append({"state": row.get("state"), "score": sum(contributions.values()),
                    "contributions": contributions})
    return sorted(out, key=lambda r: -r["score"])
