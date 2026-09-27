"""The answer-quality scorecard, and the check a release must pass.

`pytest` proves the code does what it says; nothing proved the answers were any good, so a
checkpoint or a laya bump could get worse and ship. `verdict bench` asks each suite's question of
a fixed, balanced sample of a public labelled dataset and writes a scorecard: AUC with a bootstrap
interval for a yes/no question, accuracy with recall per option for a choice. `verify` compares a
scorecard against the previous release's with no model, which is what the release workflow can
run, since CI has no checkpoint.

Suites are pinned (dataset, full revision sha, split, seed) and never vendored: several of these
datasets carry no clear license, and a pinned revision reproduces the same sample anyway.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from . import calibrate, inputs
from .metrics import bootstrap_ci

SUITES_FILE = Path(__file__).with_name("bench_suites.json")

#: A model call: (state, questions) -> {"answers": {...}}, the shape `client.decide` returns.
Ask = Callable[[Any, dict], dict]


@dataclass
class Suite:
    name: str
    dataset: str
    revision: str
    split: str
    text: str
    label: str
    #: Dataset label value (as a string) -> the class name the question uses.
    classes: dict[str, str]
    #: One question, in the wire format. Asked as id "q".
    question: dict[str, Any]
    #: For a yes/no question, the class a "yes" means.
    target: Optional[str] = None
    per_class: int = 100
    seed: int = 0
    note: str = ""
    #: Drop rows laya's language detector says the English checkpoint cannot read. The injection
    #: set mixes in German, which would measure the language gap rather than the question.
    english_only: bool = False
    #: Dataset labels to skip, as strings: a neutral or mixed class in a positive-versus-negative
    #: suite. Any other label outside `classes` still fails.
    ignore: list = field(default_factory=list)
    #: The checkpoint that reads this suite: "multi" for the multilingual one (Cyrillic and other
    #: non-English text), "en" or None for the served one.
    lang: Optional[str] = None
    config: Optional[str] = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Suite":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        s = cls(**{k: v for k, v in d.items() if k in known},
                extra={k: v for k, v in d.items() if k not in known})
        s.check()
        return s

    def check(self) -> None:
        if self.lang not in (None, "en", "multi"):
            raise ValueError(f"suite {self.name}: lang is {self.lang!r}; it takes en or multi")
        kind = self.question.get("type")
        names = set(self.classes.values())
        if kind == "noul":
            if self.target not in names:
                raise ValueError(f"suite {self.name}: a yes/no suite needs a target class, one of "
                                 f"{sorted(names)}")
        elif kind == "choice":
            crit = self.question.get("criteria") or []
            missing = names - set(crit)
            if missing:
                raise ValueError(f"suite {self.name}: choice criteria lack {sorted(missing)}")
        else:
            raise ValueError(f"suite {self.name}: only noul and choice are benchmarked, not {kind!r}")


def load_suites(path: Path = SUITES_FILE) -> list[Suite]:
    return [Suite.from_dict(d) for d in json.loads(Path(path).read_text())["suites"]]


def fetch(s: Suite) -> list[dict]:
    """The dataset split at its pinned revision. Needs the `bench` extra and, once, the network."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ValueError("bench needs the datasets library: uv sync --extra mlx --extra laya "
                         "--extra bench") from exc
    return list(load_dataset(s.dataset, s.config, split=s.split, revision=s.revision))


def sample(s: Suite, rows: list[dict]) -> list[tuple[str, str]]:
    """`per_class` rows of each class, drawn with the suite's seed; blank text is skipped."""
    by_class: dict[str, list[str]] = {c: [] for c in s.classes.values()}
    for r in rows:
        text = (r[s.text] or "").strip()
        if not text:
            continue
        if s.english_only and not (inputs.language({"text": text}) or {}).get("is_english", True):
            continue
        key = str(r[s.label])
        if key in s.ignore:
            continue
        if key not in s.classes:
            raise ValueError(f"suite {s.name}: label {key} is not in classes {sorted(s.classes)}")
        by_class[s.classes[key]].append(text)
    rng = random.Random(s.seed)
    out = []
    for cls in sorted(by_class):
        pool = by_class[cls]
        out += [(t, cls) for t in rng.sample(pool, min(s.per_class, len(pool)))]
    return out


def _auc(pairs):
    return calibrate.auc([p for p, _ in pairs], [y for _, y in pairs]) or 0.0


def score(s: Suite, items: list[tuple[str, str]], ask: Ask) -> dict[str, Any]:
    answers = [ask({"text": text}, {"q": s.question})["answers"]["q"] for text, _ in items]
    truth = [c for _, c in items]
    if s.question["type"] == "noul":
        scores = [a["noul"] for a in answers]
        labels = [c == s.target for c in truth]
        pos = [p for p, y in zip(scores, labels) if y]
        lo, hi = bootstrap_ci(list(zip(scores, labels)), _auc, iters=1000)
        return {
            "metric": "auc", "value": round(_auc(list(zip(scores, labels))), 4),
            "ci": [round(lo, 4), round(hi, 4)], "n": len(items),
            # The "leans no" symptom (FINDINGS §2, §33): how often the class a yes means clears
            # the naive 0.5 line. A good AUC with this near 0 means rank or fit a cut, never 0.5.
            "target_over_half": round(sum(p >= 0.5 for p in pos) / len(pos), 4) if pos else None,
            "fit": calibrate.fit_noul(scores, labels),
        }
    fit = calibrate.fit_choice([a["probabilities"] for a in answers], truth)
    hits = [a["choice"] == c for a, c in zip(answers, truth)]
    lo, hi = bootstrap_ci(hits, lambda h: sum(h) / len(h), iters=1000)
    out = {
        "metric": "accuracy", "value": round(sum(hits) / len(hits), 4),
        "ci": [round(lo, 4), round(hi, 4)], "n": len(items),
        "chance": round(1 / len(s.classes), 4), "fit": fit,
    }
    if len(s.classes) == 2:
        # Accuracy at the argmax mixes ranking with where the model puts the line; the yes/no
        # suites report AUC, so a binary choice carries it too. Symmetric in which side is "a".
        a = sorted(s.classes.values())[0]
        out["auc"] = round(_auc([(x["probabilities"][a], c == a) for x, c in zip(answers, truth)]), 4)
    return out


def verify(current: dict, previous: Optional[dict]) -> list[str]:
    """Regressions against the previous scorecard: a suite that vanished, or a value below the
    previous run's interval. Inside the interval is noise at these sample sizes, not a change."""
    if previous is None:
        return []
    problems = []
    for name, prev in previous["suites"].items():
        cur = current["suites"].get(name)
        if cur is None:
            problems.append(f"{name}: in the previous scorecard, missing from this one")
            continue
        if cur["value"] < prev["ci"][0]:
            problems.append(f"{name}: {cur['metric']} {cur['value']:.2f}, below the previous "
                            f"interval [{prev['ci'][0]:.2f}, {prev['ci'][1]:.2f}]")
    return problems


def _version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(p) for p in path.stem.lstrip("v").split("."))


def scorecards(directory: Path) -> list[Path]:
    return sorted(Path(directory).glob("v*.json"), key=_version_key)


def previous(directory: Path, version: str) -> Optional[Path]:
    """The newest scorecard older than `version`."""
    mine = _version_key(Path(f"v{version}.json"))
    older = [p for p in scorecards(directory) if _version_key(p) < mine]
    return older[-1] if older else None


# ---- a /v1/systemone contestant ------------------------------------------------------------------

def _default_transport():
    return None


def systemone_asker(base_url: str, model: str, key: str, *, attempts: int = 5,
                    backoff: float = 1.0, timeout: float = 30.0, transport=None):
    """A closable, callable SystemOne client. No Laya preprocessing is applied upstream."""
    from .systemone import SystemOneClient

    return SystemOneClient(base_url, model, key, attempts=attempts, backoff=backoff,
                           timeout=timeout, transport=transport or _default_transport())
