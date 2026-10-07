"""Measured questions, usable by name, and a linter built from the ones that failed.

The questions that worked lived in a skill file, where an agent had to find and copy them. Here
each one ships with what it was measured on and how it did, and `-q NAME` uses it. An entry is
either measured on real labels (a FINDINGS section) or backed by a `verdict bench` suite, in which
case its question is read from the suite, so the two cannot drift apart.

The linter is FINDINGS' failed questions turned into a check. Each of them asked about something
not on the page, a consequence or a difficulty, and each scored at chance (§25, §29), while plain
questions about what the text says or does worked. `ask` and `decide` refuse what it flags unless
given `--allow-unmeasured`; `calibrate` only warns, because calibrating is how a question gets
measured.
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from . import inputs

LIBRARY_FILE = Path(__file__).with_name("library.json")


@dataclass
class Entry:
    name: str
    question: dict[str, Any]
    #: What the state looked like when it was measured, e.g. `{"command": ...}`.
    state: str
    measured: str
    source: str
    bench: Optional[str] = None


def load(path: Path = LIBRARY_FILE) -> list[Entry]:
    from . import bench

    suites = {s.name: s for s in bench.load_suites()}
    out = []
    for d in json.loads(Path(path).read_text())["questions"]:
        d = dict(d)
        if d.get("bench"):
            d["question"] = suites[d["bench"]].question
        out.append(Entry(**d))
    return out


def get(name: str) -> Optional[Entry]:
    return next((e for e in load() if e.name == name), None)


#: Words that ask about a consequence or a difficulty rather than about the text itself.
_OFF_PAGE = re.compile(
    # Not bare "harm": "Does `post` threaten violence, harm or intimidation?" asks what the text
    # says. "cause" is what makes harm a consequence, and it catches that form.
    r"\b(hard(er)?\s+(is|for|to)|difficult(y|ies)?|consequences?|risk(y|s)?|dangerous|cause)\b",
    re.IGNORECASE,
)


#: A yes/no about whether something is unremarkable asks about the absence of a property, which
#: the text never shows. Four of four such questions failed (§33, AUC 0.32 to 0.64) where asking
#: for the property itself scored 0.93 to 0.99. As a choice option the same word worked.
_ABSENCE = re.compile(r"\b(ordinary|normal|regular|typical|benign|legitimate)\b", re.IGNORECASE)


#: Past this many options a choice degrades: they share one token budget, so at 77 each gets 3 or
#: 4 tokens, and Laya scored 0.43 on Banking77 against Jev's 0.87 (Laya's README; FINDINGS §41).
MAX_OPTIONS = 20


def lint(questions: dict[str, Any]) -> list[str]:
    """Warnings for questions whose wording has failed before. Empty when nothing matches."""
    out = []
    for qid, q in questions.items():
        n = len(q.get("criteria") or {}) if q.get("type") == "choice" else 0
        if n > MAX_OPTIONS:
            out.append(f"{qid} has {n} options; past {MAX_OPTIONS} they share too few tokens to "
                       "tell apart (Banking77, 77 options: 0.43, FINDINGS §41). Split it into "
                       "yes/no questions, shortlist the options first, or use an LLM")
        text = inputs.instruction_text(q)
        m = _OFF_PAGE.search(text)
        if m:
            out.append(f"{qid} asks about {m.group(0).strip()!r}, a consequence or difficulty the "
                       "text does not show; questions like that scored at chance (FINDINGS §25, "
                       "§29). Ask what the text says or does instead")
            continue
        m = _ABSENCE.search(text)
        if m and q.get("type") == "noul":
            out.append(f"{qid} asks whether the text is {m.group(0)!r}, the absence of a property; "
                       "yes/no questions like that scored near chance (FINDINGS §33). Ask for "
                       "the property itself, or make it a choice with both sides named")
    return out


def _canonical(q: dict[str, Any]) -> str:
    return json.dumps({k: v for k, v in q.items() if v not in (None, "")}, sort_keys=True)


@functools.cache
def _measured() -> frozenset[str]:
    return frozenset(_canonical(e.question) for e in load())


def is_measured(question: dict[str, Any]) -> bool:
    """Whether `question` is a library entry's question, as it was measured."""
    return _canonical(question) in _measured()


#: The checkpoint the library's yes/no questions were measured on, by directory name.
MEASURED_ON = "verdict-v1-mlx"


def measured_on(model: str) -> bool:
    """Whether `model` (a path or Hub id) is the checkpoint the library was measured on."""
    return Path(str(model)).name == MEASURED_ON


def as_choice(questions: dict[str, Any], keep_measured: bool = True
              ) -> tuple[dict[str, Any], set[str]]:
    """Each yes/no rewritten as a choice between `no` and `yes`, and the ids rewritten.

    A plain yes/no can follow its own `false`/`true` labels instead of the state: SST-2's "is it
    positive?" ranked 0.79 with no positive over 0.5, and as a no/yes choice 0.96, with spam and
    injection unchanged (FINDINGS §38). With `keep_measured`, a library yes/no keeps the shape it
    was measured in; that only holds on the checkpoint it was measured on, since base Laya's
    plain yes/no is the form that collapses (SST-2 0.51, as a choice 0.94, §40).
    """
    out, rewritten = {}, set()
    for qid, q in questions.items():
        if q.get("type") != "noul" or (keep_measured and is_measured(q)):
            out[qid] = q
            continue
        sides = q.get("criteria") or {}
        out[qid] = {"type": "choice", "instructions": q.get("instructions", ""),
                    "criteria": {"no": sides.get("false", ""), "yes": sides.get("true", "")}}
        rewritten.add(qid)
    return out, rewritten


#: How each front door spells the opt-out, so a refusal names the one its caller can use.
CLI_OPT_OUT = "--allow-unmeasured"
HTTP_OPT_OUT = "?allow_unmeasured=true"


def prepare(state: Any, questions: dict[str, Any], *, model: str, allow_unmeasured: bool = False,
            yesno: bool = False, opt_out: str = CLI_OPT_OUT) -> tuple[dict[str, Any], set[str], list[str]]:
    """What every front door does before asking: refuse a question about a field the state lacks
    and the shapes that measured at chance, then ask each new yes/no as a no/yes choice.

    Returns the questions to send, the ids rewritten (for `as_yesno`), and the problems let
    through by `allow_unmeasured`, as warnings. The CLI and `POST /v1/decide` both call this,
    so the same question gets the same treatment and the same number from either.
    """
    from .errors import QuestionError

    missing = inputs.missing_fields(state, questions)
    problems = lint(questions)
    if missing and not allow_unmeasured:
        raise QuestionError("; ".join(missing) + ". Refused, because the model would answer from "
                            f"nothing; fix the state or the question, or {opt_out} to "
                            "ask anyway")
    if problems and not allow_unmeasured:
        raise QuestionError(refusal(problems, opt_out))
    rewritten: set[str] = set()
    if not yesno:
        questions, rewritten = as_choice(questions, keep_measured=measured_on(model))
    return questions, rewritten, missing + problems


def refusal(problems: list[str], opt_out: str = CLI_OPT_OUT) -> str:
    return ("; ".join(problems) + ". Refused, because answers to this shape measured at chance; "
            f"{opt_out} asks anyway")


def as_yesno(answers: dict[str, Any], rewritten: set[str]) -> dict[str, Any]:
    """Choice answers to rewritten questions, back in the yes/no shape callers read."""
    out = dict(answers)
    for qid in rewritten:
        a = answers[qid]
        if a.get("type") != "choice":
            continue  # a backend that answered the yes/no as asked before; nothing to convert
        p = a["probabilities"]["yes"]
        # laya's yes/no confidence is max(p, 1 - p); a choice's is 1 - normalized entropy, a
        # different scale, so it is recomputed rather than passed through.
        out[qid] = {"type": "noul", "noul": p, "confidence": round(max(p, 1 - p), 4),
                    "asked_as": "choice"}
    return out
