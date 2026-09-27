"""What the CLI reads before it asks anything: the state, the questions, and two checks on them.

Laya's own guidance shapes all of it. A state is best a JSON object with named fields, which laya
serializes as JSON and which questions refer to in backticks ("What does the customer want in
`message`?"). A bank of questions is answered together in one pass. And the English checkpoint
does not degrade gently off English: it collapses while staying confident, so the language is
worth checking on every call, which costs microseconds.

Nothing here imports MLX, transformers or the model. The two laya files this reads, its question
presets and its language detector, are pure Python, and importing them through the `laya_mlx`
package would run its `__init__`, which loads MLX: seconds, for five dicts and a regex.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

#: Laya's built-in banks, by name, with the state field each one asks about.
PRESETS = ("guard", "triage", "moderation", "router", "email")

_FIELD = re.compile(r"`([A-Za-z_][\w.-]*)`")


def parse_state(text: str) -> Any:
    """A JSON object or array becomes a structured state; anything else stays text.

    Text that looks like JSON and fails to parse is an error, not text: answering a truncated
    object as a sentence exits 0 with a confident number about the wrong thing. `[WIP] fix it`
    is still text, since it does not also end in `]`.
    """
    text = text.strip()
    if not text:
        raise ValueError("the state is empty; give text, a JSON object, @file or - for stdin")
    looks_like_json = text[0] == "{" or (text[0] == "[" and text[-1] == "]")
    if looks_like_json:
        try:
            return json.loads(text)
        except ValueError as exc:
            raise ValueError(f"the state starts like JSON but is not valid JSON ({exc}); "
                             "pass it as a JSON string to send it as text") from exc
    return text


def read_state(arg: str) -> Any:
    """`-` reads stdin, `@path` reads a file, anything else is the state itself."""
    if arg == "-":
        return parse_state(sys.stdin.read())
    if arg.startswith("@"):
        try:
            return parse_state(Path(arg[1:]).read_text())
        except FileNotFoundError:
            raise ValueError(f"{arg}: no such file") from None
    return parse_state(arg)


def load_questions(spec: str) -> dict[str, Any]:
    """A preset name, library names (comma-separated), inline JSON, or a path to a JSON file,
    tried in that order."""
    if spec in PRESETS:
        return preset(spec)
    if not spec.lstrip().startswith(("{", "[")) and not Path(spec.lstrip("@")).is_file():
        from . import library

        names = [n.strip() for n in spec.split(",") if n.strip()]
        entries = {e.name: e for e in library.load()}
        if names and all(n in entries for n in names):
            return {n: entries[n].question for n in names}
        unknown = [n for n in names if n not in entries]
        if len(names) > 1 or re.fullmatch(r"[a-z][a-z0-9_]*", spec):
            from .cli.support import _suggest

            hint = _suggest(unknown[0], [*entries, *PRESETS]) if len(unknown) == 1 else "."
            raise ValueError(f"--questions: no preset or library question {', '.join(unknown)}"
                             f"{hint} See `verdict questions` and `verdict presets`")
    if spec.lstrip().startswith(("{", "[")):
        bank = json.loads(spec)
        if not isinstance(bank, dict):
            raise ValueError("--questions: a question bank is a JSON object of name -> question, "
                             'like {"q": {"type": "noul", "instructions": "..."}}')
        return bank
    path = Path(spec[1:] if spec.startswith("@") else spec)
    if not path.is_file():
        if spec.startswith("@") or path.suffix == ".json" or "/" in spec:
            raise ValueError(f"--questions {spec}: no such file")
        raise ValueError(f"--questions {spec!r} is not a preset ({', '.join(PRESETS)}), "
                         "inline JSON or a file")
    return json.loads(path.read_text())


def _laya_file(name: str):
    """Execute one pure-Python file from the installed `laya_mlx` without importing the package."""
    import importlib.util

    spec = importlib.util.find_spec("laya_mlx")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("this needs laya-mlx: uv sync --extra mlx --extra laya")
    path = Path(list(spec.submodule_search_locations)[0]) / f"{name}.py"
    module_spec = importlib.util.spec_from_file_location(f"_laya_{name}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def preset(name: str) -> dict[str, Any]:
    if name not in PRESETS:
        from .cli.support import _suggest

        raise ValueError(f"no preset {name!r}{_suggest(name, PRESETS)} There are: "
                         f"{', '.join(PRESETS)}")
    return getattr(_laya_file("presets"), f"{name}_questions")()


def fields_named(questions: dict[str, Any]) -> list[str]:
    """Every `field` the questions refer to, in first-seen order."""
    seen: dict[str, None] = {}
    for q in questions.values():
        for f in _FIELD.findall(q.get("instructions", "")):
            seen.setdefault(f, None)
    return list(seen)


def instruction_text(q: Any) -> str:
    """A question's instructions as text: structured ones as the JSON laya reads, missing as ""."""
    ins = q.get("instructions") if isinstance(q, dict) else None
    if ins is None:
        return ""
    return ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False)


#: Characters per budget token past which a state is certainly over the clip budget. English prose
#: runs about 4 characters a token; 5 leaves headroom so a state near the edge is not flagged. A
#: pre-filter, like `engine.maybe_truncated`: counting tokens exactly would load the tokenizer,
#: 1.3 s against a 30 ms call.
CHARS_PER_TOKEN = 5


def maybe_clipped(state: Any, budget: int | None) -> str | None:
    """A warning when `state` is far past the clip budget, so only its start is read (§29).

    One fixed line, not per state, so `--jsonl` over thousands of long states says it once."""
    if not budget:
        return None
    text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    if len(text) <= budget * CHARS_PER_TOKEN:
        return None
    return (f"a state is over {budget * CHARS_PER_TOKEN} characters; verdict reads only its first "
            f"{budget} tokens (about {budget * 4} characters). Put the field that matters first, "
            "or split long text")


def missing_fields(state: Any, questions: dict[str, Any]) -> list[str]:
    """Questions that name a `field` the dict state lacks. The model still answers, from nothing,
    and nothing else would say so: a `triage` bank asked of `{"body": ...}` is the usual case."""
    if isinstance(state, str):
        # A text state has no fields at all, so a question about `message` is answered from
        # nothing just as surely as a dict without that key. One line for the whole bank: a
        # preset names the same field in every question.
        asking = {name: _FIELD.findall(instruction_text(q)) for name, q in questions.items()}
        asking = {name: fields for name, fields in asking.items() if fields}
        if not asking:
            return []
        fields = list(dict.fromkeys(f for fs in asking.values() for f in fs))
        shape = ", ".join(f'"{f}": ...' for f in fields)
        return [f"the state is plain text, but {', '.join(asking)} "
                f"{'asks' if len(asking) == 1 else 'ask'} about "
                f"{', '.join(f'`{f}`' for f in fields)}; pass {{{shape}}} instead"]
    if not isinstance(state, dict):
        return []
    out = []
    for name, q in questions.items():
        missing = [f for f in _FIELD.findall(instruction_text(q)) if f not in state]
        if missing:
            out.append(f"question {name!r} asks about {', '.join(f'`{f}`' for f in missing)}, "
                       f"which the state does not have (keys: {', '.join(state) or 'none'})")
    return out


_lang = None


def language(state: Any) -> dict[str, Any] | None:
    """laya's own detector: script, best-guess language, and whether the English checkpoint can
    be expected to read it. None when laya-mlx is not installed, since this is only a warning."""
    global _lang
    if _lang is None:
        try:
            _lang = _laya_file("lang")
        except (ValueError, OSError):
            return None
    info = _lang.analyse(state)
    # laya guesses a Latin-script language from word frequencies, and shell reads as Portuguese or
    # French to it: 41 of 3,000 real commands were flagged. A non-Latin script is unambiguous, so
    # only a Latin-script guess is second-guessed, and only on text that is mostly code punctuation
    # (flagged commands started at 0.054; non-English prose topped out at 0.037).
    if info["script"] == "latin" and not info["is_english"]:
        text = _lang.state_text(state)
        if sum(c in _CODE_CHARS for c in text) / max(1, len(text)) >= CODE_RATIO:
            info = {**info, "is_english": True, "looks_like_code": True}
    return info


_CODE_CHARS = frozenset("/$|=;{}<>~\\`\"'_*()[]#&@")
CODE_RATIO = 0.05


def inline_question(question: str, options: list[str] | None = None,
                    levels: list[str] | None = None, true: str | None = None,
                    false: str | None = None) -> dict[str, Any]:
    """One question from flags: yes/no by default, `choice` with options, `score` with levels.

    Options take `NAME` or `NAME=description`. Describing the sides of a yes/no is allowed because
    laya allows it, though on this project's data it hurt (FINDINGS §22).
    """
    if not question.strip():
        raise ValueError("the question is empty; ask something about the text")
    if options and levels:
        raise ValueError("give options (-o) or levels (-l), not both")
    if (options or levels) and (true or false):
        raise ValueError("--true/--false describe a yes/no question; drop -o/-l to use them")
    if options is not None:
        if len(options) < 2:
            raise ValueError("a choice needs at least two -o options")
        criteria = dict(o.split("=", 1) if "=" in o else (o, o) for o in options)
        return {"type": "choice", "instructions": question, "criteria": criteria}
    if levels is not None:
        if len(levels) < 2:
            raise ValueError("a score needs at least two -l levels, lowest first")
        return {"type": "score", "instructions": question, "criteria": levels}
    q: dict[str, Any] = {"type": "noul", "instructions": question}
    sides = {k: v for k, v in (("true", true), ("false", false)) if v}
    if sides:
        q["criteria"] = sides
    return q
