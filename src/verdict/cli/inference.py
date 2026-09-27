"""Model loading and answering, shared by every command that asks a question: the in-process
engine cache, checkpoint resolution, the server-then-local-fallback call, and `_Asker`, the one
place that resolves settings, picks a checkpoint per state and warns once.

Other cli submodules reach these through this module's own attributes (`inference.load(...)`,
`inference._Asker(...)`) rather than importing the names directly, so a test that monkeypatches
`inference.load` or `inference._ENGINES` reaches every caller through the one shared slot."""
from __future__ import annotations

import argparse
import os
import sys
import time

from .. import client
from ..engine import load
from ..errors import QuestionError

#: Checkpoints loaded in this process, by path, so a run without a server loads each model once,
#: not once per state (300 loads for a 300-state `decide --jsonl`).
_ENGINES: dict[str, object] = {}


def _resolved_model(flag: str | None, settings) -> str:
    """The `--model` flag as given, else the configured checkpoint, falling back to base laya
    with a warning when that is missing (`config.resolve_model`)."""
    from .. import config

    model, warning = config.resolve_model(flag or settings.model_path, explicit=bool(flag))
    if warning:
        print(f"verdict: {warning}", file=sys.stderr)
    return model


def _answer(state, questions: dict, url: str, model_path: str, model: str | None = None,
            budget: int | None = None, server_only: bool = False, bits: int = 16,
            flags: dict | None = None) -> dict:
    """Ask the server, or load the model here if none answers. `model` is laya's checkpoint name
    (`multilingual`), passed through to the server, or `model_path` is loaded in-process."""
    t = time.perf_counter()
    try:
        payload = client.decide(state, questions, url, model=model, **(flags or {}))
        source = f"server {url}"
    except client.NoServer as exc:
        if server_only:
            raise client.NoServer(f"{exc}; start `verdict serve` or check --url") from exc
        from ..schema import laya_questions

        # Validated and filled in as the server does, before paying for a model load.
        laya = laya_questions(questions)
        key = (model_path, bits)
        if key not in _ENGINES:
            print(f"verdict: {exc}; loading the model in this process (a few seconds and about "
                  "800 MB of GPU memory; `verdict serve` keeps one loaded)", file=sys.stderr)
            # A cached Hub checkpoint still draws a "Fetching 6 files" bar on stderr.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            _ENGINES[key] = load(model_path, bits=bits) if bits != 16 else load(model_path)
        eng = _ENGINES[key]
        # Clip as the server does, so the two paths never answer the same call differently.
        payload = {"model": eng.name, "answers": eng.predict(eng.clip_state(state, budget), laya)}
        source = "in-process"
    ms = (time.perf_counter() - t) * 1000
    return {**payload, "latency_ms": round(ms, 1), "source": source}


class _Asker:
    """One place that resolves settings, picks the checkpoint per state and warns once.

    `--lang auto` never switches models by itself. It runs laya's language detector (pure Python,
    microseconds) and says so when the English checkpoint is about to read something it cannot:
    off English it collapses and stays confident (Hindi: ECE 0.855 in laya's router notes), so a
    silent answer there is the worst kind. Loading the multilingual checkpoint is `--lang multi`.
    """

    def __init__(self, args: argparse.Namespace):
        from .. import config

        self.settings = config.load()
        self.url = config.check_url(args.url, "--url") if args.url else self.settings.url
        self.lang = getattr(args, "lang", None) or self.settings.lang
        self.server_only = getattr(args, "server_only", False)
        self.allow_unmeasured = getattr(args, "allow_unmeasured", False)
        self.yesno = getattr(args, "yesno", False)
        self.warn_clipped = getattr(args, "warn_clipped", True)
        self.main_path = ((args.model or self.settings.model_path) if self.server_only
                          else _resolved_model(args.model, self.settings))
        self._warned: set[str] = set()
        #: `--model` only chooses what loads locally; a server answers with its own checkpoint.
        self.model_flag = getattr(args, "model", None)

    def warn(self, line: str) -> None:
        if line not in self._warned:
            self._warned.add(line)
            print(f"verdict: {line}", file=sys.stderr)

    def __call__(self, state, questions: dict) -> dict:
        from .. import inputs
        from ..schema import laya_questions

        from .. import library

        questions = laya_questions(questions)
        answering = (self.settings.multilingual_path if self.lang == "multi"
                     else self.main_path)
        questions, rewritten, problems = library.prepare(
            state, questions, model=answering, allow_unmeasured=self.allow_unmeasured,
            yesno=self.yesno)
        for line in problems:
            self.warn(line)
        clipped = (inputs.maybe_clipped(state, self.settings.prompt_token_budget)
                   if self.warn_clipped else None)
        if clipped:
            self.warn(clipped)
        result = self._ask(state, questions)
        served = str(result.get("model", ""))
        if (self.model_flag and result.get("source", "").startswith("server")
                and not served.endswith(str(self.model_flag).rstrip("/"))):
            self.warn(f"--model {self.model_flag} was not used: the server at {self.url} answered "
                      f"with {served}. --model picks what loads when no server answers; stop the "
                      "server, or pass --url for one serving that model")
        result["answers"] = library.as_yesno(result["answers"], rewritten)
        return result

    def _ask(self, state, questions: dict) -> dict:
        from .. import inputs

        if self.lang == "multi":
            return _answer(state, questions, self.url, self.settings.multilingual_path,
                           "multilingual", self.settings.prompt_token_budget,
                           server_only=self.server_only, flags=self._flags())
        if self.lang == "auto":
            info = inputs.language(state)
            if info and not info["is_english"]:
                seen = info.get("language") or info["script"]
                self.warn(f"the state looks like {seen}, which the English checkpoint cannot read "
                          "reliably; add --lang multi to use laya's multilingual one")
        return _answer(state, questions, self.url, self.main_path,
                       budget=self.settings.prompt_token_budget, server_only=self.server_only,
                       bits=self.settings.bits, flags=self._flags())

    def _flags(self) -> dict:
        """The opt-outs the server needs to answer as this CLI call asked: only those set, so a
        call without them looks as it always has."""
        return {k: True for k, on in (("allow_unmeasured", self.allow_unmeasured),
                                      ("yesno", self.yesno)) if on}


def _refusal(problems: list[str]) -> str:
    from .. import library

    return library.refusal(problems)
