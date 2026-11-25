"""Model loading and answering, shared by every command that asks a question: the in-process
engine cache, checkpoint resolution, the server-then-local-fallback call, and `_Asker`, the one
place that resolves settings, picks a checkpoint per state and warns once.

Other cli submodules reach these through this module's own attributes (`inference.load(...)`,
`inference._Asker(...)`) rather than importing the names directly, so a test that monkeypatches
`inference.load` or `inference._ENGINES` reaches every caller through the one shared slot."""
from __future__ import annotations

import argparse
import json
import os
import time

from .. import client
from .. import config
from ..backends.engine import load
from ..errors import QuestionError
from . import support

#: What `/v1/capabilities` reports for a server fronting something other than laya.
PROXY_FAMILIES = ("jev", "systemone")
_UNASKED = object()

#: Checkpoints loaded in this process, by path, so a run without a server loads each model once,
#: not once per state (300 loads for a 300-state `decide --jsonl`).
_ENGINES: dict[str, object] = {}


def _say_local(exc: Exception) -> None:
    support._warn(f"{exc}; loading the model in this process (a few seconds and about 800 MB "
                  "of GPU memory; `verdict serve` keeps one loaded)")


def write_atomic(path, text: str) -> None:
    """Write `text` to `path` through a temp file in the same directory and `os.replace`, so a
    killed run leaves the old file or the new one, never half of one (a scorecard gates a release)."""
    from pathlib import Path

    # Through a symlink to the real file (a dotfiles repo), so the link survives; the mode of the
    # file being replaced is kept, since a new temp file would otherwise reset it.
    target = Path(os.path.realpath(path))
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        mode = target.stat().st_mode & 0o7777 if target.exists() else None
        tmp.write_text(text)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def write_json_atomic(path, doc) -> None:
    """`write_atomic` of `doc` as indented JSON with a trailing newline."""
    write_atomic(path, json.dumps(doc, indent=2) + "\n")


def _resolved_model(flag: str | None, settings) -> str:
    """The `--model` flag as given, else the configured checkpoint, falling back to base laya
    with a warning when that is missing (`config.resolve_model`)."""
    from .. import config

    model, warning = config.resolve_model(flag or settings.model_path, explicit=bool(flag))
    if warning:
        support._warn(warning)
    return model


def _kind(result: dict) -> str:
    """Who answered, as `Asker.sources` records it: "server", or "in-process" for anything else."""
    return "server" if str(result.get("source", "")).startswith("server") else "in-process"


def _url(args: argparse.Namespace, settings) -> str:
    """The `--url` flag (checked), else the configured server."""
    from .. import config

    return config.check_url(args.url, "--url") if args.url else settings.url


def _budget(args: argparse.Namespace, settings) -> int | None:
    """The `--budget` flag when given (0 included), else the configured budget."""
    return args.budget if args.budget is not None else settings.prompt_token_budget


def _max_len(args: argparse.Namespace, settings) -> int:
    """The `--max-len` flag when given (checked), else the configured `max_len` (0: the
    checkpoint's own)."""
    flag = getattr(args, "max_len", None)
    return config.check_max_len(flag, "--max-len") if flag is not None else settings.max_len


def _answer(state, questions: dict, url: str, model_path: str, model: str | None = None,
            budget: int | None = None, server_only: bool = False, bits: int = 16,
            flags: dict | None = None, pin: tuple[str, int] | None = None,
            max_len: int = 0) -> dict:
    """Ask the server, or load the model here if none answers. `model` is laya's checkpoint name
    (`multilingual`), passed through to the server, or `model_path` is loaded in-process.
    `pin` is `(source, row)` once a run's first row was answered: `"in-process"` never asks the
    server again, `"server"` never falls back, and a server lost at `row` is an error."""
    t = time.perf_counter()
    try:
        if pin and pin[0] == "in-process":
            raise client.NoServer("this run is pinned to in-process")
        payload = client.decide(state, questions, url, model=model, **(flags or {}))
        source = f"server {url}"
    except client.NoServer as exc:
        if server_only:
            raise client.NoServer(f"{exc}; start `verdict serve` or check --url") from exc
        if pin and pin[0] == "server":
            raise client.NoServer(f"call {pin[1]}: {exc}; this run was pinned to the server by its "
                                  "first answer, so it does not fall back to another model") from exc
        from ..schema import laya_questions

        # Validated and filled in as the server does, before paying for a model load.
        laya = laya_questions(questions)
        key = (model_path, bits, max_len)
        if key not in _ENGINES:
            if not pin:
                _say_local(exc)
            # A cached Hub checkpoint still draws a "Fetching 6 files" bar on stderr.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            opts = {**({"bits": bits} if bits != 16 else {}), **({"max_len": max_len} if max_len else {})}
            _ENGINES[key] = load(model_path, **opts)
            # Judged against the checkpoint's own limit, known once its agent is loaded: the
            # multilingual one reads 1024, so 768 raises nothing there.
            if max_len:
                getattr(_ENGINES[key], "agent", None)
                native = getattr(_ENGINES[key], "native_max_len", None) or 512
                note = config.max_len_note(max_len, budget, native)
                if note:
                    support._warn(note)
        eng = _ENGINES[key]
        # Clip as the server does, so the two paths never answer the same call differently.
        clipped = eng.clip_state(state, budget)
        answers = eng.predict(clipped, laya)
        if clipped is not state:
            answers = {qid: {**a, "truncated": True} for qid, a in answers.items()}
        payload = {"model": eng.name, "answers": answers}
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
        #: `--max-len` over `$VERDICT_MAX_LEN` over the file, checked here so a bad one stops the command.
        self.settings.max_len = _max_len(args, self.settings)
        self.url = _url(args, self.settings)
        self.lang = getattr(args, "lang", None) or self.settings.lang
        self.server_only = getattr(args, "server_only", False)
        self.allow_unmeasured = getattr(args, "allow_unmeasured", False)
        self.yesno = getattr(args, "yesno", False)
        #: `--abstain-below`, already validated by the command (`abstain.parse_threshold`).
        self.abstain_below = getattr(args, "abstain_below", None)
        self.warn_clipped = getattr(args, "warn_clipped", True)
        self.main_path = ((args.model or self.settings.model_path) if self.server_only
                          else _resolved_model(args.model, self.settings))
        self._warned: set[str] = set()
        #: What the server fronts (`/v1/capabilities` family), asked once; see `_proxied`.
        self._family: object = _UNASKED
        #: Whether a failed probe was already repeated; see `__call__`.
        self._reasked = False
        #: Where and with what the first answer came, so a switch mid-run is said once.
        self._first: tuple[str, str] | None = None
        #: Rows asked so far, and the source the first answer pinned the rest of the run to.
        self._rows = 0
        self._pin: str | None = None
        #: Every checkpoint that answered this run.
        self.models: set[str] = set()
        #: `"server"` and `"in-process"`: who answered. A scorecard is stamped from this, not from
        #: the client's own settings, which describe only what would load here.
        self.sources: set[str] = set()
        #: `--model` only chooses what loads locally; a server answers with its own checkpoint.
        self.model_flag = getattr(args, "model", None)

    def _proxied(self) -> bool:
        """Whether a server answers with a model that is not laya (hosted Jev, Kev, another
        verdict). Laya's yes/no rewrite and option cap are about laya's token budget, so they apply
        only when laya is what answers: with no server, or one that cannot say, laya's rules stand.
        One cheap loopback GET per command, never per row."""
        if self._pin == "in-process":
            # Pinned: laya answers every row of this run, so laya's rules apply and a server that
            # appears later is not asked what it fronts.
            return False
        if self._family is _UNASKED:
            try:
                self._family = client.capabilities(self.url).get("family")
            except (client.NoServer, client.ServerError, client.ServerTimeout):
                self._family = None
        return self._family in PROXY_FAMILIES

    def warn(self, line: str) -> None:
        if line not in self._warned:
            self._warned.add(line)
            support._warn(line)

    def __call__(self, state, questions: dict) -> dict:
        from .. import inputs
        from ..schema import laya_questions

        from .. import library

        questions = laya_questions(questions)
        answering = (self.settings.multilingual_path if self.lang == "multi"
                     else self.main_path)
        proxied = self._proxied()
        questions, rewritten, problems = library.prepare(
            state, questions, model=answering, allow_unmeasured=self.allow_unmeasured,
            yesno=self.yesno or proxied, max_options=None if proxied else library.MAX_OPTIONS)
        for line in problems:
            self.warn(line)
        clipped = (inputs.maybe_clipped(state, self.settings.prompt_token_budget)
                   if self.warn_clipped else None)
        if clipped:
            self.warn(clipped)
        self._rows += 1
        result = self._ask(state, questions)
        kind = _kind(result)
        if self._pin is None:
            self._pin = kind
            if self._pin == "in-process" and not self.server_only:
                self.warn("this run is pinned to in-process: a server that starts later is ignored, "
                          "so every row is answered by the same model")
        self._note_source(result, kind)
        if result.get("source") == "in-process":
            # The server went away and laya answered here: its rules apply from the next row, and
            # a server that comes back gets one fresh probe (`_reasked` is spent per appearance).
            # Once the run is pinned in-process `_proxied` never probes, so this only matters for
            # a run pinned to a server.
            self._reasked = False
            if proxied:
                self._family = _UNASKED
        elif (self._family is None and not self._reasked
              and result.get("source", "").startswith("server")):
            # No server could say what it fronts, and now one answers: it may have started after
            # the first row, and it may be a proxy. Ask once more, not on every row: a server with
            # no capabilities endpoint would cost a request per row for the same answer.
            self._family, self._reasked = _UNASKED, True
        served = str(result.get("model", ""))
        if (self.model_flag and result.get("source", "").startswith("server")
                and not served.endswith(str(self.model_flag).rstrip("/"))):
            self.warn(f"--model {self.model_flag} was not used: the server at {self.url} answered "
                      f"with {served}. --model picks what loads when no server answers; stop the "
                      "server, or pass --url for one serving that model")
        result["answers"] = library.as_yesno(result["answers"], rewritten)
        if self.abstain_below is not None:
            from ..judge import abstain

            # After every rewrite, so a yes/no asked as a choice is gated as the yes/no it is.
            result["answers"] = abstain.gate(result["answers"], float(self.abstain_below),
                                             uniform=result.get("model") == "uniform")
        return result

    def _note_source(self, result: dict, kind: str) -> None:
        """Say once on stderr when a run changes checkpoint between rows: a server that appears or
        goes away mid-run answers the rest with another model, and a fit or a spread taken over
        both mixes two."""
        model = str(result.get("model", ""))
        self.models.add(model)
        self.sources.add(kind)
        if self._first is None:
            self._first = (kind, model)
        elif (kind, model) != self._first:
            self.warn(f"answers switched from {self._first[0]} ({self._first[1]}) to {kind} "
                      f"({model}) during this run; rows before and after were not answered by "
                      "the same model (pass --server-only, or stop the stray server, to pin it)")

    def _ask(self, state, questions: dict) -> dict:
        from .. import inputs

        pin = (self._pin, self._rows) if self._pin else None

        if self.lang == "multi":
            return _answer(state, questions, self.url, self.settings.multilingual_path,
                           "multilingual", self.settings.prompt_token_budget,
                           server_only=self.server_only, flags=self._flags(), pin=pin,
                           max_len=self.settings.max_len)
        if self.lang == "auto":
            info = inputs.language(state)
            if info and not info["is_english"]:
                seen = info.get("language") or info["script"]
                self.warn(f"the state looks like {seen}, which the English checkpoint cannot read "
                          "reliably; add --lang multi to use laya's multilingual one")
        return _answer(state, questions, self.url, self.main_path,
                       budget=self.settings.prompt_token_budget, server_only=self.server_only,
                       bits=self.settings.bits, flags=self._flags(), pin=pin,
                       max_len=self.settings.max_len)

    def _flags(self) -> dict:
        """The opt-outs the server needs to answer as this CLI call asked: only those set, so a
        call without them looks as it always has."""
        return client.opt_flags(self.allow_unmeasured, self.yesno)


def _refusal(problems: list[str]) -> str:
    from .. import library

    return library.refusal(problems)
