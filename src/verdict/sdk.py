"""verdict as a Python library, shaped like TypeSafe's SDK.

    from verdict import Verdict, Noul, Choice, Score

    with Verdict() as v:                       # the server `verdict serve` started
        r = v.decide({"message": "I was charged twice"}, {
            "refund": Noul(instructions="Is a refund requested?"),
            "dept": Choice(instructions="Which team?", criteria={"billing": "refunds", "tech": "bugs"}),
        })
        r.nouls["refund"].noul, r.choices["dept"].choice

The same pieces as the SDK on purpose: typed question classes that fail at construction, a result
grouped into `.nouls`, `.choices` and `.scores`, a sync `with` client, and one error type with a
`.status`. The transport is the caller's explicit choice, never a surprise: a running server by
default (`url=`), or a `Backend` in this process (`backend=`: `MlxBackend`, `JevBackend`). Nothing
loads a model unless you build one. The refusals are the CLI's and the HTTP API's, through the same
`library.ask`, so one question gets one answer to "may I ask this" everywhere.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import cached_property
from typing import Any, Literal, Mapping, Optional, Union

from pydantic import ValidationError

from . import client as _http
from . import library
from .backend import UnknownModel
from .capabilities import Capabilities
from .errors import QuestionError
from .systemone import UpstreamError
from .schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecideRequest,
    DecideResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
    _readable,
)


class Noul(NoulQuestion):
    """A yes/no question; `criteria` may describe the `true` and `false` sides."""

    type: Literal["noul"] = "noul"


class Choice(ChoiceQuestion):
    """One of several named options; `criteria` maps each option to what it means."""

    type: Literal["choice"] = "choice"


class Score(ScoreQuestion):
    """A position on an ordered rubric; `criteria` lists the levels, lowest first."""

    type: Literal["score"] = "score"


Question = Union[Noul, Choice, Score, Mapping[str, Any]]


class VerdictError(Exception):
    """Every failure of a call, with the HTTP-style `status` it maps to (None when nothing
    answered) and a stable `code` to branch on: `refused` (reword the question, or pass
    `allow_unmeasured=True`), `no_server`, `server_timeout`, `server_error`, `upstream`,
    `unknown_model`, `invalid_question`."""

    def __init__(self, message: str, *, status: Optional[int] = None, code: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.code = code


class Result:
    """Answers keyed by question name, grouped by type, with the model and token usage."""

    def __init__(self, response: DecideResponse):
        #: The wire response this wraps.
        self.raw = response

    @property
    def model(self) -> str:
        return self.raw.model

    @property
    def usage(self) -> Optional[Usage]:
        return self.raw.usage

    @property
    def answers(self) -> dict[str, Any]:
        return dict(self.raw.answers)

    @cached_property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {k: a for k, a in self.raw.answers.items() if isinstance(a, NoulAnswer)}

    @cached_property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: a for k, a in self.raw.answers.items() if isinstance(a, ChoiceAnswer)}

    @cached_property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {k: a for k, a in self.raw.answers.items() if isinstance(a, ScoreAnswer)}


@contextmanager
def _served_errors():
    """The HTTP client's failures as `VerdictError`, for every call that goes to a server."""
    try:
        yield
    except _http.ServerError as exc:
        # The client words the opt-out as the CLI flag; this caller passes a keyword.
        message = str(exc).replace(library.CLI_OPT_OUT, "allow_unmeasured=True")
        raise VerdictError(message, status=exc.status, code=exc.code or "server_error") from exc
    except _http.ServerTimeout as exc:
        raise VerdictError(str(exc), code="server_timeout") from exc
    except _http.NoServer as exc:
        raise VerdictError(f"{exc}; start one with `verdict serve`, or pass backend=",
                           code="no_server") from exc


class Verdict:
    """A client. Give it `backend=` to answer in this process, or `url=` (default: the configured
    server) to ask a running `verdict serve`. The caller owns a backend it passes in, as with
    `create_app`; closing the client does not close it."""

    def __init__(self, *, url: Optional[str] = None, backend: Any = None,
                 allow_unmeasured: bool = False, yesno: bool = False):
        if url is not None and backend is not None:
            raise ValueError("url and backend are mutually exclusive.")
        self._url, self._backend = url, backend
        self.allow_unmeasured, self.yesno = allow_unmeasured, yesno

    def decide(self, state: Any, questions: Mapping[str, Question], *,
               model: Optional[str] = None) -> Result:
        """Answer `questions` about `state`. Raises `VerdictError`."""
        try:
            request = DecideRequest(state=state, model=model, questions={
                qid: (q if isinstance(q, (NoulQuestion, ChoiceQuestion, ScoreQuestion)) else dict(q))
                for qid, q in questions.items()})
        except (ValidationError, QuestionError) as exc:
            message = _readable(exc) if isinstance(exc, ValidationError) else str(exc)
            raise VerdictError(f"invalid questions: {message}", status=422,
                               code="invalid_question") from exc
        if self._backend is not None:
            return Result(self._local(request))
        return Result(self._served(request))

    def _local(self, request: DecideRequest) -> DecideResponse:
        try:
            return library.ask(self._backend, request, allow_unmeasured=self.allow_unmeasured,
                               yesno=self.yesno, opt_out="allow_unmeasured=True")
        except QuestionError as exc:
            raise VerdictError(str(exc), status=422, code="refused") from exc
        except UnknownModel as exc:
            raise VerdictError(str(exc), status=400, code="unknown_model") from exc
        except UpstreamError as exc:  # a remote model that did not answer; anything else is a bug
            raise VerdictError(str(exc), status=502, code="upstream") from exc

    def _served(self, request: DecideRequest) -> DecideResponse:
        flags = {k: True for k, on in (("allow_unmeasured", self.allow_unmeasured),
                                       ("yesno", self.yesno)) if on}
        questions = {qid: q.model_dump(exclude_none=True) for qid, q in request.questions.items()}
        with _served_errors():
            raw = _http.decide(request.state, questions, url=self._url, model=request.model, **flags)
        return DecideResponse.model_validate(raw)

    def classify(self, state: Any, tree: dict, *, beam: int = 2, model: Optional[str] = None):
        """Classify `state` into one leaf of a two-level `tree`: the way past laya's 20-option limit
        (`verdict classify`, FINDINGS §50). Returns a `hierarchy.Classification`: `.label`, `.path`,
        `.probability`, `.candidates`, `.passes`. A level past 20 raises `VerdictError` (`refused`)."""
        from . import hierarchy

        def ask(st: Any, questions: dict) -> dict:
            answers = self.decide(st, questions, model=model).answers
            return {"answers": {qid: a.model_dump() for qid, a in answers.items()}}

        try:
            return hierarchy.classify(ask, state, tree, beam=beam)
        except QuestionError as exc:
            raise VerdictError(str(exc), status=422, code="refused") from exc

    def models(self) -> list[dict[str, Any]]:
        """What can answer: the server's list, or the backend's (its own, if it has one, else the
        checkpoint it serves and the names it will accept)."""
        if self._backend is None:
            with _served_errors():
                return _http.models(self._url)
        if hasattr(self._backend, "models"):
            return list(self._backend.models())
        known = getattr(self._backend, "known_models", lambda: set())()
        return [{"name": self._backend.name}, *({"name": n} for n in sorted(known))]

    def capabilities(self, model: Optional[str] = None) -> Capabilities:
        """The backend's limits and confidence semantics: family (`laya`, `jev`, ...), how many
        options and score levels it takes, and whether a yes/no reports a confidence."""
        if self._backend is None:
            with _served_errors():
                return Capabilities.model_validate(_http.capabilities(self._url, model))
        if hasattr(self._backend, "capabilities"):
            return self._backend.capabilities(model)
        return Capabilities(family="uniform" if self._backend.name == "uniform" else "unknown",
                            model=self._backend.name)

    def close(self) -> None:
        """Nothing to release: there is no connection pool, and a passed-in backend is the
        caller's. Kept so a `with` block reads the same as the SDK's."""

    def __enter__(self) -> "Verdict":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
