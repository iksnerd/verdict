from __future__ import annotations

from contextlib import contextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from pydantic import BaseModel, field_validator

from .backend import Backend, UniformBackend, UnknownModel
from .capabilities import Capabilities
from .errors import QuestionError
from .router import BIG_SMALL
from .schema import (
    DecideRequest,
    DecideResponse,
    ModelCard,
    ModelList,
    NoulAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    laya_questions,
)


#: Cap on one batch request. Each prompt is a separate forward pass (the port has no cross-state
#: batching), so a request's cost is linear in this and an uncapped one is an easy way to hang the
#: server for minutes on a single call.
MAX_BATCH = 512


class RouteRequest(BaseModel):
    prompt: str


class RouteResponse(BaseModel):
    model: str
    branch: str
    reason: str
    scores: dict[str, float]


class BatchRouteRequest(BaseModel):
    prompts: list[str]

    @field_validator("prompts")
    @classmethod
    def _within_limits(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("prompts must not be empty")
        if len(v) > MAX_BATCH:
            raise ValueError(f"at most {MAX_BATCH} prompts per request, got {len(v)}")
        return v


class BatchRouteItem(BaseModel):
    index: int
    branch: str
    reason: str
    scores: dict[str, float]


class BatchRouteResponse(BaseModel):
    model: str
    results: list[BatchRouteItem]


#: Explicit compatibility aliases for local backends only. A proxy forwards every model name.
#: Unknown local names must reach engine_for(), where a typo is refused rather than substituted.
_LOCAL_ALIASES = {"jev-latest", "jev-preview", "jev-1.13", "jev-1.13.0"}


def to_decide_request(body: dict | SystemOneRequest, backend: Backend | None = None) -> DecideRequest:
    """A Jev `/v1/systemone` request as ours."""
    req = body if isinstance(body, SystemOneRequest) else SystemOneRequest.model_validate(body)
    return DecideRequest(state=req.state, questions=req.questions,
                         model=_model_name(req.model, backend))


def _model_name(name: str | None, backend: Backend | None) -> str | None:
    if getattr(backend, "preserve_model_names", False):
        return name
    else:
        known = getattr(backend, "known_models", lambda: set())()
        return None if name in _LOCAL_ALIASES and name not in known else name


def _release_date(backend: Backend) -> str:
    """The checkpoint's date on disk, or today when it is not a local path (a Hub id, uniform)."""
    import datetime
    from pathlib import Path

    path = Path(str(getattr(backend, "model_id", "")))
    stamp = path.stat().st_mtime if path.exists() else datetime.datetime.now().timestamp()
    return datetime.date.fromtimestamp(stamp).isoformat()


class Refused(Exception):
    """A question the server will not ask. Answered as a 422 whose body keeps FastAPI's `detail`
    and adds the CLI's `{"error": {"code", "message"}}`, so a caller can tell a refusal from the
    other 422s (a malformed request has a `detail` list and no `error`)."""


@contextmanager
def _backend_errors():
    """One mapping of backend failures to HTTP, for every endpoint that asks the backend."""
    try:
        yield
    except UnknownModel as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except QuestionError as exc:
        raise Refused(str(exc)) from exc
    except ValueError as exc:
        # A SystemOneBackend's upstream (Kev, Von, another verdict) didn't answer: we are a
        # proxy here, so its failure is a bad gateway, not our own 500 with a traceback.
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def create_app(backend: Backend | None = None) -> FastAPI:
    """A decision API: every endpoint answers, none of them runs anything."""
    backend = backend or UniformBackend()
    from importlib.metadata import PackageNotFoundError, version

    try:
        package_version = version("verdict")
    except PackageNotFoundError:
        package_version = "0.0.0"
    app = FastAPI(title="verdict", version=package_version)

    @app.exception_handler(Refused)
    def _refused(_request, exc: Refused) -> JSONResponse:
        message = str(exc)
        return JSONResponse(status_code=422, content={
            "detail": message, "error": {"code": "refused", "message": message}})

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "backend": backend.name}

    @app.post("/v1/decide", response_model=DecideResponse, response_model_exclude={"usage"},
              response_model_exclude_none=True,
              responses={422: {"description": (
                  'A question verdict will not ask: body {"detail": ..., "error": {"code": '
                  '"refused", "message": ...}}. It names the shape and how to reword it; '
                  "?allow_unmeasured=true asks anyway. A malformed request is also 422, with a "
                  "list `detail` and no `error`.")}})
    def decide(request: DecideRequest, allow_unmeasured: bool = False,
               yesno: bool = False) -> DecideResponse:
        """Answers as `verdict decide` does: a question about a field the state lacks, or shaped
        like ones that measured at chance, is a 422 unless `allow_unmeasured=true`, and a new
        yes/no is asked as a no/yes choice unless `yesno=true` (FINDINGS §38)."""
        with _backend_errors():
            return _prepared(request, allow_unmeasured=allow_unmeasured, yesno=yesno)

    def _prepared(request: DecideRequest, *, allow_unmeasured: bool, yesno: bool
                  ) -> DecideResponse:
        """`library.prepare` around `backend.decide`, the one path both endpoints share. A proxy
        backend forwards the request untouched: those findings are about laya, not its upstream."""
        if getattr(backend, "preserve_model_names", False):
            return backend.decide(request)
        from . import library

        questions, rewritten, _ = library.prepare(
            request.state, laya_questions(request.questions),
            model=request.model or getattr(backend, "model_id", backend.name),
            allow_unmeasured=allow_unmeasured, yesno=yesno, opt_out=library.HTTP_OPT_OUT)
        out = backend.decide(DecideRequest(state=request.state, questions=questions,
                                           model=request.model))
        if not rewritten:
            return out
        dumped = library.as_yesno({qid: a.model_dump() for qid, a in out.answers.items()},
                                  rewritten)
        answers = {qid: (NoulAnswer(noul=dumped[qid]["noul"],
                                    confidence=dumped[qid].get("confidence"))
                         if qid in rewritten else a)
                   for qid, a in out.answers.items()}
        return DecideResponse(model=out.model, answers=answers, usage=out.usage)

    @app.post("/v1/systemone", response_model=SystemOneResponse, response_model_exclude_none=True)
    def systemone(request: SystemOneRequest) -> SystemOneResponse:
        """TypeSafe's Jev protocol, so its SDKs and tools can point here with
        `TYPESAFE_BASE_URL`. Same answers as `/v1/decide`, plus `usage`. The bearer key the SDK
        insists on is ignored: the server binds to localhost."""
        with _backend_errors():
            # The same rewrite as /v1/decide, so the two agree; no refusals, since Jev's protocol
            # has no way to ask for the opt-out and its SDKs expect an answer.
            out = _prepared(to_decide_request(request, backend), allow_unmeasured=True,
                            yesno=False)
        return SystemOneResponse(model=out.model, answers=out.answers,
                                 usage=out.usage or Usage(input_tokens=0))

    @app.get("/v1/models", response_model=ModelList)
    def models() -> ModelList:
        if hasattr(backend, "models"):
            #: A proxied backend (SystemOneBackend) has its own real model list; the cards below
            #: describe laya/Jev's fixed vocabulary and would misdescribe whatever this actually
            #: forwards to.
            return ModelList(models=[ModelCard(**card) for card in backend.models()])
        date = _release_date(backend)
        cards = [
            ModelCard(name="jev-latest", release_date=date,
                      description=f"Alias for the served checkpoint, {backend.name}. "
                                  "Compatibility alias; does not load hosted Jev."),
            ModelCard(name="english", release_date=date,
                      description=f"The served checkpoint, {backend.name}."),
        ]
        known = getattr(backend, "known_models", lambda: set())()
        if "multilingual" in known:
            cards.append(ModelCard(name="multilingual", release_date=date,
                                   description=f"laya's multilingual checkpoint, "
                                               f"{backend.multilingual_id}; loaded on first use."))
        for name in sorted(known - {"multilingual"}):
            cards.append(ModelCard(name=name, release_date=date,
                                   description=f"{backend.extra_checkpoints[name]}; "
                                               "loaded on first use. [model.extra] in verdict.toml."))
        return ModelList(models=cards)

    @app.get("/v1/capabilities", response_model=Capabilities)
    def capabilities(model: str | None = None) -> Capabilities:
        """Verdict extension; model cards stay compatible with TypeSafe's SDKs."""
        try:
            if hasattr(backend, "capabilities"):
                return backend.capabilities(_model_name(model, backend))
            return Capabilities(family="uniform" if backend.name == "uniform" else "unknown",
                                model=backend.name)
        except UnknownModel as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _answers_for(prompt: str):
        """Clip, then ask the backend. Clipping here keeps the HTTP path the one the thresholds
        were fitted on; the CLI clips because it has a tokenizer loaded, and before this the
        server did not, so the two front doors could disagree on a long prompt. A backend with no
        engine (uniform) has no tokenizer and needs none."""
        engine = getattr(backend, "engine", None)
        if engine is not None:
            prompt = engine.clip(prompt)
        return backend.decide(DecideRequest(state=prompt, questions=BIG_SMALL.questions)).answers

    @app.post("/v1/route", response_model=RouteResponse)
    def route(request: RouteRequest) -> RouteResponse:
        """Big model or small one, as a `Switch` (see `switch.py`).

        The prompt is clipped to `PROMPT_TOKEN_BUDGET` tokens, the same as the CLI does, so both
        front doors reproduce the FINDINGS §15 numbers. On the `uniform` backend every prompt
        routes to the default branch, which is the point of having a default.
        """
        # Clip here too, so the HTTP path is the one the thresholds were fitted on. The CLI
        # clips because it has a tokenizer loaded; before this the server did not, so the two
        # front doors could disagree on a long prompt. A backend with no engine (uniform) has
        # no tokenizer and needs none.
        with _backend_errors():
            branch = BIG_SMALL.decide(_answers_for(request.prompt))
        return RouteResponse(
            model=backend.name, branch=branch.name, reason=branch.reason,
            scores=branch.scores,
        )

    @app.post("/v1/route/batch", response_model=BatchRouteResponse)
    def route_batch(request: BatchRouteRequest) -> BatchRouteResponse:
        """Route many prompts over one connection, in order.

        There is no cross-state batching underneath: each prompt is its own forward pass, so this
        saves the per-call overhead rather than the inference. That overhead is most of the cost
        for a caller doing many, which is the case this exists for: a client spawning one process
        per prompt pays about 2.1 s each, and one that asks a warm server pays about 70 ms each,
        of which 35 ms is the round trip.

        Results carry `index` so a caller can pair them up without relying on ordering, though
        ordering is preserved.
        """
        results = []
        for i, prompt in enumerate(request.prompts):
            with _backend_errors():
                branch = BIG_SMALL.decide(_answers_for(prompt))
            results.append(BatchRouteItem(
                index=i, branch=branch.name, reason=branch.reason,
                scores=branch.scores,
            ))
        return BatchRouteResponse(model=backend.name, results=results)

    return app


app = create_app()
