"""Hosted Jev through TypeSafe's official SDK (`pip install 'verdict[jev]'`).

The SDK owns the transport: authentication, retries with backoff and `Retry-After`, timeouts,
HTTP/2 if asked, and request ids on errors. This module only translates: verdict's questions into
the SDK's typed ones, its typed answers back into verdict's, and its exceptions into verdict's two
(`QuestionError` for a question the service rejects, `UpstreamError` for everything else).

For any other /v1/systemone server (Kev, another verdict) use `SystemOneBackend`, which does not
need the SDK. A request goes upstream as given: laya's clipping and rewrites do not apply.
"""
from __future__ import annotations

from typing import Any

try:
    import typesafe_sdk as sdk
except ImportError as exc:  # pragma: no cover - the message is the point
    raise ImportError("JevBackend needs TypeSafe's SDK: pip install 'verdict[jev]'") from exc

from .capabilities import Capabilities, jev_capabilities
from .errors import QuestionError
from .schema import ChoiceQuestion, DecideRequest, DecideResponse, NoulQuestion, ScoreQuestion
from .systemone import UpstreamError, validate_response

DEFAULT_MODEL = "jev-latest"
#: Jev rounds each probability to two decimals and adjusts them to sum to 1 (FINDINGS §48).
JEV_PRECISION = 0.01


def sdk_question(question: Any) -> sdk.Question:
    """One of verdict's questions as the SDK's typed one."""
    fields = question.model_dump(exclude_none=True)
    fields.pop("type")
    if isinstance(question, ChoiceQuestion):
        return sdk.Choice(**fields)
    if isinstance(question, ScoreQuestion):
        return sdk.Score(**fields)
    if isinstance(question, NoulQuestion):
        return sdk.Noul(**fields)
    raise QuestionError(f"unknown question type {type(question).__name__}")


def _failure(exc: sdk.TypeSafeError) -> Exception:
    """Verdict's error for an SDK one. Names the status and request id and nothing from the body:
    the service can echo the state, which can be private (the same rule as `systemone.py`)."""
    if isinstance(exc, sdk.TypeSafeAPIError):
        where = f" (request_id={exc.request_id})" if exc.request_id else ""
        what = f"TypeSafe answered {exc.status}{where}"
        if isinstance(exc, sdk.TypeSafeUnprocessableEntityError):
            return QuestionError(f"{what}: it rejected the question as written")
        if isinstance(exc, (sdk.TypeSafeAuthenticationError, sdk.TypeSafePermissionDeniedError)):
            return UpstreamError(f"{what}: check $TYPESAFE_API_KEY")
        return UpstreamError(what)
    if isinstance(exc, sdk.TypeSafeAPITimeoutError):
        return UpstreamError("TypeSafe did not answer in time")
    if isinstance(exc, sdk.TypeSafeAPIConnectionError):
        return UpstreamError("could not reach TypeSafe")
    return UpstreamError("TypeSafe's SDK could not complete the request")


class JevBackend:
    """`model` is the default; a request's model overrides it, including version-pinned ids.

    `api_key` and `base_url` fall back to the SDK's own `$TYPESAFE_API_KEY` and
    `$TYPESAFE_BASE_URL`. `retry` is the SDK's `RetryPolicy`. Own its lifetime with a context
    manager or `close()`; `create_app` does not close a backend it was given.
    """

    preserve_model_names = True

    def __init__(self, model: str = DEFAULT_MODEL, *, api_key: str | None = None,
                 base_url: str | None = None, retry: sdk.RetryPolicy | None = None,
                 **client_options: Any):
        self.model = model
        self._client = sdk.TypeSafeClient(api_key=api_key, base_url=base_url, model=model,
                                          retry=retry, **client_options)
        # Where the SDK will actually send: an explicit argument, else $TYPESAFE_BASE_URL, else the
        # default. The name appears in logs, /healthz and scorecards, so it must not claim a
        # different host than the states go to. `_config` is the SDK's own resolution; if a future
        # version renames it, say the default rather than guess.
        resolved = getattr(getattr(self._client, "_config", None), "base_url", None)
        self.base_url = str(resolved or base_url or "https://api.typesafe.ai").rstrip("/")
        self.name = f"jev:{model}@{self.base_url}"

    def capabilities(self, model: str | None = None) -> Capabilities:
        return jev_capabilities(model or self.model)

    def models(self) -> list[dict[str, Any]]:
        """What the service lists; unavailable or malformed listings degrade to an empty list."""
        try:
            listed = self._client.models.list()
        except sdk.TypeSafeError:
            return []
        return [m.model_dump(mode="json", exclude_none=True) for m in listed.models]

    def decide(self, request: DecideRequest) -> DecideResponse:
        model = request.model or self.model
        self.capabilities(model).validate_request(request)
        questions = {qid: sdk_question(q) for qid, q in request.questions.items()}
        try:
            result = self._client.system_one(request.state, questions, model=model)
        except sdk.TypeSafeAPIResponseValidationError as exc:
            raise UpstreamError("upstream returned an invalid SystemOne response") from exc
        except sdk.TypeSafeError as exc:
            raise _failure(exc) from exc
        raw = {"model": result.model,
               "answers": {qid: a.model_dump(mode="json") for qid, a in result.answers.items()},
               "usage": result.usage.model_dump(mode="json")}
        return validate_response(raw, request, precision=JEV_PRECISION)

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class JevAsker:
    """The shape `bench` and `examples` call: `(state, questions, model=None)` returning the raw
    answers dict, with `close()`. Hosted Jev through the SDK, so a bench run gets its retries and
    request ids rather than a second implementation of them."""

    def __init__(self, base_url: str, model: str, key: str, *, attempts: int = 5,
                 backoff: float = 1.0, timeout: float = 30.0, transport: Any = None):
        retry = sdk.RetryPolicy(max_retries=max(0, attempts - 1), backoff_initial=backoff,
                                backoff_max=max(backoff, 5.0) if backoff else 0)
        self.backend = JevBackend(model, api_key=key, base_url=base_url, retry=retry,
                                  timeout=timeout, transport=transport)

    def __call__(self, state: Any, questions: dict, *, model: str | None = None) -> dict:
        request = DecideRequest(state=state, questions=questions, model=model or self.backend.model)
        return self.backend.decide(request).model_dump(exclude_none=True)

    def close(self) -> None:
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
