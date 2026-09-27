"""Talk to a running verdict server, so the CLI does not pay to start one.

The measurement this exists for: a one-shot `verdict route` costs about 2.1 s to make a 26 ms
decision. `import transformers` alone is 1277 ms, and it is imported for one reason, to clip the
prompt to 128 tokens, which saves about 50 ms of inference. For a process that exits immediately
that optimisation costs roughly 25x what it saves. In a server it loads once and pays for itself
on every later call, which is the whole argument for having one.

So the CLI asks a server first and only falls back to loading the model itself.

`urllib.request` rather than `httpx`, although `httpx` is already a dependency: 13 ms to import
against 48 ms. On a budget where the decision itself is 26 ms that is not a rounding error, and a
single JSON POST needs nothing httpx provides.

Nothing here imports `laya_mlx` or `transformers`, so the server path never pays for them.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

#: Where the CLI looks. Not 8765: other local services use it (a speech engine on the development
#: machine did), and a default that collides with a service someone actually runs means
#: `verdict serve` dies on a bind error while `verdict route` silently talks to a stranger.
DEFAULT_URL = "http://127.0.0.1:8799"

#: Short on purpose. This is a liveness question on loopback, and the fallback (loading the model
#: in-process) costs hundreds of milliseconds, so waiting a long time to discover "no server" makes
#: the miss case worse than having never asked.
CONNECT_TIMEOUT = 0.35


def server_url() -> str:
    return os.environ.get("VERDICT_URL", DEFAULT_URL).rstrip("/")


class NoServer(Exception):
    """No server answered, or the one that did has no model behind it."""


class ServerError(Exception):
    """A verdict server answered and refused the call. Not a reason to fall back: the local model
    would be loaded for a call the server has already said is wrong, and its error would be lost."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"the server answered {status}: {detail}")
        self.status = status
        self.detail = detail


#: Statuses meaning "no such endpoint here": another service holds the port, so fall back.
_NOT_VERDICT = frozenset({404, 405})


#: A batch is one forward pass per prompt, so its wall time scales with the count. The connect
#: failure that matters for fallback (nothing listening) is refused instantly on loopback rather
#: than timing out, so a generous ceiling here does not slow the miss case down.
BATCH_TIMEOUT_BASE = 5.0
BATCH_TIMEOUT_PER_PROMPT = 0.5


def _post(
    base: str, path: str, body: dict[str, Any], timeout: float, expect: frozenset[str]
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # HTTPError is a URLError, so it has to be caught first or a live server's 422 reads as
        # nothing listening.
        if exc.code in _NOT_VERDICT:
            raise NoServer(f"{base} has no {path} ({exc.code})") from exc
        raise ServerError(exc.code, _detail(exc)) from exc
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise NoServer(f"no verdict server at {base} ({exc})") from exc

    if not isinstance(payload, dict) or "model" not in payload or not (expect & payload.keys()):
        # Something answered, but it is not a verdict server. Ports get reused: a collision on
        # 8765 is what moved the default. Trusting any 200 that
        # happens to be JSON would route on a stranger's output.
        raise NoServer(f"{base} answered but does not look like a verdict server")
    if payload.get("model") == "uniform":
        raise NoServer(f"the server at {base} is running the uniform backend, which has no model")
    return payload


def _detail(exc: urllib.error.HTTPError) -> str:
    """FastAPI's `detail`, as text; the raw body when it is not JSON."""
    try:
        body = exc.read().decode(errors="replace")
    except OSError:
        return exc.reason or ""
    try:
        detail = json.loads(body).get("detail", body)
    except (ValueError, AttributeError):
        return body
    return detail if isinstance(detail, str) else json.dumps(detail)


def route_batch(
    prompts: list[str], url: str | None = None, timeout: float | None = None
) -> list[dict[str, Any]]:
    """`POST /v1/route/batch`. One connection for the lot; raises `NoServer` like `route`.

    Returns the `results` list, each carrying `index`, sorted by it. The server preserves order
    already; sorting here means a caller pairing prompts to branches by position is not at the
    mercy of that staying true, and puts the guarantee at the library boundary rather than in
    whichever caller remembered.
    """
    base = (url or server_url()).rstrip("/")
    if timeout is None:
        timeout = BATCH_TIMEOUT_BASE + BATCH_TIMEOUT_PER_PROMPT * len(prompts)
    results = _post(
        base, "/v1/route/batch", {"prompts": prompts}, timeout, frozenset({"results"})
    )["results"]
    return sorted(results, key=lambda r: r["index"])


def decide(
    state: Any, questions: dict[str, Any], url: str | None = None, timeout: float | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """`POST /v1/decide`: arbitrary typed questions, not the big-vs-small switch.

    The general path. Nothing is routed and nothing is dispatched, so this is what a caller uses
    when it wants a bounded answer with a number on it rather than a model to send work to.
    """
    if timeout is None:
        timeout = BATCH_TIMEOUT_BASE + BATCH_TIMEOUT_PER_PROMPT * max(1, len(questions))
    body: dict[str, Any] = {"state": state, "questions": questions}
    if model is not None:
        body["model"] = model
    return _post((url or server_url()).rstrip("/"), "/v1/decide", body, timeout,
                 frozenset({"answers"}))


def capabilities(url: str | None = None, model: str | None = None,
                 timeout: float = CONNECT_TIMEOUT) -> dict[str, Any]:
    """Read Verdict's optional capability extension. None-valued limits mean unknown.

    Older Verdict versions and direct TypeSafe endpoints may not expose it; NoServer here
    means capability discovery is unavailable, not that inference should change backends.
    """
    from urllib.parse import urlencode

    base = (url or server_url()).rstrip("/")
    path = "/v1/capabilities" + ("?" + urlencode({"model": model}) if model is not None else "")
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code in _NOT_VERDICT:
            raise NoServer(f"{base} does not expose capability discovery") from exc
        raise ServerError(exc.code, _detail(exc)) from exc
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise NoServer(f"capability discovery unavailable at {base}") from exc
    if not isinstance(payload, dict) or not {"family", "model"} <= payload.keys():
        raise NoServer(f"{base} returned invalid capability metadata")
    return payload


def route(prompt: str, url: str | None = None, timeout: float = CONNECT_TIMEOUT) -> dict[str, Any]:
    """`POST /v1/route`. Raises `NoServer` when the caller should fall back to in-process.

    A server running the `uniform` backend is treated as no server at all. It answers every
    question with maximum uncertainty, so its route is always the default branch: not a decision,
    just the absence of one. Silently returning that would break the project's one standing rule,
    that a uniform answer must never gate anything. Falling back is right rather than failing,
    because the local checkpoint is usually available even when the served one is not.
    """
    return _post(
        (url or server_url()).rstrip("/"), "/v1/route", {"prompt": prompt}, timeout,
        frozenset({"branch"}),
    )
