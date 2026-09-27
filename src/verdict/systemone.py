"""Reusable SystemOne HTTP transport with explicit lifetime and validated responses."""
from __future__ import annotations

import math
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from .capabilities import Capabilities, jev_capabilities
from .schema import DecideRequest, DecideResponse


class UpstreamError(ValueError):
    """An upstream server failed or returned an invalid decision; never a usable answer."""


def validate_response(raw: Any, request: DecideRequest) -> DecideResponse:
    """Check semantic consistency as well as types, without rewriting provider values."""
    try:
        result = DecideResponse.model_validate(raw, strict=True)
        if not result.model or set(result.answers) != set(request.questions):
            raise ValueError("model and exactly one answer per question are required")
        for qid, q in request.questions.items():
            a = result.answers[qid]
            if raw["answers"][qid].get("type") != q.type or a.type != q.type:
                raise ValueError(f"{qid}: answer type does not match question")
            confidence = a.confidence
            if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
                raise ValueError(f"{qid}: invalid confidence")
            if a.type == "noul":
                if not math.isfinite(a.noul) or not 0 <= a.noul <= 1:
                    raise ValueError(f"{qid}: invalid yes/no probability")
                continue
            p = a.probabilities
            if set(p) != set(q.option_keys()):
                raise ValueError(f"{qid}: probability keys do not match criteria")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()):
                raise ValueError(f"{qid}: invalid probabilities")
            # Four-decimal Laya distributions can accumulate rounding error over many options.
            tolerance = max(1e-6, len(p) * 0.000051)
            if abs(sum(p.values()) - 1) > tolerance:
                raise ValueError(f"{qid}: probabilities must sum to 1")
            if a.type == "choice":
                if a.choice not in p or p[a.choice] + .0001 < max(p.values()):
                    raise ValueError(f"{qid}: choice is not a highest-probability option")
            else:
                if set(a.legend) != set(p) or any(a.legend[str(i)] != v for i, v in enumerate(q.criteria)):
                    raise ValueError(f"{qid}: score legend does not match criteria")
                if not math.isfinite(a.score) or not 0 <= a.score <= len(p) - 1:
                    raise ValueError(f"{qid}: score outside rubric")
                expected = sum(int(k) * v for k, v in p.items())
                if abs(a.score - expected) > max(.0001, len(p) ** 2 * .000051):
                    raise ValueError(f"{qid}: score does not match probability-weighted levels")
        if result.usage:
            for value in (result.usage.input_tokens, result.usage.output_tokens):
                if value is not None and value < 0:
                    raise ValueError("negative token usage")
        return result
    except (ValidationError, ValueError, TypeError, KeyError) as exc:
        # Do not echo raw upstream bodies: they can include private state or credentials.
        raise UpstreamError("upstream returned an invalid SystemOne response") from exc


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0., seconds) if math.isfinite(seconds) else None


class SystemOneClient:
    """Callable raw-JSON client shared by the SDK backend and benchmarks.

    No clipping, question rewriting, fallback to a different model, or fabricated confidence.
    Use as a context manager, or call close() when finished. profile='jev' is an explicit opt-in
    for a Jev gateway; only api.typesafe.ai is recognized automatically.
    """

    def __init__(self, base_url: str, model: str, key: str, *, attempts: int = 5,
                 backoff: float = 1., timeout: float = 30., transport=None,
                 profile: str | None = None):
        if not isinstance(attempts, int) or attempts < 1:
            raise ValueError("attempts must be a positive integer")
        if not math.isfinite(backoff) or backoff < 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("backoff must be nonnegative and timeout positive, both finite")
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
        self.profile = profile or ("jev" if url.hostname == "api.typesafe.ai" else "systemone")
        if self.profile not in ("jev", "systemone"):
            raise ValueError("profile must be 'jev' or 'systemone'")
        self.base_url, self.model = base_url.rstrip("/"), model
        self.attempts, self.backoff = attempts, backoff
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport,
                                 headers={"Authorization": f"Bearer {key}"})

    def capabilities(self, model: str | None = None) -> Capabilities:
        selected = model if model is not None else self.model
        return (jev_capabilities(selected) if self.profile == "jev"
                else Capabilities(family="systemone", model=selected,
                                  noul_confidence="provider_defined"))

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        for attempt in range(self.attempts):
            response = None
            try:
                response = self.http.request(method, path, **kwargs)
                if response.status_code not in {408, 429, 500, 502, 503, 504, 529}:
                    if response.status_code != 200:
                        raise UpstreamError(f"{self.base_url}{path} answered {response.status_code}")
                    return response
                reason = f"answered {response.status_code}"
            except httpx.TransportError as exc:
                reason = exc.__class__.__name__
            if attempt == self.attempts - 1:
                raise UpstreamError(f"{self.base_url}{path} {reason} after {self.attempts} attempts; "
                                    "it may still be loading or unavailable")
            delay = self.backoff * 2 ** attempt
            if response is not None:
                after = _retry_after(response.headers.get("Retry-After"))
                if after is not None:
                    delay = max(delay, after)
            print(f"verdict: {self.base_url}{path} {reason}; it may still be loading -- "
                  f"retrying ({attempt + 2}/{self.attempts})", file=sys.stderr)
            if delay:
                time.sleep(delay)
        raise AssertionError("unreachable")

    def __call__(self, state, questions, *, model: str | None = None) -> dict[str, Any]:
        selected = self.model if model is None else model
        request = DecideRequest(state=state, questions=questions, model=selected)
        self.capabilities(selected).validate_request(request)
        response = self._request("POST", "/v1/systemone", json=request.model_dump(exclude_none=True))
        try:
            raw = response.json()
        except ValueError as exc:
            raise UpstreamError("upstream returned invalid JSON") from exc
        validate_response(raw, request)
        return raw

    def models(self) -> list[dict[str, Any]]:
        from .schema import ModelList

        response = self._request("GET", "/v1/models")
        try:
            cards = ModelList.model_validate(response.json())
            return [c.model_dump() for c in cards.models]
        except (ValueError, TypeError) as exc:
            raise UpstreamError("upstream returned an invalid model list") from exc

    def close(self) -> None:
        self.http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
