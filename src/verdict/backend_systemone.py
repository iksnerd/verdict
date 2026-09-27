"""A typed backend for Jev or any SystemOne-compatible service.

The protocol is shared; model capabilities and confidence semantics are not. Requests are
forwarded unchanged, with no local clipping or question rewriting. Own its lifetime with a
context manager or close(). The API application does not close a caller-owned backend.
"""
from __future__ import annotations

from typing import Any

from .schema import DecideRequest, DecideResponse, Usage
from .systemone import SystemOneClient, UpstreamError


class SystemOneBackend:
    """model is the default; a request's model overrides it, including version-pinned Jev IDs.

    profile='jev' opts a custom gateway into TypeSafe's documented limits. Only the official
    api.typesafe.ai hostname is detected automatically; other providers retain unknown limits.
    """

    preserve_model_names = True

    def __init__(self, base_url: str, model: str = "jev-latest", key: str = "local", **ask_kwargs: Any):
        self.base_url, self.model = base_url, model
        self.name = f"systemone:{model}@{base_url}"
        self._ask = SystemOneClient(base_url, model, key, **ask_kwargs)

    def capabilities(self, model: str | None = None):
        return self._ask.capabilities(model)

    def models(self) -> list[dict[str, Any]]:
        """Forward the upstream list. Unavailable or malformed listings degrade to an empty list."""
        try:
            return self._ask.models()
        except UpstreamError:
            return []

    def decide(self, request: DecideRequest) -> DecideResponse:
        questions = {qid: q.model_dump(exclude_none=True) for qid, q in request.questions.items()}
        raw = self._ask(request.state, questions, model=request.model)
        result = DecideResponse.model_validate(raw)
        # Missing counts are unknown, not zero. Local Laya computes its own explicit zeros.
        usage = raw.get("usage") or {}
        result.usage = Usage(input_tokens=usage.get("input_tokens"),
                             output_tokens=usage.get("output_tokens"))
        return result

    def close(self) -> None:
        self._ask.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
