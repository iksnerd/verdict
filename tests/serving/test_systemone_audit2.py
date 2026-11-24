"""Audit 2: a closed SystemOneClient is an upstream failure, not an httpx RuntimeError."""
from __future__ import annotations

import httpx
import pytest

from verdict.systemone import SystemOneClient, UpstreamError


def test_a_call_after_close_raises_upstream_error():
    c = SystemOneClient("http://127.0.0.1:1", "m", "k",
                        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    c.close()
    with pytest.raises(UpstreamError, match="closed"):
        c.models()
    with pytest.raises(UpstreamError, match="closed"):
        c({"a": 1}, {"q": {"type": "noul", "instructions": "x"}})
