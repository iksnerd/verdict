from __future__ import annotations

import pytest

pytest.importorskip("httpx2")
sdk = pytest.importorskip("typesafe_sdk")

from verdict.backends.jev import JevBackend  # noqa: E402
from verdict.schema import DecideRequest  # noqa: E402
from verdict.systemone import UpstreamError  # noqa: E402
from fakes import NOUL_BANK  # noqa: E402


def test_models_none_on_failure_and_empty_when_none():
    b = JevBackend(api_key="k", base_url="http://x.example")

    class Boom:
        def list(self):
            raise sdk.TypeSafeError("down")

    class Empty:
        def list(self):
            class R:
                models = []
            return R()

    b._client.models = Boom()
    assert b.models() is None
    b._client.models = Empty()
    assert b.models() == []


def test_decide_after_close_says_closed():
    b = JevBackend(api_key="k", base_url="http://x.example")
    b.close()
    with pytest.raises(RuntimeError, match="closed"):
        b.decide(DecideRequest(state="x", questions=NOUL_BANK))
