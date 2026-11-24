"""Audit 3: a failure on prompt N of /v1/route/batch keeps the N-1 results already computed. No model."""
from __future__ import annotations

from fastapi.testclient import TestClient

from fakes import FailsAt as FailsOnThird
from verdict.api import create_app


def _post(backend, n=5):
    return TestClient(create_app(backend)).post(
        "/v1/route/batch", json={"prompts": [f"prompt {i}" for i in range(n)]})


def test_a_failure_keeps_the_results_computed_before_it():
    backend = FailsOnThird()
    r = _post(backend)
    assert r.status_code == 207
    results = r.json()["results"]
    assert [x["index"] for x in results] == [0, 1, 2, 3, 4]
    assert all("branch" in x and "error" not in x for x in results[:2])
    assert "upstream went away" in results[2]["error"] and "branch" not in results[2]
    assert all("not attempted" in x["error"] for x in results[3:])
    assert backend.calls == 3          # nothing after the failure was asked


def test_a_clean_batch_keeps_its_shape():
    r = _post(FailsOnThird(fail_at=99))
    assert r.status_code == 200
    assert all(set(x) == {"index", "branch", "reason", "scores"} for x in r.json()["results"])


def test_a_failure_on_the_first_prompt_is_still_the_plain_error():
    assert _post(FailsOnThird(fail_at=0)).status_code == 502
