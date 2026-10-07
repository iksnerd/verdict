import math

from fastapi.testclient import TestClient

from verdict.api import create_app
from verdict.backend import UniformBackend

BODY = {
    "state": {"message": "I was charged twice"},
    "questions": {
        "intent": {"type": "choice", "instructions": "?", "criteria": {"refund": "", "info": "", "other": ""}},
        "frustration": {"type": "score", "instructions": "?", "criteria": ["calm", "annoyed", "angry", "furious"]},
        "urgent": {"type": "noul", "instructions": "?"},
    },
}


def client():
    return TestClient(create_app(UniformBackend()))


def test_healthz():
    assert client().get("/healthz").json() == {"status": "ok", "backend": "uniform"}


def test_decide_returns_one_answer_per_question():
    r = client().post("/v1/decide", json=BODY)
    assert r.status_code == 200
    answers = r.json()["answers"]
    assert set(answers) == set(BODY["questions"])
    assert answers["intent"]["type"] == "choice"
    assert answers["frustration"]["type"] == "score"
    assert answers["urgent"]["type"] == "noul"


def test_probabilities_are_normalised():
    answers = client().post("/v1/decide", json=BODY).json()["answers"]
    for qid in ("intent", "frustration"):
        assert math.isclose(sum(answers[qid]["probabilities"].values()), 1.0, abs_tol=1e-3)
    assert 0.0 <= answers["urgent"]["noul"] <= 1.0


def test_uniform_backend_is_maximally_unsure():
    answers = client().post("/v1/decide", json=BODY).json()["answers"]
    assert answers["intent"]["confidence"] == 0.0
    assert answers["urgent"]["noul"] == 0.5


def test_invalid_request_is_422():
    assert client().post("/v1/decide", json={"state": "x", "questions": {}}).status_code == 422


def test_route_endpoint_uses_the_default_branch_when_there_is_no_model():
    """The safety property, end to end over HTTP: a uniform backend cannot pick `small`."""
    from verdict.router import BIG

    client = TestClient(create_app())
    r = client.post("/v1/route", json={"prompt": "what does chmod 755 mean"})
    assert r.status_code == 200
    body = r.json()
    assert body["branch"] == BIG
    assert body["model"] == "uniform"
    assert "argv" not in body, "a decision, not a command to run"
    assert "default" in body["reason"]


def test_route_endpoint_rejects_a_missing_prompt():
    client = TestClient(create_app())
    assert client.post("/v1/route", json={}).status_code == 422


def test_batch_route_preserves_order_and_indexes_results():
    from verdict.router import BIG

    client = TestClient(create_app())
    prompts = ["one", "two", "three"]
    r = client.post("/v1/route/batch", json={"prompts": prompts})
    assert r.status_code == 200
    body = r.json()
    assert [item["index"] for item in body["results"]] == [0, 1, 2]
    assert len(body["results"]) == len(prompts)
    # uniform backend: every prompt takes the default branch, and says why
    assert {item["branch"] for item in body["results"]} == {BIG}


def test_batch_route_rejects_an_empty_list():
    """An empty batch is a caller bug, not a zero-length success."""
    assert TestClient(create_app()).post("/v1/route/batch", json={"prompts": []}).status_code == 422


def test_batch_route_caps_the_request_size():
    """Each prompt is its own forward pass, so an uncapped batch hangs the server for minutes."""
    from verdict.api import MAX_BATCH

    client = TestClient(create_app())
    assert client.post("/v1/route/batch", json={"prompts": ["x"] * (MAX_BATCH + 1)}).status_code == 422
    assert client.post("/v1/route/batch", json={"prompts": ["x"] * MAX_BATCH}).status_code == 200


def test_batch_and_single_agree_on_the_same_prompt():
    """Two endpoints, one switch: they must not drift apart."""
    client = TestClient(create_app())
    single = client.post("/v1/route", json={"prompt": "what is 17% of 340"}).json()
    batch = client.post("/v1/route/batch", json={"prompts": ["what is 17% of 340"]}).json()
    item = batch["results"][0]
    assert item["branch"] == single["branch"]
    assert item["reason"] == single["reason"]
    assert item["scores"] == single["scores"]


def _refused(question="How dangerous is this ticket?", state=None):
    bank = {"priority": {"type": "noul", "instructions": question}}
    return client().post("/v1/decide", json={"state": state or {"message": "hi"}, "questions": bank})


def test_a_refusal_names_the_http_opt_out_not_the_cli_flag():
    r = _refused()
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "?allow_unmeasured=true" in detail
    assert "--allow-unmeasured" not in detail


def test_a_refusal_carries_a_stable_code_beside_detail():
    body = _refused().json()
    assert body["error"]["code"] == "refused"
    assert body["error"]["message"] == body["detail"]
    assert "'dangerous'" in body["detail"]


def test_a_missing_field_refusal_also_has_the_code_and_http_opt_out():
    body = _refused("Is `message` about money?", state={"text": "x"}).json()
    assert body["error"]["code"] == "refused"
    assert "?allow_unmeasured=true" in body["detail"]


def test_a_malformed_request_keeps_fastapis_shape():
    r = client().post("/v1/decide", json={"state": "x", "questions": {}})
    assert r.status_code == 422
    assert "error" not in r.json()


def test_the_openapi_schema_describes_the_refusal():
    spec = client().get("/openapi.json").json()
    text = spec["paths"]["/v1/decide"]["post"]["responses"]["422"]["description"]
    assert "refused" in text and "allow_unmeasured" in text
