"""The client decides whether the CLI pays 2 s to load a model, so its fallback rules are tested
rather than trusted. No server and no model: `urlopen` is replaced."""
from __future__ import annotations

import json
import urllib.error
from io import BytesIO

import pytest

from verdict import client
from verdict.router import BIG, SMALL

PAYLOAD = {
    "model": "laya-mlx:models/verdict-v1-mlx",
    "branch": SMALL,
    "reason": "easy (difficulty 1.20/3) and not sensitive",
    "scores": {"difficulty": 1.2, "sensitive": 0.1},
}


class FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def serve(monkeypatch, payload, captured=None):
    def fake_urlopen(request, timeout=None):
        if captured is not None:
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            captured["method"] = request.method
        return FakeResponse(json.dumps(payload).encode())

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)


def refuse(monkeypatch, exc=None):
    def fake_urlopen(request, timeout=None):
        raise exc or urllib.error.URLError("Connection refused")

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)


def test_a_served_route_comes_back_whole(monkeypatch):
    serve(monkeypatch, PAYLOAD)
    assert client.route("anything") == PAYLOAD


def test_it_posts_json_to_the_route_endpoint(monkeypatch):
    captured: dict = {}
    serve(monkeypatch, PAYLOAD, captured)
    client.route("what is 17% of 340", url="http://example:9999")
    assert captured["url"] == "http://example:9999/v1/route"
    assert captured["method"] == "POST"
    assert captured["body"] == {"prompt": "what is 17% of 340"}


def test_a_trailing_slash_does_not_double_up(monkeypatch):
    captured: dict = {}
    serve(monkeypatch, PAYLOAD, captured)
    client.route("x", url="http://example:9999/")
    assert captured["url"] == "http://example:9999/v1/route"


def test_no_server_raises_rather_than_returning_a_branch(monkeypatch):
    """The caller must be able to tell 'no server' from 'the server said big'."""
    refuse(monkeypatch)
    with pytest.raises(client.NoServer):
        client.route("anything")


@pytest.mark.parametrize(
    "exc",
    [urllib.error.URLError("refused"), OSError("timed out"), ConnectionResetError()],
    ids=["url error", "os error", "reset"],
)
def test_every_transport_failure_is_a_missing_server(monkeypatch, exc):
    refuse(monkeypatch, exc)
    with pytest.raises(client.NoServer):
        client.route("anything")


def test_a_malformed_body_is_a_missing_server_not_a_crash(monkeypatch):
    monkeypatch.setattr(
        client.urllib.request, "urlopen",
        lambda request, timeout=None: FakeResponse(b"<html>not json</html>"),
    )
    with pytest.raises(client.NoServer):
        client.route("anything")


def test_a_uniform_backend_counts_as_no_server(monkeypatch):
    """A uniform answer is the absence of a decision. Returning it would break the one standing
    rule, and the local checkpoint is usually loadable even when the served one is not."""
    serve(monkeypatch, {**PAYLOAD, "model": "uniform", "branch": BIG})
    with pytest.raises(client.NoServer, match="uniform"):
        client.route("anything")


def test_the_timeout_is_short_enough_to_be_worth_asking(monkeypatch):
    """Discovering 'no server' must cost less than the fallback it protects against."""
    captured: dict = {}
    serve(monkeypatch, PAYLOAD, captured)
    client.route("x")
    assert captured["timeout"] == client.CONNECT_TIMEOUT
    assert client.CONNECT_TIMEOUT < 1.0


def test_the_url_comes_from_the_environment_then_the_default(monkeypatch):
    monkeypatch.delenv("VERDICT_URL", raising=False)
    assert client.server_url() == client.DEFAULT_URL
    monkeypatch.setenv("VERDICT_URL", "http://elsewhere:1234/")
    assert client.server_url() == "http://elsewhere:1234"


def test_the_client_pulls_in_nothing_expensive():
    """The whole point is avoiding a 1277 ms transformers import for a 26 ms decision.

    Parsed rather than grepped: the module docstring says the words "import transformers" while
    explaining why it does not do that, and a substring search cannot tell the two apart.
    """
    import ast

    tree = ast.parse(client.__loader__.get_source("verdict.client"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"transformers", "laya_mlx", "torch", "mlx", "httpx"}), imported


def test_capabilities_encodes_model_and_keeps_unknown_values(monkeypatch):
    calls = []
    def respond(url, timeout):
        calls.append(url)
        return FakeResponse(b'{"family":"systemone","model":"a/b","max_choice_options":null}')
    monkeypatch.setattr(client.urllib.request, "urlopen", respond)
    result = client.capabilities("http://example:1234", model="a/b")
    assert calls == ["http://example:1234/v1/capabilities?model=a%2Fb"]
    assert result["max_choice_options"] is None


def test_capabilities_missing_endpoint_is_explicit(monkeypatch):
    refuse(monkeypatch, urllib.error.HTTPError("http://example", 404, "missing", {}, BytesIO()))
    with pytest.raises(client.NoServer, match="capability"):
        client.capabilities()


def test_route_batch_returns_results_in_index_order(monkeypatch):
    """The server preserves order, but the client sorts on `index` anyway: a caller pairing
    prompts to branches by position must not be at the mercy of that staying true."""
    scrambled = {
        "model": PAYLOAD["model"],
        "results": [
            {"index": 2, "branch": BIG, "reason": "c", "scores": {}},
            {"index": 0, "branch": SMALL, "reason": "a", "scores": {}},
            {"index": 1, "branch": BIG, "reason": "b", "scores": {}},
        ],
    }
    serve(monkeypatch, scrambled)
    got = client.route_batch(["a", "b", "c"])
    assert [r["index"] for r in got] == [0, 1, 2]
    assert [r["branch"] for r in got] == [SMALL, BIG, BIG]


def test_route_batch_posts_to_the_batch_endpoint(monkeypatch):
    captured: dict = {}
    serve(monkeypatch, {"model": PAYLOAD["model"], "results": []}, captured)
    client.route_batch(["a", "b"], url="http://example:9999")
    assert captured["url"] == "http://example:9999/v1/route/batch"
    assert captured["body"] == {"prompts": ["a", "b"]}


def test_route_batch_scales_its_timeout_with_the_prompt_count(monkeypatch):
    """One forward pass per prompt, so a fixed short timeout would abort a large batch."""
    captured: dict = {}
    serve(monkeypatch, {"model": PAYLOAD["model"], "results": []}, captured)
    client.route_batch(["x"] * 100)
    assert captured["timeout"] > client.CONNECT_TIMEOUT
    assert captured["timeout"] >= client.BATCH_TIMEOUT_BASE


def test_route_batch_falls_back_like_the_single_route(monkeypatch):
    refuse(monkeypatch)
    with pytest.raises(client.NoServer):
        client.route_batch(["a"])
    serve(monkeypatch, {"model": "uniform", "results": []})
    with pytest.raises(client.NoServer, match="uniform"):
        client.route_batch(["a"])


def test_a_stranger_on_the_port_is_not_trusted(monkeypatch):
    """Ports get reused. A collision with another local service on 8765 is what moved the
    default. Any 200 that happens to be JSON must not be routed on."""
    for impostor in (
        {"status": "ok", "result": "something else entirely"},
        {"model": "whisper-large-v3", "text": "a transcription"},
        ["not", "even", "an", "object"],
        {},
    ):
        serve(monkeypatch, impostor)
        with pytest.raises(client.NoServer, match="does not look like a verdict server"):
            client.route("anything")


def test_a_stranger_is_not_trusted_on_the_batch_endpoint_either(monkeypatch):
    serve(monkeypatch, {"model": "whisper-large-v3", "segments": []})
    with pytest.raises(client.NoServer, match="does not look like a verdict server"):
        client.route_batch(["a"])


def test_the_default_port_avoids_the_one_local_whisper_holds():
    """Not a style point: a colliding default means `serve` dies on bind while `route` talks to
    whatever else is listening."""
    assert ":8765" not in client.DEFAULT_URL


def http_error(monkeypatch, code, body):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, code, "error", {},
                                     BytesIO(json.dumps(body).encode()))

    monkeypatch.setattr(client.urllib.request, "urlopen", fake_urlopen)


@pytest.mark.parametrize("code", [400, 422, 500])
def test_a_server_that_rejects_the_call_is_not_a_missing_server(monkeypatch, code):
    """Falling back here hides the server's error and loads a second model for a call the server
    already said is wrong."""
    http_error(monkeypatch, code, {"detail": "score needs at least 2 levels"})
    with pytest.raises(client.ServerError, match="at least 2 levels") as info:
        client.decide("x", {"q": {"type": "noul"}})
    assert not isinstance(info.value, client.NoServer)
    assert info.value.status == code


def test_a_404_is_a_stranger_on_the_port(monkeypatch):
    """Something else listening, with no such endpoint: the same as no verdict server."""
    http_error(monkeypatch, 404, {"detail": "Not Found"})
    with pytest.raises(client.NoServer):
        client.decide("x", {"q": {"type": "noul"}})
