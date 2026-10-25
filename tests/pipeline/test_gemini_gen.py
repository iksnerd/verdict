import httpx

from verdict.gen.gemini import GeminiGenClient


class R:
    def __init__(self, status, body):
        self.status_code, self._b, self.text = status, body, str(body)

    def json(self):
        return self._b


def test_returns_text_and_counts_tokens(monkeypatch):
    sent = {}

    def post(url, headers, json, timeout):
        sent.update(url=url, json=json, headers=headers)
        return R(200, {"candidates": [{"content": {"parts": [{"text": "hello there"}]}}],
                       "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 7}})

    monkeypatch.setattr(httpx, "post", post)
    c = GeminiGenClient("gemini-2.5-flash-lite", key="k")
    assert c.generate("write", seed=5) == "hello there"
    assert sent["json"]["generationConfig"]["seed"] == 5 and "gemini-2.5-flash-lite" in sent["url"]
    assert (c.tokens_in, c.tokens_out) == (40, 7) and c.cost_usd() > 0


def test_blocked_or_failed_calls_return_empty_string_so_the_loop_retries(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: R(200, {"candidates": []}))
    assert GeminiGenClient("gemini-2.5-flash-lite", key="k").generate("x", 1) == ""
    monkeypatch.setattr(httpx, "post", lambda *a, **k: R(500, {}))
    assert GeminiGenClient("gemini-2.5-flash-lite", key="k").generate("x", 1) == ""
