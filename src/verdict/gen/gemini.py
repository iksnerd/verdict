"""Generate states with a cheap Gemini model. Same interface as OllamaClient, for when a few
hundred states are needed faster than a local 4B model can write them (~1 s against ~11 s)."""
from __future__ import annotations

import httpx

from ..label.teacher import cost_of


class GenError(RuntimeError):
    """The generator refused the request itself (a bad key, an unknown model, a malformed call).
    No new seed fixes that, so it must not look like a text the loop rejected."""


class GeminiGenClient:
    def __init__(self, model: str, key: str, temperature: float = 1.0):
        self.model, self.key, self.temperature = model, key, temperature
        self.tokens_in = self.tokens_out = 0
        #: Calls that timed out or met a server error; the loop retries those with a new seed.
        self.failures = 0

    def generate(self, prompt: str, seed: int) -> str:
        try:
            r = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self.key},
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": self.temperature, "maxOutputTokens": 500, "seed": seed}},
                timeout=60,
            )
        except httpx.HTTPError:
            self.failures += 1
            return ""
        if r.status_code in (400, 401, 403, 404):
            raise GenError(f"{self.model} answered {r.status_code}: check the key and the model name")
        if r.status_code != 200:
            self.failures += 1
            return ""
        j = r.json()
        u = j.get("usageMetadata", {})
        self.tokens_in += u.get("promptTokenCount", 0)
        self.tokens_out += u.get("candidatesTokenCount", 0)
        try:
            return j["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return ""  # blocked or empty: the generation loop retries with a new seed

    def cost_usd(self) -> float:
        return cost_of(self.model, self.tokens_in, self.tokens_out)
