from __future__ import annotations

import httpx


DEFAULT_MODEL = "gemma3:4b"
DEFAULT_URL = "http://localhost:11434"


class OllamaClient:
    """Local generation only. Never used to label: a 3-4B model's distributions are too noisy."""

    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL, temperature: float = 0.9):
        self.model = model
        self.url = url.rstrip("/")
        self.temperature = temperature

    def generate(self, prompt: str, seed: int) -> str:
        r = httpx.post(
            f"{self.url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature, "seed": seed, "num_predict": 400},
            },
            timeout=180,
        )
        r.raise_for_status()
        return r.json()["response"]
