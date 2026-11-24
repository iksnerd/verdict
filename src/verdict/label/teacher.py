"""Gemini teachers: ask for a probability distribution per question, parse and validate it.

The API key is never stored in this project. `load_key` reads GOOGLE_API_KEY/GEMINI_API_KEY
from the environment or, failing that, only that one variable out of a key file you name."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

import httpx

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LabelError(Exception):
    pass


PRICES = {  # USD per 1M tokens (input, output); from ai.google.dev pricing, 2026-09-19. Re-verify.
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}


def load_key(key_file: str | None = None) -> str:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    if key_file:
        for line in open(key_file):
            m = re.match(r"\s*(?:export\s+)?(GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*(.*)", line)
            if m and m.group(2).strip().strip("\"'"):
                return m.group(2).strip().strip("\"'")
    raise LabelError("no GEMINI_API_KEY or GOOGLE_API_KEY in the environment or the key file")


def _options(q: dict) -> list[str]:
    if q["type"] == "noul":
        return ["true", "false"]
    crit = q["criteria"]
    return list(crit) if isinstance(crit, dict) else [str(i) for i in range(len(crit))]


DEFAULT_PREAMBLE = (
    "You label one entry from a shared engineering room's ledger. For each question, give your honest "
    "probability distribution over its options, reflecting real uncertainty (do not force 0 or 1 unless "
    "you are certain). Probabilities for a question must sum to 1."
)


def build_prompt(state: Any, bank: dict, preamble: str | None = None) -> str:
    """`preamble` is the only editable part (an optimizer's target file). The entry, the questions and
    the reply format are fixed so an edit cannot break parsing."""
    lines = [
        (preamble or DEFAULT_PREAMBLE).strip(),
        "",
        "ENTRY:",
        json.dumps(state, ensure_ascii=False),
        "",
        "QUESTIONS:",
    ]
    for qid, q in bank.items():
        lines.append(f'- "{qid}": {q["instructions"]}')
        if q["type"] == "choice" and isinstance(q["criteria"], dict):
            for k, v in q["criteria"].items():
                lines.append(f'    "{k}": {v}')
        else:
            lines.append(f"    options: {_options(q)}")
    lines += [
        "",
        'Reply with JSON only, shaped {"<question id>": {"<option>": <probability>, ...}, ...}. '
        'For yes/no questions the options are "true" and "false".',
    ]
    return "\n".join(lines)


def parse_labels(raw: str, bank: dict) -> dict:
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LabelError(f"not JSON: {e}") from e
    out: dict[str, Any] = {}
    for qid, q in bank.items():
        if qid not in data or not isinstance(data[qid], dict):
            raise LabelError(f"missing question {qid}")
        dist = data[qid]
        opts = _options(q)
        if any(o not in dist for o in opts):
            raise LabelError(f"{qid}: missing options")
        vals = [max(0.0, float(dist[o])) for o in opts]
        total = sum(vals)
        if total <= 0:
            raise LabelError(f"{qid}: zero probability mass")
        probs = [v / total for v in vals]
        out[qid] = probs[0] if q["type"] == "noul" else dict(zip(opts, probs))
    return out


def mean_labels(labels: list[dict]) -> dict:
    out: dict[str, Any] = {}
    for qid in labels[0]:
        first = labels[0][qid]
        if isinstance(first, dict):
            out[qid] = {k: sum(l[qid][k] for l in labels) / len(labels) for k in first}
        else:
            out[qid] = sum(l[qid] for l in labels) / len(labels)
    return out


def agreement(a: dict, b: dict) -> dict:
    """Per question: same argmax for choice, same side of 0.5 for noul."""
    out = {}
    for qid in a:
        if isinstance(a[qid], dict):
            out[qid] = float(max(a[qid], key=a[qid].get) == max(b[qid], key=b[qid].get))
        else:
            out[qid] = float((a[qid] >= 0.5) == (b[qid] >= 0.5))
    return out


Transport = Callable[[str, str], "tuple[str, dict]"]


def gemini_transport(key: str, temperature: float = 0.7) -> Transport:
    def call(model: str, prompt: str):
        try:
            r = httpx.post(
                GEMINI_URL.format(model=model),
                headers={"x-goog-api-key": key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
                },
                timeout=120,
            )
        except httpx.HTTPError as e:  # a timeout costs one label, not the rest of the run
            raise LabelError(f"{type(e).__name__}: {e}") from e
        if r.status_code != 200:
            raise LabelError(f"HTTP {r.status_code}: {r.text[:200]}")
        j = r.json()
        try:
            text = j["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LabelError(f"no candidate text: {str(j)[:200]}") from e
        return text, j.get("usageMetadata", {})

    return call


class UnpricedModelError(ValueError):
    """A model with no price: counted as free, the spend cap could never trip on it."""


def price_of(model: str, prices: dict | None = None) -> tuple[float, float]:
    """USD per 1M tokens (input, output): an explicit price first, then `PRICES`. An unknown model
    is an error, never $0."""
    if prices and model in prices:
        pin, pout = prices[model]
        return float(pin), float(pout)
    if model in PRICES:
        return PRICES[model]
    raise UnpricedModelError(f"no price for model {model!r}: add it to PRICES in verdict/label/teacher.py "
                             "or state it in the config as label.prices, {\"model\": [input, output]} "
                             "USD per 1M tokens. An unpriced model would cost $0 and the spend cap could never trip.")


def cost_of(model: str, tokens_in: int, tokens_out: int, prices: dict | None = None) -> float:
    pin, pout = price_of(model, prices)
    return (tokens_in * pin + tokens_out * pout) / 1e6


class Teacher:
    def __init__(self, model: str, transport: Transport, preamble: str | None = None, prices: dict | None = None):
        self.model, self.transport, self.preamble, self.prices = model, transport, preamble, prices
        self.tokens_in = self.tokens_out = self.calls = 0
        self.last_in = self.last_out = 0  # tokens spent by the most recent label() call, retries included

    def label(self, state: Any, bank: dict) -> dict:
        self.last_in = self.last_out = 0
        prompt = build_prompt(state, bank, self.preamble)
        last: Exception | None = None
        for _ in range(2):
            try:  # a timeout or an HTTP error gets the same second try as a malformed reply
                text, usage = self.transport(self.model, prompt)
            except LabelError as e:
                last = e
                continue
            self.calls += 1
            tin = usage.get("promptTokenCount", 0)
            tout = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
            self.tokens_in += tin
            self.tokens_out += tout
            self.last_in += tin
            self.last_out += tout
            try:
                return parse_labels(text, bank)
            except LabelError as e:
                last = e
        raise LabelError(f"{self.model}: gave up after 2 tries: {last}")

    def cost_usd(self) -> float:
        return cost_of(self.model, self.tokens_in, self.tokens_out, self.prices)
