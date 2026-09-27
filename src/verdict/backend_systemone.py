"""Any /v1/systemone-compatible server as a `Backend`: Kev, a real Jev, or another verdict
instance. They all answer this exact shape, so unlike backend_mlx.py this needs no state/questions
translation -- only typing the raw JSON. The HTTP itself (auth, retries, backoff) is bench.py's
`systemone_asker`, already tested there; this is purely the adaptation layer on top of it."""
from __future__ import annotations

from typing import Any

from .bench import systemone_asker
from .schema import ChoiceAnswer, DecideRequest, DecideResponse, NoulAnswer, ScoreAnswer, Usage


def _answer(raw: dict[str, Any]):
    if raw["type"] == "choice":
        return ChoiceAnswer(choice=raw["choice"], probabilities=raw["probabilities"],
                            confidence=raw["confidence"])
    if raw["type"] == "score":
        return ScoreAnswer(score=raw["score"], legend=raw["legend"],
                           probabilities=raw["probabilities"], confidence=raw["confidence"])
    #: Jev's own noul answer carries no confidence (docs/FINDINGS, tests/test_systemone.py), and
    #: Kev's matches it. Ours requires one, so a missing one is synthesized the same way Kev's own
    #: docs define Choice confidence, (p_max - 1/K)/(1 - 1/K), read at K=2: abs(noul - 0.5) * 2.
    confidence = raw.get("confidence", abs(raw["noul"] - 0.5) * 2)
    return NoulAnswer(noul=raw["noul"], confidence=confidence)


class SystemOneBackend:
    """`base_url` is any /v1/systemone server: a local `kev.serve`, api.typesafe.ai, or another
    `verdict serve`. `model` is the name that server should answer as (its own default if
    unrecognized, per every /v1/systemone server's own leniency)."""

    def __init__(self, base_url: str, model: str = "jev-latest", key: str = "local", **ask_kwargs: Any):
        self.base_url = base_url
        self.model = model
        self.name = f"systemone:{model}@{base_url}"
        self._key = key
        self._transport = ask_kwargs.get("transport")
        self._ask = systemone_asker(base_url, model, key, **ask_kwargs)

    def models(self) -> list[dict[str, Any]]:
        """The upstream server's own `GET /v1/models`, forwarded as-is: it knows its real model
        list, where the hardcoded jev-latest/english cards elsewhere are laya/Jev-specific and
        would misdescribe whatever this is actually proxying to. A listing failure degrades to an
        empty list rather than breaking the whole endpoint."""
        import httpx

        try:
            with httpx.Client(base_url=self.base_url.rstrip("/"), transport=self._transport,
                              headers={"Authorization": f"Bearer {self._key}"}) as http:
                r = http.get("/v1/models")
                r.raise_for_status()
                return r.json().get("models", [])
        except httpx.HTTPError:
            return []

    def decide(self, request: DecideRequest) -> DecideResponse:
        questions = {qid: q.model_dump(exclude_none=True) for qid, q in request.questions.items()}
        raw = self._ask(request.state, questions)
        answers = {qid: _answer(a) for qid, a in raw["answers"].items()}
        usage = raw.get("usage") or {}
        return DecideResponse(model=raw.get("model", self.name), answers=answers,
                              usage=Usage(input_tokens=usage.get("input_tokens"),
                                         output_tokens=usage.get("output_tokens", 0)))
