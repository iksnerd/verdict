from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..gen.ollama import DEFAULT_MODEL
from ..label.teacher import DEFAULT_CHEAP_MODELS, DEFAULT_REFERENCE_MODEL


class GenCfg(BaseModel):
    provider: str = "ollama"  # "ollama" (local, free, ~11 s/state) or "gemini" (a few cents, ~1 s/state)
    model: str = DEFAULT_MODEL
    n: int = 500
    seed: int = 0
    workers: int = 1  # parallel generation calls (use ~8 for a hosted API, 1 for local Ollama)


class LabelCfg(BaseModel):
    models: list[str] = Field(default_factory=lambda: list(DEFAULT_CHEAP_MODELS))
    reference: str | None = DEFAULT_REFERENCE_MODEL
    repeats: int = 2
    max_cost_usd: float = 1.0
    prices: dict[str, tuple[float, float]] = Field(default_factory=dict)  # model -> (input, output) USD per 1M tokens; for a model not in PRICES
    key_file: str | None = None  # path to an env file; only GOOGLE_API_KEY/GEMINI_API_KEY is read from it


class SplitCfg(BaseModel):
    test_frac: float = 0.2
    holdout_frac: float = 0.0
    seed: int = 0
    require_type_coverage: bool = False  # search seeds until every intended type is in test and holdout


class PipelineConfig(BaseModel):
    name: str
    domain: str = "hub_message_typing"
    gen: GenCfg = Field(default_factory=GenCfg)
    label: LabelCfg = Field(default_factory=LabelCfg)
    split: SplitCfg = Field(default_factory=SplitCfg)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        return cls.model_validate_json(Path(path).read_text())
