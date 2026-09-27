"""Laya on Apple Silicon, as a `Backend`.

This is the HTTP seam: a `DecideRequest` in, a validated `DecideResponse` out. It is a thin
translation layer over an `Engine` (`engine.py`), which is the batch seam and the thing that
actually loads the checkpoint. Bulk callers such as scoring scripts use the `Engine`
directly rather than paying for pydantic on every question.

The port is the `mizorewww/laya-mlx` project on GitHub, which is what `pip install laya-mlx` builds
from; `aac6fef/laya-mlx` is the Hugging Face repo holding pre-converted FP16 weights. There is no
`mizorewww/laya-mlx` model on Hugging Face and that path 401s.

Install with `uv sync --extra mlx`. The port claims to reproduce upstream's answers on its own 63
validation questions. Checked on our own data as well (FINDINGS §13): 160/160 argmax agreement
against the PyTorch original on the frozen test split, max probability difference 0.0011.
"""
from __future__ import annotations

from typing import Any

from .backend import UnknownModel
from .engine import DEFAULT_MODEL, Engine, load
from .schema import (ChoiceAnswer, DecideRequest, DecideResponse, NoulAnswer, ScoreAnswer, Usage,
                     laya_questions)


class MlxBackend:
    """`load_options` go straight through to `laya_mlx.load`; see `engine.load` for which ones."""

    def __init__(self, model_id: str = DEFAULT_MODEL, agent: Any = None,
                 multilingual_id: str | None = None, extra_checkpoints: dict[str, str] | None = None,
                 budget: int | None = None, bits: int = 16, **load_options: Any):
        # `bits` applies to the fine-tune only: §36 measured it, not the multilingual checkpoint.
        self.engine: Engine = (
            Engine(model_id, agent=agent, bits=bits, **load_options) if agent is not None
            else load(model_id, bits=bits, **load_options)
        )
        self.model_id = model_id
        self.name = self.engine.name
        self.multilingual_id = multilingual_id
        #: Further named checkpoints a request can select (config.py's [model.extra]). Named
        #: separately from `multilingual`, which keeps its own dedicated config key and
        #: lang-detection behaviour; this is plain alternates with neither.
        self.extra_checkpoints = dict(extra_checkpoints) if extra_checkpoints else {}
        #: Tokens of each `decide` state the model sees; None reads it whole (up to laya's 512).
        self.budget = budget
        self.load_options = load_options
        #: Lazily loaded and cached by name, `multilingual` included: loading any of these at
        #: startup would put a second model on the GPU for traffic that may never ask for it.
        self._loaded: dict[str, Engine] = {}

    def known_models(self) -> set[str]:
        """Names besides the served checkpoint that a request may select."""
        names = set(self.extra_checkpoints)
        if self.multilingual_id:
            names.add("multilingual")
        return names

    def engine_for(self, model: str | None) -> Engine:
        """The engine a request names, loading and caching it on first use."""
        if model is None or model == "english":
            return self.engine
        checkpoint_id = self.extra_checkpoints.get(model)
        if model == "multilingual" and self.multilingual_id:
            checkpoint_id = self.multilingual_id
        if checkpoint_id is None:
            known = ", ".join(sorted(self.known_models())) or "none"
            raise UnknownModel(f"this server does not know a checkpoint named {model!r}; "
                               f"known extra checkpoints: {known}. Set [model.extra] or "
                               "[model].multilingual in verdict.toml and restart it")
        if checkpoint_id not in self._loaded:
            self._loaded[checkpoint_id] = load(checkpoint_id, **self.load_options)
        return self._loaded[checkpoint_id]

    def decide(self, request: DecideRequest) -> DecideResponse:
        questions = laya_questions(request.questions)
        engine = self.engine_for(request.model)
        result = engine.run(engine.clip_state(request.state, self.budget), questions)
        raw = result["answers"]
        answers: dict[str, Any] = {}
        for qid, a in raw.items():
            if a["type"] == "choice":
                answers[qid] = ChoiceAnswer(
                    choice=a["choice"], probabilities=a["probabilities"], confidence=a["confidence"]
                )
            elif a["type"] == "score":
                answers[qid] = ScoreAnswer(
                    score=a["score"],
                    legend=a["legend"],
                    probabilities=a["probabilities"],
                    confidence=a["confidence"],
                )
            else:
                answers[qid] = NoulAnswer(noul=a["noul"], confidence=a["confidence"])
        usage = result.get("usage") or {}
        return DecideResponse(model=engine.name, answers=answers,
                              usage=Usage(input_tokens=int(usage.get("input_tokens", 0))))
