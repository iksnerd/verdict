"""Backend facts, separate from the shared SystemOne wire format.

None means unknown, not unlimited. Jev values are a documentation snapshot (2026-09-27),
not a tokenizer or a claim that another SystemOne server is Jev.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .errors import QuestionError
from .schema import ChoiceQuestion, DecideRequest, ScoreQuestion


class Capabilities(BaseModel):
    model_config = {"frozen": True}

    family: Literal["laya", "jev", "systemone", "uniform", "unknown"] = "unknown"
    model: str
    max_request_tokens: int | None = None
    max_state_and_question_tokens: int | None = None
    state_token_budget: int | None = None
    max_choice_options: int | None = None
    recommended_choice_options: int | None = None
    max_score_levels: int | None = None
    option_token_cap: int | None = None
    question_execution: Literal["per_question", "parallel", "unknown"] = "unknown"
    noul_confidence: Literal["max_probability", "absent", "provider_defined", "unknown"] = "unknown"
    choice_score_confidence: str = "provider_defined"
    token_usage: str = "provider_reported"
    notes: tuple[str, ...] = ()

    def validate_request(self, request: DecideRequest) -> None:
        for qid, q in request.questions.items():
            if isinstance(q, ChoiceQuestion):
                if self.max_choice_options and len(q.criteria) > self.max_choice_options:
                    raise QuestionError(f"{qid}: {self.family} accepts at most {self.max_choice_options} choice options")
                if self.family == "jev" and not isinstance(q.criteria, dict):
                    raise QuestionError(f"{qid}: Jev choice criteria must be a map of named options")
            if isinstance(q, ScoreQuestion) and self.max_score_levels and len(q.criteria) > self.max_score_levels:
                raise QuestionError(f"{qid}: {self.family} accepts at most {self.max_score_levels} score levels")
            if self.family == "jev" and q.instructions is None:
                raise QuestionError(f"{qid}: Jev requires instructions")


def jev_capabilities(model: str) -> Capabilities:
    return Capabilities(
        family="jev", model=model, max_request_tokens=64000,
        max_state_and_question_tokens=32000, max_choice_options=255, max_score_levels=10,
        question_execution="parallel", noul_confidence="absent",
        notes=("TypeSafe documentation snapshot: 2026-09-27; upstream enforces token limits.",
               "No local clipping or Laya question rewriting; refit thresholds when changing models."),
    )
