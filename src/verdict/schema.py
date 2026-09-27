"""Wire format. Mirrors laya's `Agent.predict(state, questions)` so any Laya-style
backend can be dropped in without translation, and so training data built here loads
in the Laya fine-tune notebook."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

from .errors import QuestionError


#: What to ask: text, or JSON that laya renders compactly. Optional, as in Jev's API; a missing one
#: reaches laya as "". Laya and Jev both accept structured instructions, so this does too.
Instructions = Union[str, dict[str, Any], list[Any], None]


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Instructions = None
    #: A criterion may be structured. laya renders a dict or list as compact JSON, so a rubric
    #: arrives as JSON rather than a Python repr. The schema must accept whatever the format it
    #: mirrors accepts.
    criteria: Union[dict[str, Any], list[Any]]

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _enough_options(self):
        if len(self.criteria) < 2:
            raise ValueError("choice needs at least 2 options")
        return self

    def option_keys(self) -> list[str]:
        return list(self.criteria)


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Instructions = None
    criteria: list[Any]

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _enough_levels(self):
        if len(self.criteria) < 2:
            raise ValueError("score needs at least 2 levels")
        return self

    def option_keys(self) -> list[str]:
        return [str(i) for i in range(len(self.criteria))]


class NoulQuestion(BaseModel):
    """A yes/no question.

    `criteria` describes what each side *means*. Without it the model literally reads
    "true: yes, the statement holds", which is no information at all, and it is the only context
    channel this format has: there is no system prompt, so the levers are the instructions, the
    option descriptions and the state. It measurably moves answers: describing both sides took a
    pure lookup that mentions money ("what's the current VAT rate in Germany") from 0.574 to 0.384
    on `is_sensitive`, which is the difference between escalating it and not (FINDINGS §22).
    """

    type: Literal["noul"]
    instructions: Instructions = None
    criteria: Union[dict[str, Any], None] = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _known_sides(self):
        if self.criteria is not None:
            unknown = set(self.criteria) - {"true", "false"}
            if unknown:
                raise ValueError(f"noul criteria keys must be 'true' and/or 'false', got {sorted(unknown)}")
        return self


Question = Annotated[Union[ChoiceQuestion, ScoreQuestion, NoulQuestion], Field(discriminator="type")]

_QUESTIONS = TypeAdapter(dict[str, Question])


_TYPES = {"choice", "score", "noul"}


def _readable(exc: ValidationError) -> str:
    """`q: score needs at least 2 levels`, not pydantic's `dict[str,tagged-union[...]]` dump: the
    person or agent reading it knows their questions, not our schema's internals."""
    out = []
    for e in exc.errors():
        # loc is (question id, [the type tag pydantic dispatched on], field...); drop the tag.
        loc = [str(p) for p in e["loc"]]
        if len(loc) > 1 and loc[1] in _TYPES:
            del loc[1]
        msg = e["msg"].removeprefix("Value error, ")
        out.append(f"{'.'.join(loc)}: {msg[0].lower() + msg[1:] if msg else msg}")
    return "; ".join(out)


def laya_questions(questions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate a bank and put it in the shape laya reads, for the server and the in-process path
    alike, so a question never works through one and fails through the other. laya-mlx raises on
    a question with no `instructions` key; Jev treats it as optional, so a missing one is ""."""
    if not questions:
        raise QuestionError("at least one question is required")
    try:
        parsed = _QUESTIONS.validate_python(questions)
    except ValidationError as exc:
        raise QuestionError(f"invalid questions: {_readable(exc)}") from exc
    return {qid: {"instructions": "", **q.model_dump(exclude_none=True)}
            for qid, q in parsed.items()}


class DecideRequest(BaseModel):
    state: Union[str, dict[str, Any], list[Any]]
    questions: dict[str, Question]
    #: Which checkpoint answers, named as laya's `Router.predict(model=...)` names them. Omitted
    #: means the served one. The valid names are server-configured (config.py's [model.extra],
    #: plus multilingual), not fixed here, so an unknown one is refused by the backend
    #: (UnknownModel -> 400) rather than by this schema.
    model: Optional[str] = None

    @model_validator(mode="after")
    def _non_empty(self):
        if not self.questions:
            raise ValueError("at least one question is required")
        return self


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    #: Each level as the question gave it: laya returns structured criteria unchanged, and Jev's
    #: Score contract allows them.
    legend: dict[str, Union[str, dict[str, Any], list[Any]]]
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float
    confidence: float


Answer = Annotated[Union[ChoiceAnswer, ScoreAnswer, NoulAnswer], Field(discriminator="type")]


class Usage(BaseModel):
    #: Jev's own `Usage` allows either count to be missing when unreported; every backend here
    #: always computes a real one (backend.py, backend_mlx.py, api.py), so `output_tokens` stays
    #: a concrete `0` rather than widening to match a case that never happens (docs/api.md).
    input_tokens: Optional[int] = None
    output_tokens: int = 0


class DecideResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    #: Tokens the model read. Served on `/v1/systemone` only; `/v1/decide` leaves it out, since
    #: two consumers parse that output (CLAUDE.md).
    usage: Optional[Usage] = None


class SystemOneRequest(BaseModel):
    """TypeSafe's Jev request (`POST /v1/systemone`). `model` is required there, so it is here;
    `api.to_decide_request` maps its name onto a checkpoint."""

    state: Union[str, dict[str, Any], list[Any]]
    model: str
    questions: dict[str, Question]

    @model_validator(mode="after")
    def _non_empty(self):
        if not self.questions:
            raise ValueError("at least one question is required")
        return self


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelCard(BaseModel):
    name: str
    description: str
    release_date: str


class ModelList(BaseModel):
    models: list[ModelCard]
