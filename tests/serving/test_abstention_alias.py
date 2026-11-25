"""`schema.Abstention` is the one spelling of the mark's values; `judge/abstain.py` names the same."""
from typing import get_args

from verdict import schema
from verdict.judge import abstain


def test_abstention_alias_matches_constants():
    assert get_args(schema.Abstention) == (abstain.PASSED, abstain.ABSTAINED, abstain.UNEVALUATED)


def test_answers_use_the_alias():
    for model in (schema.ChoiceAnswer, schema.ScoreAnswer, schema.NoulAnswer):
        props = model.model_json_schema()["properties"]["abstention"]
        values = [b["enum"] for b in props["anyOf"] if "enum" in b]
        assert values == [list(get_args(schema.Abstention))]
