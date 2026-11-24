"""One precedence for which checkpoint a request's model name selects."""
from __future__ import annotations

import pytest

from verdict.backends.base import checkpoint_for

EXTRAS = {"fast": "hub/fast", "multilingual": "hub/extra-ml", "english": "hub/extra-en", "jev-latest": "hub/jev"}


@pytest.mark.parametrize("name, multi, expected", [
    (None, "hub/ml", None),
    ("english", "hub/ml", None),            # always the served one, even when an extra claims it
    ("multilingual", "hub/ml", "hub/ml"),   # beats an extra of the same name
    ("multilingual", None, "hub/extra-ml"),
    ("jev-latest", None, "hub/jev"),        # an extra claims the alias
    ("fast", None, "hub/fast"),
    ("nope", "hub/ml", None),
])
def test_checkpoint_for(name, multi, expected):
    assert checkpoint_for(name, EXTRAS, multi) == expected


def test_jev_latest_is_the_served_one_when_no_extra_claims_it():
    assert checkpoint_for("jev-latest", {}, None) is None
