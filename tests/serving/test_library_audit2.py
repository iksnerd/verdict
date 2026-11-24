"""Audit 2: a request's model name resolves to the checkpoint served before `measured_on`."""
from __future__ import annotations

import pytest

from verdict import library
from verdict.schema import DecideRequest, DecideResponse, NoulAnswer


class Served:
    name = "fake"
    model_id = "/somewhere/verdict-v1-mlx"
    multilingual_id = "aac6fef/laya-multilingual-mlx"
    extra_checkpoints = {"ft": "/elsewhere/verdict-v1-mlx"}

    def __init__(self):
        self.sent = None

    def decide(self, request):
        self.sent = request
        return DecideResponse(model="fake", answers={
            q: NoulAnswer(noul=0.5) for q in request.questions})


def _measured_noul():
    return next(e.question for e in library.load() if e.question["type"] == "noul")


def _asked(model):
    backend = Served()
    request = DecideRequest(state={"text": "x", "command": "ls"}, model=model,
                            questions={"q": _measured_noul()})
    library.ask(backend, request, allow_unmeasured=True)
    return backend.sent.questions["q"].type


@pytest.mark.parametrize("model", [None, "english", "jev-latest", "ft"])
def test_the_served_fine_tune_keeps_its_measured_yes_no(model):
    assert _asked(model) == "noul"


def test_the_multilingual_checkpoint_is_not_the_one_measured():
    assert _asked("multilingual") == "choice"


def test_served_checkpoint_resolution():
    b = Served()
    assert library.served_checkpoint(b, None) == b.model_id
    assert library.served_checkpoint(b, "english") == b.model_id
    assert library.served_checkpoint(b, "multilingual") == b.multilingual_id
    assert library.served_checkpoint(b, "ft") == "/elsewhere/verdict-v1-mlx"
    assert library.served_checkpoint(b, "typo") == "typo"
