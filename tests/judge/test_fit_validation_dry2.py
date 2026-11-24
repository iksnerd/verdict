"""A malformed fit is a QuestionError naming the question and the key, wherever it is used."""
from __future__ import annotations

import pytest

from verdict.errors import QuestionError
from verdict.judge import calibrate, cascade

BANK = {"spam": {"type": "noul", "instructions": "Is this spam?"}}


def ask(state, questions):
    return {"answers": {"spam": {"type": "noul", "noul": 0.7, "confidence": 0.7}}}


def test_cuts_names_a_noul_fit_without_a_cut():
    with pytest.raises(QuestionError, match=r"'spam'.*'cut'"):
        cascade._cuts(BANK, {"spam": {"type": "noul"}})


def test_cuts_names_a_fit_without_a_type():
    with pytest.raises(QuestionError, match=r"'spam'.*'type'"):
        cascade._cuts(BANK, {"spam": {"cut": 0.4}})


def test_cuts_keeps_its_mismatch_wording():
    with pytest.raises(QuestionError, match="but the question is a noul; fit it on the same type$"):
        cascade._cuts(BANK, {"spam": {"type": "choice", "temperature": 1.0}})


def test_check_names_a_noul_fit_without_a_cut():
    with pytest.raises(QuestionError, match=r"'spam'.*'cut'"):
        cascade.check([{}], BANK, [{"spam": True}], ask, ask, share=0.5,
                      fits={"spam": {"type": "noul"}})


def test_check_refuses_a_fit_of_the_wrong_type():
    with pytest.raises(QuestionError, match="'spam'"):
        cascade.check([{}], BANK, [{"spam": True}], ask, ask, share=0.5,
                      fits={"spam": {"type": "choice", "temperature": 1.0}})


def test_apply_keeps_its_own_mismatch_wording():
    with pytest.raises(QuestionError, match="but the answer is a noul; fit it on the same type of question$"):
        calibrate.apply({"spam": {"type": "noul", "noul": 0.5}},
                        {"spam": {"type": "choice", "temperature": 1.0}})
