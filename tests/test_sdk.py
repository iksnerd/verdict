"""The typed Python API: `from verdict import Verdict, Noul, Choice, Score`.

Structured like TypeSafe's SDK on purpose: typed question classes, a result with `.nouls`,
`.choices` and `.scores`, a sync `with` client, one error type with `.status`. The seams are
`Verdict.decide()` and the names `verdict` exports; the backend is a stub, never a model.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from verdict import client as http


def test_importing_verdict_stays_light():
    """The CLI pays for every import at startup; the typed API loads on first use."""
    code = "import verdict, sys; assert 'pydantic' not in sys.modules, 'pydantic imported'"
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_the_names_are_exported():
    from verdict import Choice, Noul, Result, Score, Verdict, VerdictError  # noqa: F401


def test_questions_are_typed_and_leave_unset_fields_off_the_wire():
    from verdict import Choice, Noul, Score
    assert Noul(instructions="Is it spam?").model_dump(exclude_none=True) == {
        "type": "noul", "instructions": "Is it spam?"}
    assert Choice(instructions="Which?", criteria={"a": "x", "b": "y"}).type == "choice"
    assert Score(instructions="How urgent?", criteria=["low", "high"]).criteria == ["low", "high"]


def test_a_question_that_cannot_be_asked_fails_at_construction():
    from verdict import Choice
    with pytest.raises(ValueError):
        Choice(instructions="?", criteria={"only": "one"})


@pytest.fixture
def uniform():
    from verdict import Verdict
    from verdict.backends.base import UniformBackend
    return Verdict(backend=UniformBackend())


def questions():
    from verdict import Choice, Noul, Score
    return {
        "spam": Noul(instructions="Is this spam?"),
        "dept": Choice(instructions="Which team?", criteria={"billing": "refunds", "tech": "bugs"}),
        "urgency": Score(instructions="How urgent?", criteria=["low", "mid", "high"]),
    }


def test_results_group_answers_by_type_like_the_sdk(uniform):
    r = uniform.decide({"message": "I was charged twice"}, questions())
    assert set(r.nouls) == {"spam"} and set(r.choices) == {"dept"} and set(r.scores) == {"urgency"}
    assert r.nouls["spam"].noul == 0.5
    assert r.choices["dept"].choice in {"billing", "tech"}
    assert set(r.answers) == {"spam", "dept", "urgency"} and r.model == "uniform"


def test_raw_dict_questions_work_too(uniform):
    r = uniform.decide("hello", {"q": {"type": "noul", "instructions": "Is this a greeting?"}})
    assert "q" in r.nouls


def test_a_question_measured_at_chance_is_refused_with_a_status_and_code(uniform):
    from verdict import Noul, VerdictError
    with pytest.raises(VerdictError) as info:
        uniform.decide({"message": "hi"}, {"p": Noul(instructions="How dangerous is this ticket?")})
    err = info.value
    assert err.status == 422 and err.code == "refused"
    assert "allow_unmeasured=True" in str(err) and "--allow-unmeasured" not in str(err)
    assert isinstance(err, Exception)


def test_the_opt_out_asks_anyway():
    from verdict import Noul, Verdict
    from verdict.backends.base import UniformBackend
    v = Verdict(backend=UniformBackend(), allow_unmeasured=True)
    assert "p" in v.decide({"message": "hi"}, {"p": Noul(instructions="How dangerous is this?")}).nouls


def test_a_state_missing_a_named_field_is_refused(uniform):
    from verdict import Noul, VerdictError
    with pytest.raises(VerdictError, match="message"):
        uniform.decide({"text": "x"}, {"q": Noul(instructions="Is `message` about money?")})


def test_a_server_url_is_the_default_transport(monkeypatch):
    from verdict import Noul, Verdict
    seen = {}

    def fake(state, questions, url=None, model=None, **flags):
        seen.update(state=state, url=url, model=model, flags=flags)
        return {"model": "served", "answers": {"q": {"type": "noul", "noul": 0.9, "confidence": 0.9}}}

    monkeypatch.setattr(http, "decide", fake)
    r = Verdict(url="http://127.0.0.1:9999", allow_unmeasured=True).decide(
        "hi", {"q": Noul(instructions="Is this a greeting?")}, model="english")
    assert r.nouls["q"].noul == 0.9 and r.model == "served"
    assert seen["url"] == "http://127.0.0.1:9999" and seen["model"] == "english"
    assert seen["flags"] == {"allow_unmeasured": True}


def test_a_servers_refusal_keeps_its_status_and_code(monkeypatch):
    from verdict import Noul, Verdict, VerdictError

    def refuse(*a, **k):
        raise http.ServerError(422, "q asks about 'risk'", code="refused")

    monkeypatch.setattr(http, "decide", refuse)
    with pytest.raises(VerdictError) as info:
        Verdict(url="http://127.0.0.1:9999").decide("hi", {"q": Noul(instructions="Is it ok?")})
    assert info.value.status == 422 and info.value.code == "refused"


def test_no_server_says_how_to_start_one(monkeypatch):
    from verdict import Noul, Verdict, VerdictError

    def gone(*a, **k):
        raise http.NoServer("no verdict server at http://127.0.0.1:8799 (refused)")

    monkeypatch.setattr(http, "decide", gone)
    with pytest.raises(VerdictError) as info:
        Verdict().decide("hi", {"q": Noul(instructions="Is it ok?")})
    assert info.value.code == "no_server" and info.value.status is None
    assert "verdict serve" in str(info.value)


def test_it_is_a_context_manager(uniform):
    with uniform as v:
        assert v.decide("hi", {"q": {"type": "noul", "instructions": "Is this a greeting?"}})


def test_a_served_refusal_names_the_python_opt_out_not_the_cli_flag(monkeypatch):
    from verdict import Noul, Verdict, VerdictError

    def refuse(*a, **k):
        # What client.decide raises: the client already rewrote the server's HTTP wording to the
        # CLI flag, which is the wrong spelling for a Python caller.
        raise http.ServerError(422, "q asks about 'risk'. Refused; --allow-unmeasured asks anyway",
                               code="refused")

    monkeypatch.setattr(http, "decide", refuse)
    with pytest.raises(VerdictError) as info:
        Verdict(url="http://127.0.0.1:9999").decide("hi", {"q": Noul(instructions="Is it ok?")})
    assert "allow_unmeasured=True" in str(info.value) and "--allow-unmeasured" not in str(info.value)


# ---- models() and capabilities(), as the SDK's client has models ------------------------------

def test_a_local_backend_lists_its_models_and_capabilities(uniform):
    names = [m["name"] for m in uniform.models()]
    assert "uniform" in names
    caps = uniform.capabilities()
    assert caps.family == "uniform" and caps.model == "uniform"


def test_a_backend_that_lists_its_own_models_is_asked(monkeypatch):
    from verdict import Verdict

    class Own:
        name = "own"
        def models(self): return [{"name": "m1"}, {"name": "m2"}]
        def decide(self, request): raise AssertionError

    assert [m["name"] for m in Verdict(backend=Own()).models()] == ["m1", "m2"]


def test_a_servers_models_and_capabilities_come_over_http(monkeypatch):
    from verdict import Verdict
    monkeypatch.setattr(http, "models", lambda url=None, **k: [{"name": "english"}])
    monkeypatch.setattr(http, "capabilities",
                        lambda url=None, model=None, **k: {"family": "laya", "model": model or "english"})
    v = Verdict(url="http://127.0.0.1:9999")
    assert v.models() == [{"name": "english"}]
    assert v.capabilities("multilingual").family == "laya"
    assert v.capabilities("multilingual").model == "multilingual"


def test_models_with_no_server_says_how_to_start_one(monkeypatch):
    from verdict import Verdict, VerdictError

    def gone(*a, **k):
        raise http.NoServer("no verdict server at http://127.0.0.1:8799 (refused)")

    monkeypatch.setattr(http, "models", gone)
    monkeypatch.setattr(http, "capabilities", gone)
    for call in (Verdict().models, Verdict().capabilities):
        with pytest.raises(VerdictError) as info:
            call()
        assert info.value.code == "no_server" and "verdict serve" in str(info.value)


def test_only_an_upstream_failure_is_labelled_upstream():
    """A bug that raises ValueError must look like a bug, not like a remote outage to retry."""
    from verdict import Noul, Verdict, VerdictError
    from verdict.systemone import UpstreamError

    class Down:
        name = "down"
        def decide(self, request): raise UpstreamError("503 from upstream")

    class Buggy:
        name = "buggy"
        def decide(self, request): raise ValueError("a bug in the backend")

    q = {"q": Noul(instructions="Is it a greeting?")}
    with pytest.raises(VerdictError) as info:
        Verdict(backend=Down()).decide("hi", q)
    assert info.value.code == "upstream" and info.value.status == 502
    with pytest.raises(ValueError, match="a bug") as bug:
        Verdict(backend=Buggy()).decide("hi", q)
    assert not isinstance(bug.value, VerdictError)


def test_classify_walks_a_tree_through_the_clients_own_decide(uniform):
    tree = {"instructions": "What is it about?",
            "groups": {"a": {"options": {"a1": None, "a2": None}},
                       "b": {"options": {"b1": None, "b2": None}}}}
    r = uniform.classify("hello", tree, beam=2)
    assert r.label in {"a1", "a2", "b1", "b2"} and r.passes == 3
    assert abs(r.probability - 0.25) < 1e-6   # uniform: 1/2 * 1/2


def test_classify_reports_a_too_wide_tree_as_a_verdict_error(uniform):
    from verdict import VerdictError
    wide = {"instructions": "?", "groups": {"g": {"options": {f"o{i}": None for i in range(21)}}}}
    with pytest.raises(VerdictError) as info:
        uniform.classify("hello", wide)
    assert info.value.status == 422 and info.value.code == "refused"
