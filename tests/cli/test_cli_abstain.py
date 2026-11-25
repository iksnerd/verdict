"""`--abstain-below P` on `verdict decide` and `verdict ask`: an opt-in mark, nothing else moves."""
from __future__ import annotations

import json

import pytest

from verdict import cli, client
from fakes import RISKY_BANK, noul, typed

STATE = '{"kind": "a"}'


def serve(monkeypatch, p, model="fake"):
    def fake_decide(state, questions, url=None, model_=None, **flags):
        return {"model": model, "answers": typed(questions, lambda k: noul(p))}
    monkeypatch.setattr(client, "decide", fake_decide)


def run(capsys, *argv):
    code = cli.main(["decide", STATE, "-q", RISKY_BANK, "--allow-unmeasured", *argv])
    return code, capsys.readouterr()


def answer(out):
    return json.loads(out)["answers"]["risky"]


def test_decide_marks_passed_and_abstained(monkeypatch, capsys):
    serve(monkeypatch, 0.9)
    code, cap = run(capsys, "--abstain-below", "0.8")
    assert code == 0
    assert answer(cap.out)["abstention"] == "passed"
    assert answer(cap.out)["abstention_threshold"] == 0.8
    serve(monkeypatch, 0.6)
    _, cap = run(capsys, "--abstain-below", "0.8")
    assert answer(cap.out)["abstention"] == "abstained"
    assert answer(cap.out)["noul"] == 0.6   # the answer itself is untouched


def test_without_the_flag_no_new_keys_appear(monkeypatch, capsys):
    serve(monkeypatch, 0.6)
    _, cap = run(capsys)
    assert not [k for k in answer(cap.out) if k.startswith("abstention")]


def test_jsonl_rows_carry_the_mark(monkeypatch, capsys):
    serve(monkeypatch, 0.6)
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO('{"kind": "a"}\n{"kind": "b"}\n'))
    code = cli.main(["decide", "--jsonl", "-q", RISKY_BANK, "--allow-unmeasured",
                     "--abstain-below", "0.8", "--pause", "0"])
    rows = [json.loads(l) for l in capsys.readouterr().out.splitlines()]
    assert code == 0 and [r["answers"]["risky"]["abstention"] for r in rows] == ["abstained"] * 2


def test_a_uniform_answer_is_unevaluated(monkeypatch, capsys):
    serve(monkeypatch, 0.5, model="uniform")
    _, cap = run(capsys, "--abstain-below", "0.8")
    assert answer(cap.out)["abstention"] == "unevaluated"


def test_ask_json_carries_the_mark(monkeypatch, capsys):
    serve(monkeypatch, 0.6)
    code = cli.main(["ask", "some text", "Is it bad?", "--json", "--abstain-below", "0.8",
                     "--allow-unmeasured"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["answers"]["q"]["abstention"] == "abstained"


@pytest.mark.parametrize("bad", ["0", "1.5", "-1", "x", "true"])
def test_an_invalid_threshold_is_a_one_line_usage_error(monkeypatch, capsys, bad):
    serve(monkeypatch, 0.6)
    code, cap = run(capsys, "--abstain-below", bad)
    assert code == 2
    assert "answers" not in cap.out          # decide's JSON error line, never an answer
    assert json.loads(cap.out)["error"]["code"] == "usage" if cap.out else True
    assert "Traceback" not in cap.err
    assert cap.err.strip().count("\n") == 0 and cap.err.startswith("verdict: ")
