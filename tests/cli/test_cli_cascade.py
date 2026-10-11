"""`verdict cascade`: Laya (or any primary server) answers every state; the answers it is least sure
of are asked again of a second server. Two fake servers stand in, told apart by URL, and record what
each was sent: the point is that the second one sees only what was escalated."""
from __future__ import annotations

import io
import json

import pytest

from verdict import cli, client

PRIMARY = "http://127.0.0.1:8799"
SECOND = "http://127.0.0.1:9999"
BANK = json.dumps({
    "spam": {"type": "noul", "instructions": "Is `text` spam?"},
    "dept": {"type": "choice", "instructions": "Which team handles `text`?",
             "criteria": {"billing": "", "tech": ""}},
})
#: per row i: P(spam), P(billing) from the primary. Rows 1 and 3 are unsure about spam, 2 and 3 about dept.
TABLE = [(0.99, 0.95), (0.55, 0.90), (0.97, 0.52), (0.45, 0.48)]


@pytest.fixture
def servers(monkeypatch):
    seen = {"primary": [], "second": []}

    def decide(state, questions, url=None, model=None, **flags):
        i = int(state["text"].split()[-1])
        out = {}
        if url == SECOND:
            seen["second"].append((i, sorted(questions)))
            for q, spec in questions.items():
                out[q] = ({"type": "noul", "noul": 0.01} if spec["type"] == "noul" else
                          {"type": "choice", "choice": "tech", "probabilities": {"billing": 0.1, "tech": 0.9}})
            return {"model": "second", "answers": out}
        seen["primary"].append(i)
        spam, billing = TABLE[i]
        for q, spec in questions.items():
            keys = list(spec["criteria"])
            if keys == ["no", "yes"]:                     # a yes/no, asked as the no/yes choice
                out[q] = {"type": "choice", "choice": "yes" if spam >= .5 else "no",
                          "probabilities": {"no": 1 - spam, "yes": spam}, "confidence": 0.5}
            else:
                out[q] = {"type": "choice", "choice": "billing" if billing >= .5 else "tech",
                          "probabilities": {"billing": billing, "tech": 1 - billing}, "confidence": 0.5}
        return {"model": "primary", "answers": out}

    monkeypatch.setattr(client, "decide", decide)
    return seen


def stdin(monkeypatch, n=4):
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(json.dumps({"text": f"msg {i}"}) + "\n" for i in range(n))))


def run(argv, capsys):
    code = cli.main(["cascade", "-q", BANK, "--escalate", SECOND, "--server-only", *argv])
    out = capsys.readouterr()
    return code, [json.loads(l) for l in out.out.splitlines() if l.startswith("{")], out.err


def test_rows_say_which_tier_answered_and_only_the_escalated_reach_the_second_server(servers, monkeypatch, capsys):
    stdin(monkeypatch)
    code, rows, err = run(["--below", "0.6"], capsys)
    assert code == 0 and [r["index"] for r in rows] == [0, 1, 2, 3]
    assert servers["second"] == [(1, ["spam"]), (2, ["dept"]), (3, ["dept", "spam"])]
    assert rows[0]["escalated"] == [] and rows[0]["answers"]["spam"]["tier"] == "primary"
    assert rows[1]["answers"]["spam"]["tier"] == "escalated" and rows[1]["answers"]["spam"]["noul"] == 0.01
    assert rows[1]["answers"]["dept"]["tier"] == "primary"
    assert 0 not in {i for i, _ in servers["second"]}


def test_the_summary_says_how_much_left_the_machine_and_where_it_went(servers, monkeypatch, capsys):
    stdin(monkeypatch)
    _, _, err = run(["--below", "0.6"], capsys)
    assert "4 of 8" in err and "3 of 4 states" in err and SECOND in err


def test_a_share_sends_that_fraction_of_each_question(servers, monkeypatch, capsys):
    stdin(monkeypatch)
    code, rows, _ = run(["--share", "0.5"], capsys)
    assert code == 0 and sum(len(r["escalated"]) for r in rows) == 4


def test_the_default_is_a_share_not_nothing(servers, monkeypatch, capsys):
    stdin(monkeypatch)
    code, rows, _ = run([], capsys)
    assert code == 0 and sum(len(r["escalated"]) for r in rows) > 0


def test_share_and_below_together_are_a_usage_error(servers, monkeypatch, capsys):
    stdin(monkeypatch)
    code, _, err = run(["--share", "0.2", "--below", "0.6"], capsys)
    assert code == 2 and "share" in err and servers["second"] == []


def test_a_second_server_that_is_down_keeps_the_primary_answers_and_exits_one(servers, monkeypatch, capsys):
    stdin(monkeypatch)
    real = client.decide

    def flaky(state, questions, url=None, model=None, **flags):
        if url == SECOND:
            raise client.NoServer("no verdict server at the second url")
        return real(state, questions, url=url, model=model, **flags)

    monkeypatch.setattr(client, "decide", flaky)
    code, rows, err = run(["--below", "0.6"], capsys)
    assert code == 1 and len(rows) == 4
    assert rows[1]["answers"]["spam"]["tier"] == "primary" and "escalation_error" in rows[1]
    assert "could not be escalated" in err


def test_a_malformed_escalate_url_is_named(servers, monkeypatch, capsys):
    stdin(monkeypatch)
    code = cli.main(["cascade", "-q", BANK, "--escalate", "not-a-url", "--server-only"])
    assert code == 2 and "--escalate" in capsys.readouterr().err


# ---- --check: does escalating help for THIS task? --------------------------------------------

def labelled(tmp_path):
    rows = [{"state": {"text": f"msg {i}"}, "labels": lab} for i, lab in enumerate([
        {"spam": True, "dept": "billing"}, {"spam": True, "dept": "billing"},
        {"spam": False, "dept": "billing"}, {"spam": False, "dept": "tech"}])]
    path = tmp_path / "labelled.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_check_reports_each_question_and_says_when_escalating_would_not_help(servers, tmp_path, capsys):
    code = cli.main(["cascade", "-q", BANK, "--escalate", SECOND, "--server-only", "--share", "0.5",
                     "--check", labelled(tmp_path), "--json"])
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and set(report) == {"spam", "dept"}
    assert report["dept"]["helps"] is False and report["dept"]["n"] == 4
    assert {"primary", "cascade", "secondary", "escalated"} <= set(report["dept"])


def test_check_in_words_names_the_questions_that_would_not_help(servers, tmp_path, capsys):
    cli.main(["cascade", "-q", BANK, "--escalate", SECOND, "--server-only", "--check", labelled(tmp_path)])
    out = capsys.readouterr()
    assert "would not help" in out.out + out.err and "dept" in out.out + out.err


def test_the_command_has_help_with_an_example_and_its_exit_status(capsys):
    with pytest.raises(SystemExit):
        cli.main(["cascade", "--help"])
    out = capsys.readouterr().out
    assert "Example" in out and "Exit status" in out and "FINDINGS" in out


# ---- a fitted cut: --calibration to use one, --check --fit to find one ---------------------------

SPAM_BANK = json.dumps({"spam": {"type": "noul", "instructions": "Is `text` spam?"}})


@pytest.fixture
def many(monkeypatch):
    """20 states. The primary scores spam 0.40 for even i and 0.05 for odd, so a cut at 0.5 calls all
    of them no; the second server is right on every one."""
    seen = []

    def decide(state, questions, url=None, model=None, **flags):
        i = int(state["text"].split()[-1])
        if url == SECOND:
            seen.append(i)
            return {"model": "second", "answers": {"spam": {"type": "noul", "noul": 0.95 if i % 2 == 0 else 0.02}}}
        p = 0.40 if i % 2 == 0 else 0.05
        return {"model": "primary", "answers": {"spam": {"type": "choice", "choice": "no",
                "probabilities": {"no": 1 - p, "yes": p}, "confidence": 0.5}}}

    monkeypatch.setattr(client, "decide", decide)
    return seen


def labelled_many(tmp_path, n=20):
    path = tmp_path / "many.jsonl"
    path.write_text("".join(json.dumps({"state": {"text": f"msg {i}"}, "label": i % 2 == 0}) + "\n"
                            for i in range(n)))
    return str(path)


def write_fit(tmp_path, cut):
    path = tmp_path / "fit.json"
    path.write_text(json.dumps({"questions": {"spam": {"type": "noul", "cut": cut}}}))
    return str(path)


def test_calibration_makes_the_unsure_the_ones_nearest_the_cut(servers, monkeypatch, tmp_path, capsys):
    fit = write_fit(tmp_path, 0.97)          # row 2's spam score is 0.97: it sits on the cut
    stdin(monkeypatch)
    code, rows, _ = cli.main(["cascade", "-q", BANK, "--escalate", SECOND, "--server-only", "--share",
                              "0.25", "--calibration", fit]), None, None
    out = capsys.readouterr().out
    rows = [json.loads(l) for l in out.splitlines() if l.startswith("{")]
    assert code == 0
    assert [i for i, r in enumerate(rows) if "spam" in r["escalated"]] == [2]
    assert rows[0]["answers"]["spam"]["decision"] is True and rows[0]["answers"]["spam"]["cut"] == 0.97


def test_a_fitted_cut_and_below_are_a_usage_error(servers, monkeypatch, tmp_path, capsys):
    stdin(monkeypatch)
    code, _, err = run(["--below", "0.6", "--calibration", write_fit(tmp_path, 0.5)], capsys)
    assert code == 2 and "share" in err and servers["second"] == []


def test_fit_needs_check(servers, monkeypatch, capsys):
    stdin(monkeypatch)
    code, _, err = run(["--fit"], capsys)
    assert code == 2 and "--check" in err


def test_check_fit_finds_the_cut_reports_the_held_out_part_and_saves_the_fit(many, tmp_path, capsys):
    out_file = tmp_path / "fit.json"
    code = cli.main(["cascade", "-q", SPAM_BANK, "--escalate", SECOND, "--server-only", "--share", "0.5",
                     "--check", labelled_many(tmp_path), "--fit", "--out", str(out_file), "--json"])
    report = json.loads(capsys.readouterr().out)["spam"]
    assert code == 0 and report["n"] == 6, "6 of 20 are held out"
    assert 0.05 < report["cut"] < 0.40 and report["primary_raw"] < report["primary"]
    saved = json.loads(out_file.read_text())["questions"]["spam"]
    assert saved["type"] == "noul" and saved["cut"] == report["cut"]


def test_check_fit_in_words_says_what_was_held_out(many, tmp_path, capsys):
    cli.main(["cascade", "-q", SPAM_BANK, "--escalate", SECOND, "--server-only", "--share", "0.5",
              "--check", labelled_many(tmp_path), "--fit"])
    out = capsys.readouterr().out
    assert "held out" in out and "cut " in out and "at 0.5" in out


def test_a_saved_fit_is_what_calibration_reads(many, tmp_path, monkeypatch, capsys):
    out_file = tmp_path / "fit.json"
    cli.main(["cascade", "-q", SPAM_BANK, "--escalate", SECOND, "--server-only", "--share", "0.5",
              "--check", labelled_many(tmp_path), "--fit", "--out", str(out_file)])
    capsys.readouterr()
    monkeypatch.setattr("sys.stdin", io.StringIO("".join(json.dumps({"text": f"msg {i}"}) + "\n" for i in range(6))))
    code = cli.main(["cascade", "-q", SPAM_BANK, "--escalate", SECOND, "--server-only", "--share", "0.5",
                     "--calibration", str(out_file)])
    rows = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    assert code == 0 and all("decision" in r["answers"]["spam"] for r in rows)
