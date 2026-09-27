"""The parser. Help text lives here, next to the flags it describes, so the two cannot drift."""
from __future__ import annotations

import argparse
import os
import sys

from . import (catalog, decisions, docs_cmd, evaluation, examples_cmd, inference, routing, setup,
              support, update)
from .. import client
from ..config import ConfigError
from ..errors import QuestionError

DESCRIPTION = """\
A System 1, in Kahneman's sense: fast, automatic, intuitive judgments. Give it a state (text,
or a JSON object with named fields) and typed questions; get back a probability per option in
about 40 ms of inference, about 150 ms a call end to end against a running server. No generated
text, nothing to parse, nothing run on your behalf.

Pair it with a System 2, a person or an LLM agent, that deliberates. Let verdict make the same
quick judgment over hundreds of items, and spend slow thought where it flags something. Like
any System 1 it is confident whether or not it is right, so check it against labels
(`verdict calibrate`) before trusting a number.

Question types, as laya defines them:
  noul    yes/no            -> probability it holds, 0 to 1
  choice  pick one option   -> the option, with a probability for each
  score   ordinal levels    -> expected level, with a probability for each

Start a server first so each call costs milliseconds instead of seconds; without one, every
command still works by loading the model itself.
"""

EPILOG = """\
Examples:
  verdict serve &                                                   # load the model once
  verdict ask "commit the fix and push it" "Is this an instruction to perform an action?"
  verdict ask "$MSG" "What does the user want?" -o refund -o info -o other
  verdict ask "$TICKET" "How urgent?" -l "not urgent" -l soon -l critical
  verdict questions                                                 # measured questions, by name
  verdict validate -q bank.json --json                              # check without inference
  verdict decide '{"command": "cat .env"}' --questions touches_secret
  verdict decide '{"post": "buy followers now!!"}' --questions moderation
  verdict decide --jsonl --questions bank.json < states.jsonl > answers.jsonl
  verdict rank answers.jsonl about_money=1 says_leaving=0.5          # search by named answers
  verdict calibrate labelled.jsonl --questions bank.json --out fit.json
  verdict docs findings 38                                          # read a cited section

The scores are uncalibrated: the ordering is trustworthy, the absolute numbers are not. Fit a
cut on your own labelled examples (`verdict calibrate`) before gating anything on one.

Ask about what the text says. `ask` and `decide` refuse (exit 2) the shapes that scored at chance:
a question about a consequence, difficulty or risk (FINDINGS §25, §29), a yes/no about an absence
("is this ordinary", §33), and a state missing a field the question names. --allow-unmeasured
asks anyway. A new yes/no is asked as a choice between no and yes and answered as a yes/no (§38).

Settings: flag > environment ($VERDICT_URL, $VERDICT_MODEL, $VERDICT_MULTILINGUAL, $VERDICT_BITS,
$VERDICT_LANG) > ./verdict.toml > ~/.config/verdict/config.toml > built-in. `verdict init` writes
one; `verdict config` shows each resolved value and where it came from.

Errors: one `verdict:` line on stderr and exit 2. With --json or --jsonl, also
{"error": {"code", "message"}} on stdout; codes: usage, refused, no_server, server_timeout,
server_error, config, not_found.
"""

ASK_HELP = """\
Ask one question about one text, inline, and print one line.

  no -o or -l      yes/no; prints the probability (0.72), or yes/no with --cut
  -o A -o B ...    choice; prints the top option and its probability (refund 0.61)
  -l LOW -l ...    score, levels lowest first; prints expected level and top level (1.30/2 high)

A new yes/no is asked as a choice between no and yes and printed as P(yes): the plain form ranked
positive reviews at 0.79 with none over 0.5, the no/yes choice at 0.96 (FINDINGS §33, §38).
On the fine-tune, measured questions (`verdict questions`) are asked as measured; --yesno sends
a plain yes/no.

A question about a consequence, difficulty or risk ("could this cause harm") or a yes/no about an
absence ("is this ordinary") is refused, exit 2: those scored at chance. Ask what the text says or
does instead; --allow-unmeasured asks anyway.
"""

ASK_EPILOG = """\
Examples:
  verdict ask "why is the build so slow" "Does this ask why or how something works?"
  verdict ask "rm -rf data/" "Does this command delete files or folders?" --cut 0.55 && echo stop
  verdict ask "$MSG" "What does the customer want?" -o "refund=money back" -o info -o other
  verdict ask "$REVIEW" "Is the review positive?" -o "positive=it is positive" -o "negative=it is negative"
  git diff | verdict ask - "Does this change touch authentication?"
  verdict ask "Der Kunde wurde zweimal belastet" "Is this a billing issue?" --lang multi

Exit status: 0, or with --cut 0 for yes and 1 for no; 2 for a usage error or a refused question.
With --json, successful calls always exit 0. --server-only fails instead of loading locally.
"""

DECIDE_HELP = """\
Ask a bank of typed questions of a state, all in one pass, and print JSON.

The state is best a JSON object with named fields: laya serializes it as JSON, and questions
refer to fields in backticks ("What does the customer want in `message`?").

It refuses (exit 2) a state missing a field a question names, since the model would answer from
nothing, and the question shapes that scored at chance: a consequence, difficulty or risk, or a
yes/no about an absence. --allow-unmeasured asks anyway. A new yes/no is asked as a choice
between no and yes and answered as a yes/no, P(yes) (FINDINGS §38); --yesno opts out.
"""

DECIDE_EPILOG = """\
Examples:
  verdict decide '{"message": "charged twice, refund today or we cancel"}' --questions triage
  verdict decide @ticket.json --questions bank.json --calibration fit.json
  echo '{"command": "git push --force"}' | verdict decide - --questions flags.json
  verdict decide --jsonl --questions moderation < posts.jsonl
  verdict decide --jsonl --questions is_instruction,is_approval < turns.jsonl

--questions takes a preset (triage, moderation, email; guard and router ask about consequences
and difficulty, so they need --allow-unmeasured), measured questions by name
(`verdict questions`, comma-separated: is_instruction,touches_secret), inline JSON, or a file.
A bank looks like:
  {"urgent": {"type": "noul",   "instructions": "Is `message` urgent?"},
   "intent": {"type": "choice", "instructions": "What does `message` want?",
              "criteria": {"refund": "money back", "info": "a question", "other": "none fit"}},
   "anger":  {"type": "score",  "instructions": "How angry is `message`?",
              "criteria": ["calm", "annoyed", "angry"]}}
A choice with more than 20 options is refused: they share a 192-token budget (77 options scored
0.43, FINDINGS §41). Each option is also cut at 48 tokens without an error (--check reports where).

With --jsonl over 10 or more states, each yes/no question's range is printed to stderr, and a
range under 0.1 is flagged: an ordering squeezed that tight ranks almost nothing, and the cause
is usually the inputs (near-identical, or boilerplate), not the model.

Exit status: 0, or 2 for a usage error or a refused question.
Use `verdict validate -q bank.json --json` to check a bank without inference.
Add --server-only to require a running server. On a batch error, earlier output lines remain valid.
"""

CALIBRATE_HELP = """\
Fit the numbers you gate on, from your own labelled examples, and report them on a held-out
split the fit never saw.

  noul    a cut: the threshold with the best balanced accuracy (robust to skewed labels)
  choice  a temperature: one scalar that makes the probabilities mean what they say

Each line of EXAMPLES is {"state": ..., "label": ...} for a one-question bank, or
{"state": ..., "labels": {"question": ...}}. Yes/no labels take true/false, yes/no or 1/0.
"""

CALIBRATE_EPILOG = """\
Examples:
  verdict calibrate blocked.jsonl --questions touches_secret --out secret.fit.json
  verdict decide '{"command": "vercel env rm KEY production"}' --questions touches_secret --calibration secret.fit.json
  verdict ask "$CMD" "Does this read, print or change a secret?" --cut secret.fit.json

It asks the model once per example, with a pause between calls, and stops at --limit (400 by
default): a few hundred labels settle most questions, and a sweep of thousands holds the GPU
flat out for minutes. Fewer than 10 examples is refused.

Exit status: 0, or 2 for a usage error, too few examples, or labels a cut cannot be fitted on
(one class only, in the whole set or in the training split).
"""

PRESETS_EPILOG = """\
Examples:
  verdict presets                        # every bank, the state field it expects, its questions
  verdict presets triage > my-bank.json  # a bank as JSON, to edit and pass to --questions
"""

ROUTE_EPILOG = """\
Examples:
  verdict route "what is 17% of 340"
  verdict route "$PROMPT" --quiet        # prints small or big
  verdict route --batch < prompts.txt

At chance on real agent traffic (FINDINGS §25): kept as the worked example of a switch, not as
a router to trust. `verdict ask` with a surface question is the better tool.

Exit status: 0 for small, 1 for big, 2 for an error.
"""

SERVE_EPILOG = """\
Examples:
  verdict serve                  # binds server.url, 127.0.0.1:8799 by default
  verdict serve --port 8800

Loads one checkpoint, and the multilingual one only when a call asks for it (--lang multi).
If model.path names a local checkpoint that is missing, it loads base laya instead and says so. Localhost only, no auth. Stop it when you are done; it
holds the model in GPU memory.

It also speaks TypeSafe Jev's protocol (POST /v1/systemone, GET /v1/models), so Jev's SDKs and
tools run against it with TYPESAFE_BASE_URL=http://127.0.0.1:8799 and any TYPESAFE_API_KEY.
"""


def _formatter(argv: list[str]):
    """rich-argparse only when help is being printed. It costs about 4 ms to import, and every
    other invocation should not pay for colour it never shows."""
    if not argv or any(a in ("-h", "--help") for a in argv):
        from rich_argparse import RawDescriptionRichHelpFormatter

        # Keep group titles as written; the default title-cases "question shape (default:
        # yes/no)" into "Question Shape (Default: Yes/No)".
        RawDescriptionRichHelpFormatter.group_name_formatter = str.capitalize
        return RawDescriptionRichHelpFormatter
    return argparse.RawDescriptionHelpFormatter


def _seconds(text: str) -> float:
    value = float(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{text} is negative; a pause is 0 or more seconds")
    return value


class _Parser(argparse.ArgumentParser):
    """argparse, plus "did you mean" on a mistyped command or choice."""

    def error(self, message):
        import re

        m = re.search(r"invalid choice: '([^']*)' \(choose from (.*)\)", message)
        if m:
            choices = [c.strip().strip("'") for c in m.group(2).split(",")]
            message += support._suggest(m.group(1), choices, end="")
        super().error(message)


def _quiet_dependencies() -> None:
    """Keep other packages' chatter off stderr, where an agent reads verdict's own warnings.
    laya-mlx warns on every load that the checkpoint's temperatures are clamped (verdict's docs
    already say confidences are uncalibrated), and the Hub nags about tokens while downloading."""
    import warnings

    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", message="laya-mlx: this checkpoint ships temperatures",
                            category=RuntimeWarning)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    _quiet_dependencies()
    fmt = _formatter(argv)
    # allow_abbrev=False: `decide --json` (a guess) used to expand to --jsonl and wait on stdin.
    ap = _Parser(prog="verdict", allow_abbrev=False, description=DESCRIPTION, epilog=EPILOG,
                                 formatter_class=fmt)
    ap.add_argument("-V", "--version", action="version", version=support._version())
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND", title="Commands")

    def command(name, summary, description=None, epilog=None):
        return sub.add_parser(name, help=summary, description=description or summary,
                              epilog=epilog, formatter_class=fmt, allow_abbrev=False)

    def yesno_flag(p):
        p.add_argument("--yesno", action="store_true",
                       help="send a new yes/no question as a plain yes/no; by default it is asked "
                            "as a choice between no and yes, which ranked as well or better on "
                            "every set measured (FINDINGS §38), and answered as a yes/no")

    def quality_flag(p):
        p.add_argument("--allow-unmeasured", action="store_true",
                       help="ask a question shaped like ones that measured at chance (about a "
                            "consequence, difficulty or absence), or about a field the state "
                            "lacks; refused without this")

    def server_flags(p):
        g = p.add_argument_group("Model and server")
        g.add_argument("--lang", choices=("auto", "en", "multi"), default=None,
                       help="auto warns on non-English state; multi uses laya's multilingual "
                            "checkpoint (loaded on first use); en never checks. Overrides "
                            "model.lang (default auto)")
        g.add_argument("--url", help="server to ask; overrides server.url and $VERDICT_URL")
        g.add_argument("--server-only", action="store_true",
                       help="fail if the server is unavailable; never fall back to local inference")
        g.add_argument("--model", help="checkpoint to load when no server answers; "
                                       "overrides model.path (a missing model.path falls back "
                                       "to base laya with a warning; a missing --model fails)")

    v = command("validate", "check a question bank without loading a model",
                "Validate a bank: its structure, and the question shapes `decide` refuses because "
                "they scored at chance (a consequence, difficulty or risk; a yes/no about an "
                "absence). Possible option truncation is a warning. No inference, downloads or "
                "server required. This checks structure, not prediction quality.",
                "Examples:\n  verdict validate -q bank.json\n"
                "  cat bank.json | verdict validate -q - --json\n\n"
                "Exit status: 0 for valid (truncation warnings included), 2 for invalid or "
                "refused (--allow-unmeasured accepts the refused shapes). With --json, "
                "stdout contains valid plus questions/warnings, or valid=false plus error.")
    v.add_argument("-q", "--questions", required=True, metavar="PRESET|JSON|FILE|-",
                   help="bank source, measured question names, or - for stdin")
    v.add_argument("--json", action="store_true", help="print a machine-readable validation result")
    quality_flag(v)
    v.set_defaults(fn=catalog._validate_cmd)

    a = command("ask", "one question about a text, inline: yes/no, choice or score",
                ASK_HELP, ASK_EPILOG)
    a.add_argument("text", help="the text to judge; a JSON object is sent as a structured "
                                "state; @FILE reads a file; - reads stdin")
    a.add_argument("question", help="what to ask about it, as a plain question")
    shape = a.add_argument_group("Question shape (default: yes/no)")
    shape.add_argument("-o", "--option", dest="options", action="append", metavar="NAME[=DESC]",
                       help="a choice option; give two or more")
    shape.add_argument("-l", "--level", dest="levels", action="append", metavar="LEVEL",
                       help="a score level, lowest first; give two or more")
    shape.add_argument("--true", metavar="DESC", help="describe the yes side of a yes/no; measured "
                       "to lower AUC (FINDINGS §22), so prefer -o with both sides named")
    shape.add_argument("--false", metavar="DESC", help="describe the no side of a yes/no; see "
                       "--true")
    out = a.add_argument_group("Output")
    out.add_argument("--cut", metavar="NUM|FILE",
                     help="yes/no threshold, or a `verdict calibrate` file; prints yes/no and "
                          "sets the exit status")
    out.add_argument("--json", action="store_true", help="print the full answer as JSON")
    quality_flag(a)
    yesno_flag(a)
    server_flags(a)
    a.set_defaults(fn=decisions._ask_cmd)

    d = command("decide", "a bank of typed questions about a state, as JSON",
                DECIDE_HELP, DECIDE_EPILOG)
    d.add_argument("state", nargs="?", help="a JSON object, plain text, @FILE, or - for stdin")
    d.add_argument("-q", "--questions", metavar="PRESET|JSON|FILE",
                   help="the bank: a preset, measured question names (comma-separated), inline "
                        "JSON or a file; omit to read it from stdin")
    d.add_argument("--jsonl", action="store_true",
                   help="one state per stdin line against one bank; prints NDJSON")
    d.add_argument("--calibration", metavar="FILE",
                   help="a `verdict calibrate` file: adds a decision to each yes/no and "
                        "calibrated probabilities to each choice")
    d.add_argument("--pause", type=_seconds, default=0.05, metavar="SEC",
                   help="between --jsonl calls, so a long run leaves the machine usable "
                        "(default: 0.05)")
    d.add_argument("--check", action="store_true",
                   help="count option tokens exactly to report truncation (about 1.3 s)")
    quality_flag(d)
    yesno_flag(d)
    server_flags(d)
    d.set_defaults(fn=decisions._decide_cmd)

    rk = command("rank", "rank a decide --jsonl run by named, weighted answers",
                 "Each question is a named dimension: a yes/no is P(yes), a choice gives NAME.OPTION "
                 "per option, a score its place on the scale. Each dimension is rescaled to its "
                 "percentile over the file, then ranked by the weighted sum, with what each "
                 "dimension contributed. A weighted sum, not cosine: cosine matched an item that "
                 "scored low on everything (FINDINGS §39). No model is loaded.",
                 "Example:\n  verdict decide --jsonl -q bank.json < tickets.jsonl > scored.jsonl\n"
                 "  verdict rank scored.jsonl about_money=1 says_leaving=0.5 -k 5\n"
                 "  verdict rank scored.jsonl asks_how=-1   # away from how-to questions\n\n"
                 "Exit status: 0, or 2 for a usage error, an unknown dimension, or a file that "
                 "is not `decide --jsonl` output.")
    rk.add_argument("scored", help="the output of `verdict decide --jsonl`")
    rk.add_argument("weights", nargs="+", metavar="NAME=WEIGHT",
                    help="a weight per dimension; negative ranks away from it")
    rk.add_argument("-k", type=int, default=10, help="how many to print (default: 10; 0 for all)")
    rk.add_argument("--json", action="store_true",
                    help="one JSON line per result: state, score, contributions")
    rk.set_defaults(fn=evaluation._rank_cmd)

    dc = command("docs", "read the README, guide, API notes, FINDINGS, pipeline or routing",
                 "Print a project document to stdout. The help and error messages cite FINDINGS "
                 "sections (§25, §38): `verdict docs findings 38` prints one. No model is loaded.",
                 "Examples:\n  verdict docs                  # the README\n"
                 "  verdict docs guide | less\n  verdict docs findings 38      # one section\n\n"
                 "Exit status: 0, or 2 for an unknown section.")
    dc.add_argument("topic", nargs="?", default="readme", choices=list(docs_cmd.TOPIC_NAMES),
                    help="readme (default), guide, api or findings")
    dc.add_argument("section", nargs="?", type=int, help="with findings: one section, by number")
    dc.set_defaults(fn=docs_cmd._docs_cmd)

    qs = command("questions", "questions measured to work, usable by name with -q")
    qs.add_argument("name", nargs="?", help="print this question as a JSON bank")
    qs.add_argument("--json", action="store_true",
                    help="every question as JSON: name -> {question, state, measured, source}")
    qs.set_defaults(fn=catalog._questions_cmd)

    p = command("presets", "laya's built-in question banks", epilog=PRESETS_EPILOG)
    p.add_argument("name", nargs="?", help="print this bank as JSON")
    p.add_argument("--json", action="store_true", help="every preset as JSON: name -> bank")
    p.set_defaults(fn=catalog._presets_cmd)

    c = command("calibrate", "fit a cut or temperature on your own labelled examples",
                CALIBRATE_HELP, CALIBRATE_EPILOG)
    c.add_argument("examples", help="JSONL of labelled states")
    c.add_argument("-q", "--questions", required=True, metavar="PRESET|JSON|FILE",
                   help="the bank to fit")
    c.add_argument("--out", metavar="FILE", help="write the fit here instead of printing it")
    c.add_argument("--limit", type=int, default=400, help="at most this many examples, "
                   "sampled with --seed (default: 400; 0 for all)")
    c.add_argument("--heldout", type=float, default=0.3, help="fraction held out (default: 0.3)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--pause", type=_seconds, default=0.05, metavar="SEC",
                   help="between calls (default: 0.05)")
    yesno_flag(c)
    server_flags(c)
    c.set_defaults(fn=evaluation._calibrate_cmd)

    b = command("bench", "score answer quality on pinned public datasets; gate a release",
                "Score each suite's question on a fixed, balanced sample of a public labelled "
                "dataset, pinned by revision: AUC for yes/no, accuracy for choice, each with a "
                "95% bootstrap interval. --verify compares a scorecard with the previous one, "
                "with no model.",
                """Examples:
  verdict bench                                    # every suite, print the table
  verdict bench --suite sst2 --per-class 50        # one suite, a quick look
  verdict bench --out bench/scorecards/vX.Y.Z.json # the release scorecard
  verdict bench --verify bench/scorecards/vX.Y.Z.json

Needs the bench extra (uv sync --extra mlx --extra laya --extra bench) and, once, the network to
download the datasets. A regression is a value below the previous scorecard's interval.

Exit status: 0; with --verify, 1 when any suite regressed (the release workflow gates on this);
2 for a usage error.
""")
    b.add_argument("--suite", action="append", metavar="NAME", help="only this suite (repeatable)")
    b.add_argument("--per-class", type=int, metavar="N", help="override each suite's sample size")
    b.add_argument("--out", metavar="FILE", help="write the scorecard here")
    b.add_argument("--verify", metavar="SCORECARD",
                   help="compare this scorecard with the newest older one beside it; no model")
    b.add_argument("--against", metavar="SCORECARD", help="with --verify, compare with this one")
    b.add_argument("--pause", type=_seconds, default=0.05, metavar="SEC",
                   help="between calls (default: 0.05)")
    b.add_argument("--systemone", metavar="URL",
                   help="score a TypeSafe-compatible /v1/systemone endpoint instead of verdict: "
                        "https://api.typesafe.ai for Jev (needs TYPESAFE_API_KEY), or a "
                        "`verdict serve` URL")
    b.add_argument("--systemone-model", default="jev-latest", metavar="NAME",
                   help="the model field sent with --systemone (default: jev-latest)")
    server_flags(b)
    b.set_defaults(fn=evaluation._bench_cmd)

    ex = command("examples", "run examples/ banks against a backend",
                "Runs one or more examples/ banks against a backend and prints each row's "
                "answers plus ms/call, the same real inputs examples/README.md's own numbers "
                "came from. Where a row carries an `expected` label (most banks; examples/README.md "
                "lists them), also prints how many answers match it, so a new backend gets a real accuracy and latency number "
                "instead of one-off questions and an eyeballed guess.",
                """Examples:
  verdict examples                                       # every example, the served checkpoint
  verdict examples citation-check checklist-check        # just these two
  verdict examples citation-check --systemone http://127.0.0.1:8009 --systemone-model kev-latest

Needs a checkout (or an install that packs examples/) to find the banks.
""")
    ex.add_argument("name", nargs="*", metavar="NAME", help="an examples/ directory name; omit for all")
    ex.add_argument("--path", action="store_true",
                    help="print each named example's directory instead of running it, so an "
                         "installed copy's bank can be passed to -q")
    ex.add_argument("--pause", type=_seconds, default=0.05, metavar="SEC",
                    help="between calls (default: 0.05)")
    ex.add_argument("--systemone", metavar="URL",
                    help="score with a TypeSafe-compatible /v1/systemone endpoint instead of "
                         "verdict: a `verdict serve`, a Kev server, or https://api.typesafe.ai "
                         "(needs TYPESAFE_API_KEY)")
    ex.add_argument("--systemone-model", default="jev-latest", metavar="NAME",
                    help="the model field sent with --systemone (default: jev-latest)")
    quality_flag(ex)
    yesno_flag(ex)
    server_flags(ex)
    ex.set_defaults(fn=examples_cmd._examples_cmd)

    r = command("route", "big model or small one, the switch example",
                "Route a prompt to a big or a small model: two questions and a fitted switch.",
                ROUTE_EPILOG)
    r.add_argument("prompt", nargs="?", help="omit with --batch")
    r.add_argument("--model", help="overrides model.path")
    r.add_argument("--budget", type=int, help="prompt tokens the model sees; overrides config")
    r.add_argument("-q", "--quiet", action="store_true", help="print only the branch name")
    r.add_argument("--json", action="store_true", help="print a JSON object")
    r.add_argument("--url", help="verdict server to ask; overrides server.url and $VERDICT_URL")
    r.add_argument("--no-server", action="store_true",
                   help="always load the model in-process, never ask a server")
    r.add_argument("--batch", action="store_true",
                   help="read prompts from stdin, one per line; print NDJSON")
    r.set_defaults(fn=routing._route_cmd)

    cs = command("cases", "show the route switch: its cases, default and questions")
    cs.set_defaults(fn=routing._cases_cmd)

    s = command("serve", "hold the model in memory so every call is fast",
                "Serve the model on localhost so each call costs milliseconds, not seconds.",
                SERVE_EPILOG)
    s.add_argument("--model", help="overrides model.path")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, help="overrides the port in server.url")
    s.add_argument("--budget", type=int, metavar="TOKENS",
                   help="tokens of each state the model reads (default: model.prompt_token_budget, "
                        "128); 0 reads it whole. Fewer is faster, and intent is usually up front")
    s.add_argument("--bits", type=int, choices=(16, 8),
                   help="8 quantizes the model at load: about half the GPU memory, the same "
                        "answers on every bench suite (FINDINGS §36); overrides model.bits")
    s.add_argument("--log-level", default="warning")
    s.set_defaults(fn=setup._serve_cmd)

    u = command("update", "install the newest release (or pull, in a dev checkout)",
                "Update verdict. Installed with `uv tool install`, it reinstalls the newest vX.Y.Z "
                "release tag, with the mlx and laya extras. Run from a git checkout, it "
                "fast-forwards the checkout and re-syncs with both extras instead.",
                """Examples:
  verdict update           # newest release (or pull + sync in a checkout), and what changed
  verdict update --check   # report only: exit 0 up to date, 1 behind

Releases come from $VERDICT_REPO (default https://github.com/iksnerd/verdict.git) and are the
tags the release workflow tested. In a checkout it refuses on uncommitted changes and only
fast-forwards. A running `verdict serve` keeps the old code until restarted.

Exit status: 0 on success, 2 on failure. --check exits 0 when up to date and 1 when a newer
release (or, in a checkout, newer commits) is available.
""")
    u.add_argument("--check", action="store_true", help="report how far behind, change nothing")
    u.set_defaults(fn=update._update_cmd)

    cf = command("config", "show each resolved setting and where it came from",
                 "Show every setting as the CLI resolves it (flag > environment > ./verdict.toml "
                 "> ~/.config/verdict/config.toml > built-in), and which of those it came from.")
    cf.add_argument("--json", action="store_true", help="print {settings: [{key, value, source}]}")
    cf.set_defaults(fn=setup._config_cmd)

    i = command("init", "write a config from what this machine has")
    i.add_argument("--out", default=str(support._user_config()),
                   help="where to write (default: the user config, read from every directory; "
                        "./verdict.toml overrides it per project)")
    i.add_argument("-y", "--yes", action="store_true", help="accept every default, no prompts")
    i.set_defaults(fn=setup._init_cmd)

    args = ap.parse_args(argv)
    # decide prints JSON whatever its flags, so its failures are JSON too.
    support.JSON_ERRORS = bool(getattr(args, "json", False) or getattr(args, "jsonl", False)
                               or args.cmd == "decide")
    try:
        return args.fn(args)
    except (FileNotFoundError, QuestionError, client.ServerError, client.NoServer,
            client.ServerTimeout, ConfigError) as e:
        return support._fail(e)
    except BrokenPipeError:
        # The reader quit early (`verdict docs | head`): normal use, not an error. Point stdout at
        # /dev/null so the interpreter's own flush at exit does not raise again (Python docs,
        # "Note on SIGPIPE").
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
        except (AttributeError, OSError, ValueError):
            pass
        return 0
