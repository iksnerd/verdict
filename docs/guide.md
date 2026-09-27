# Using verdict

← [Back to the README](../README.md)

A task-first guide. For the flags themselves, `verdict --help` and `verdict COMMAND --help`. For
where each claim here comes from, [FINDINGS.md](FINDINGS.md). For runnable inputs,
[examples/](../examples/README.md).

## Contents

1. [Install and update](#install-and-update)
2. [First answers: the three question types](#first-answers-the-three-question-types)
3. [Many items at once](#many-items-at-once)
4. [Turning a score into a decision](#turning-a-score-into-a-decision)
5. [Reading the numbers](#reading-the-numbers)
6. [Which tool for which job](#which-tool-for-which-job)
7. [Worked recipes](#worked-recipes)
8. [Running it without cooking the laptop](#running-it-without-cooking-the-laptop)
9. [Troubleshooting](#troubleshooting)
10. [Command reference](#command-reference)
11. [Configuration reference](#configuration-reference)
12. [Scripts and agents](#scripts-and-agents)

## Install and update

verdict runs on Apple Silicon with Python 3.11:

```sh
uv tool install --python 3.11 'verdict[mlx,laya] @ git+https://github.com/iksnerd/verdict.git@v0.5.0'
verdict init      # writes ~/.config/verdict/config.toml
verdict --version
```

verdict answers with base Laya (`aac6fef/laya-mlx`), which downloads itself from Hugging Face on
first use. The author's fine-tune is private (its labels came from Gemini) and within noise of base
Laya once a yes/no is asked as a no/yes choice (FINDINGS §40). `[model].path` points at any other
checkpoint, a Hub id or a local folder.

To update, run `verdict update`, which installs the newest release tag. `verdict update --check`
only reports, and exits 1 when a newer release is available. Each release has passed the full test suite in CI, and its answer-quality scorecard
(`verdict bench`) has not fallen below the previous release's interval.

## First answers: the three question types

Start a server once so each call costs milliseconds instead of seconds. Without one, every command
still works by loading the model itself.

```sh
verdict serve &
```

**Yes/no** gives a probability that the statement holds:

```sh
$ verdict ask "commit the fix and push it" "Is this an instruction to perform an action?"
0.79
```

**Choice** picks one of several options, with `-o NAME` or `-o NAME=description`:

```sh
$ verdict ask "I was charged twice" "What does the customer want?" -o "refund=money back" -o info -o other
refund 0.75
```

**Score** gives an expected level on an ordinal scale; list the levels lowest first with `-l`:

```sh
$ verdict ask "the app crashed again, fix it NOW" "How frustrated?" -l calm -l annoyed -l furious
1.38/2 annoyed
```

`ask` reads the text from stdin with `-`, so `git diff | verdict ask - "Does this change touch
authentication?"` works. Stop the server when you're done: `kill $(lsof -ti TCP:8799 -sTCP:LISTEN)` (your `url`'s
port). Killing by name would also stop a server another session or user started.

## Many items at once

For more than a handful, write the questions once as a **bank** and stream the items through
`decide --jsonl`:

```sh
verdict decide --jsonl -q examples/room-triage/bank.json < examples/room-triage/updates.jsonl > scored.jsonl
```

- **Give the state named fields**, and name them in the question in backticks. For example, the
  state `{"latest": "Shipped: v2.1 ..."}` with the question "Does `latest` say the work is
  finished?". This is Laya's recommended form. `ask` and `decide` refuse a state that lacks a
  field its question names (exit 2): the model would answer from nothing. That is how you catch
  a bank written for different data.
- **Put the field that matters first.** Each state is clipped to 128 tokens, which is 1.9x faster
  than reading it whole with no loss (FINDINGS §29).
- **One bank, many states.** Every question in a bank costs a forward pass, so ask the one or two
  questions you need, not eight. Eight combined scored *below* the best single one (§27).
  Packing several states into one pass doesn't help on an M3 Pro: at best 1.5x, sometimes half
  the speed (§34). The forward pass is the whole cost.
- **Watch the spread line.** At the end of a run, stderr shows each yes/no question's range, such
  as `spread done: 0.28 to 0.75`. A range under 0.1 is flagged: the order is noise, and the inputs
  are usually to blame (near-identical, or boilerplate).
- **Presets.** `verdict presets` lists Laya's built-in banks (guard, triage, moderation, router,
  email), with the field each expects. `verdict presets triage > my-bank.json` gives you one to
  edit. `guard` and `router` are refused without `--allow-unmeasured`: guard's `harm_severity`
  asks what harm the text could cause, and router's two questions are the ones at chance (§25).

## Turning a score into a decision

The scores are uncalibrated. The ordering carries signal; the absolute number doesn't. An
`rm -rf` scored 0.37 "destructive". To gate on a score, fit a cut on examples whose answer you
know:

```sh
verdict calibrate labelled.jsonl -q bank.json --out fit.json
verdict decide --jsonl -q bank.json --calibration fit.json < items.jsonl   # adds "decision"
verdict ask "$CMD" "Does \`command\` touch a secret?" --cut fit.json && echo flagged
```

Each line of `labelled.jsonl` is `{"state": ..., "label": ...}`. The fit is reported on a held-out
30% it never saw. A yes/no question gets a cut (the best balanced accuracy, which is robust to
skewed labels). A choice gets a temperature, plus recall per option and the top confusions, so you
can see which options to trust.

**Labels are often free.** Agent transcripts (Claude Code keeps them per project) record what
happened after every turn: which tools ran, and which commands were blocked or rejected.
Conventional-commit prefixes label commit types. A ticket tracker records which items were closed.
Run a secret scan before using transcripts: one in 2,348 prompts held a live credential.

## Reading the numbers

- **Rank, take the top slice, or fit a cut.** Never read 0.5 as the line.
- **Skewed labels need AUC, not accuracy.** With 91% of items in one class, always answering "no"
  scores 91% accuracy and means nothing.
- **Per-option recall beats one accuracy number.** On commit types, 48% overall hid feat at 0.70
  next to chore at 0.17 (§31).
- **Compare against a keyword baseline** before believing any score. On commit types, hand-written
  keyword rules came within 0.02 of verdict (§32).
- **Report the machine conditions with any latency.** The same 3-question call measured 31 ms on
  a quiet machine and 64 ms with other jobs on the GPU (§28).

## Which tool for which job

| the job | use | why |
|---|---|---|
| one item, or a few dozen | read them yourself | you're more accurate, and it fits in one read |
| a yes/no about what the text shows, over hundreds of items | **verdict** | matched Claude Haiku 4.5 (0.77 vs 0.73 AUC), free and local (§32) |
| anything that must stay on your machine | **verdict** | nothing leaves it |
| sorting into several categories, or anything needing world knowledge or a maintainer's judgment | a small LLM, if the data may leave the machine | Haiku 0.67 vs verdict 0.44 to 0.48 on 7 commit types (§32, §41); GitHub bug vs support question 0.59, most questions read as bugs, not fixed by two rounds of fine-tuning (§41, §45) |
| more than 20 options | a shortlist first, yes/no per option, or an LLM | options share one token budget; 77 options scored 0.43 (§41). verdict refuses past 20 |
| long documents | an LLM, or split the text | only the first 128 tokens are read; verdict warns when a state is far past that |
| consequences, difficulty, counting, dates, pulling out values | an LLM | nothing on the page answers them; the first two score at chance (§25, §29) |
| a fixed pattern (a file name, a flag) | a regex | exact, instant, and a floor for the others to beat |

**verdict wins when a task has all five of these:** many items (hundreds or more), a short text,
a judgment about what the text shows, yes/no or a few named options, and a need for a number to
rank, cut or combine, or for the data to stay local. It matches an LLM on quality there (§32) and
beats it on cost, speed and privacy: 400 items in 62 s for $0, against 403 s and $0.28. Outside
that it loses, and a good pipeline uses both: verdict over everything, an LLM on what it flags.

Ask about **what is on the page**, never about something the text can't show. "Is this an
instruction to perform an action?" predicts tool use at 0.77 AUC. "How hard is this for a language
model?" is at chance, 0.48 (§25, §26). Write the plain question first: it beat the clever one three
times out of three (§28).

**Start from a question that has been measured.** `verdict questions` lists them, each with what it
was measured on and how it did. Use them by name: `verdict decide STATE -q is_instruction` or
`-q is_instruction,touches_secret`.

**`ask` and `decide` refuse the shapes that measured at chance** and exit 2 before asking: a
question about a consequence, a difficulty or a risk ("could this cause harm", "how hard is this",
§25, §29), a yes/no about an absence ("is this ordinary", §33), and a choice with more than 20
options (§41). The error says why and how to
reword it. `--allow-unmeasured` asks anyway, with a warning. `calibrate` only warns, because
calibrating on real labels is how you find out whether such a question works for you.

**A new yes/no is asked as a choice between `no` and `yes`**, and answered as a yes/no: the
`noul` value is P(yes), and the answer carries `"asked_as": "choice"`. A plain yes/no can follow
its own labels instead of the text: "Is `text` positive?" ranked reviews at 0.79 and put every
positive one under 0.5; the same words as a no/yes choice rank at 0.96, and spam and injection
are unchanged (§33, §38). `--true`/`--false` descriptions become the two options' descriptions.
`--yesno` sends a
plain yes/no.

Naming both sides yourself is at least as good, and says what each side means:

```sh
verdict ask "$review" "Is the review positive?" -o "positive=the review is positive" -o "negative=the review is negative"
```

Never ask a yes/no about whether something is ordinary or normal. That asks about a missing
property, which the text can't show. Ask for the property itself.

## Worked recipes

**Flag the commands in a log that touch secrets** (the task in FINDINGS §29):

1. Put each command in a state: `{"command": "..."}`, one per line.
2. `verdict decide --jsonl -q examples/secret-commands/bank.json < commands.jsonl > scored.jsonl`
3. Sort by `answers.secrets.noul` and read the top slice yourself. On 400 real commands this
   question separated blocked ones at 0.77 AUC. It misses commands that write secrets without
   naming them, such as `vercel env pull`.
4. To gate automatically, label a few hundred and run `verdict calibrate` first.

**Triage a backlog of status threads:** score each thread's latest human or agent message with
`examples/room-triage/bank.json`, then read the top of each column. Filter out bot and linter posts
first: in one log a linter had written the latest message in 99 of 137 rooms (§30).

**Search by named answers instead of an opaque embedding.** Ask a few questions about each item;
the answers are a vector whose every number has a name, so you can see why something matched.

```sh
verdict decide --jsonl -q examples/ticket-search/bank.json < examples/ticket-search/tickets.jsonl > scored.jsonl
verdict rank scored.jsonl about_money=1 says_leaving=0.5 -k 5
#  1.47  about_money +1.00  says_leaving +0.47  | I was charged twice this month. Please refund ...
#  1.30  about_money +0.87  says_leaving +0.43  | Thinking about switching to a competitor, ...
```

`rank` rescales each dimension to its percentile over the file and ranks by the weighted sum; a
negative weight ranks away from a dimension (`asks_how=-1`). It does not use cosine: cosine
ignores magnitude, and on 16 tickets it put "thanks team!", which scored about 0.13 on everything,
in the top five for money trouble (§39). Each dimension is a full pass, so ten questions over
10,000 items is a batch job, not a per-query one. Named dimensions catch what you thought to ask;
for "tickets like this one", an embedding still does better.

**Tag thousands of transcript turns:** one yes/no over `{"text": ...}` states, then read the top
of the ranking. "Is this an approval (ok, yes, go ahead)?" and "Is this an instruction?" both
worked on real turns (§26 to §28).

## Running it without cooking the laptop

Loading several models at once and scoring thousands of items back to back can overheat a laptop.
The rules that came out of doing it once:

- **One model at a time.** Before `verdict serve`, check `ollama ps` and `pgrep -fl 'verdict serve'`.
- **Sample before sweeping.** 200 to 400 items settles most questions.
- **Keep the pause.** `--jsonl` and `calibrate` wait 50 ms between calls by default. Run long jobs
  under `taskpolicy -b`.
- **Stop the server afterwards.** It holds the model in GPU memory.
- **Short on memory?** `bits = 8` under `[model]` (or `VERDICT_BITS=8`, or `verdict serve --bits 8`)
  quantizes the model as it loads: about 430 MB of GPU memory instead of 800, with the same
  answers on every bench suite (§36). It is no faster, and the download stays 843 MB. 4-bit is not
  offered: it changed too many answers.
- **Non-English input:** verdict warns, because the English checkpoint stays confident while
  wrong. `--lang multi` loads Laya's multilingual checkpoint on first use (Hindi refund request:
  0.28 on the English checkpoint, 1.00 on the multilingual one). For Cyrillic it is the difference
  between chance and usable: Bulgarian reviews 0.58 against 0.93, Russian 0.64 against 0.88 (§37).
  If most of your text is not English, set `lang = "multi"` under `[model]` (or `VERDICT_LANG=multi`)
  instead of passing the flag every time.
- **More than one local checkpoint:** `[model.extra]` names further ones a request can select by
  name (`support = "org/support-mlx"`), loaded on first use like `multilingual` is. Each one that
  gets used stays resident, so naming several in one session puts that many models on the GPU —
  it is for a checkpoint you will actually use across the session, not a menu to sample from.
- **A genuinely different model, not another laya checkpoint:** `SystemOneBackend` (`docs/api.md`)
  forwards to any other `/v1/systemone` server instead of loading anything into this process — a
  local [Kev](https://github.com/jaredpalmer/kev) server, say. It runs in its own process with its
  own memory footprint (Kev-0.8B: about 1.8 GB of weights), so "one model at a time" still means
  checking what that other server is holding before you start it, the same as checking `ollama ps`
  before `verdict serve`.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| every call takes about 2 s | no server running, so each call loads the model | `verdict serve &` |
| answers are all 0.5 / "no model" | a server running the `uniform` backend (`uvicorn verdict.api:app`) | stop it; use `verdict serve` |
| `verdict` breaks after a hand-typed `uv sync` | plain `uv sync` drops the mlx and laya extras | `verdict update`, or `uv sync --extra mlx --extra laya` in the repo |
| "question 'x' asks about `field`, which the state does not have" | the bank was written for different data | rename the state's key or the backticked field |
| `spread q: 0.58 to 0.65 ... narrow range` | the inputs don't vary where the question looks, or they're boilerplate | check who wrote them; filter bots and templates |
| "... not found, so using base laya" | `[model].path` names a local checkpoint that is not there | fix the path, or remove it to use base Laya on purpose |
| "the server answered 422: ..." | the server refused the question bank; verdict reports this rather than loading a model locally | fix the bank the message names; a malformed bank fails the same way with no server running |
| "... Refused, because answers to this shape measured at chance" | the question asks about a consequence, difficulty or absence, or names a field the state lacks | reword it to ask what the text says; `--allow-unmeasured` to ask anyway, or `calibrate` it on real labels |
| "a yes/no cut needs both yes and no labels" | every labelled example in the calibration set has the same answer | label some of the other class; there is no cut to fit on one |
| "answers came from more than one model" | `calibrate` got answers from two models, such as a server that restarted on another checkpoint mid-run | run it again against one model; a cut fitted across two fits neither |
| "the state looks like es ..." on English text | Laya's language guess on short prose | `--lang en` for that run |
| "`<url>`/v1/systemone ... it may still be loading" | a `--systemone` server (Kev, Von, another verdict) hasn't answered a request yet; verdict retries this automatically and says so on stderr | wait for it, or start the server earlier; if it persists after 5 attempts, check that server's own logs -- it may have crashed on an earlier request |
| `verdict update` fails with "fatal: unable to read tree" (or `verdict` disappears from PATH afterward) | `uv`'s own git cache for this repo got into a bad state; a plain re-fetch doesn't self-heal, and `uv tool install` isn't atomic against it -- a failed reinstall can remove the old install without completing the new one | `uv cache clean verdict --force`, then re-run `verdict update` (or the `uv tool install` line from the README); confirm with `uv tool list` |

## Command reference

| command | does |
|---|---|
| `verdict ask TEXT QUESTION` | one question inline: yes/no, `-o` choice, `-l` score; `--cut` sets the exit status |
| `verdict decide STATE -q BANK` | a bank of questions about one state, as JSON; `--jsonl` for many states |
| `verdict calibrate LABELLED -q BANK` | fits a cut or temperature on labelled examples, reported held out |
| `verdict rank SCORED NAME=WEIGHT ...` | ranks a `decide --jsonl` run by a weighted sum of its rescaled answers, with each dimension's share; no model |
| `verdict docs [readme\|guide\|api\|findings\|pipeline\|routing] [N]` | prints a project document from the installed tool; `findings 38` prints one section |
| `verdict questions [NAME] [--json]` | questions measured to work, each with its result; use by name with `-q` |
| `verdict presets [NAME] [--json]` | Laya's built-in banks |
| `verdict validate -q BANK [--json]` | checks a bank with no server or model: structure, wording, truncation ([scripts and agents](#scripts-and-agents)) |
| `verdict bench [--verify CARD]` | answer quality on pinned public datasets; the release gate; `--systemone URL` scores a Jev-compatible endpoint |
| `verdict examples [NAME...]` | runs `examples/` banks against a backend, printing ms/call and, where a row carries an `expected` label, a match count against it; `--systemone URL` checks a different backend (Kev, another verdict) against the same real inputs; `--path NAME` prints an installed example's directory |
| `verdict serve` | holds the model on localhost:8799; also speaks TypeSafe's Jev protocol ([api.md](api.md)) |
| `verdict update [--check]` | installs the newest release (or pulls, in a dev checkout) |
| `verdict init` | writes the config from what the machine has |
| `verdict config [--json]` | each resolved setting and where it came from: a flag, `$VAR`, a file or the default |
| `verdict route PROMPT` | the big-or-small switch example; at chance on real traffic ([routing.md](routing.md)) |
| `verdict cases` | prints that switch: its branches, default and questions |

Exit status is 2 for a usage error everywhere. `ask --cut` exits 0 for yes and 1 for no.
`update --check` exits 1 when a newer release is available.

An error is one `verdict:` line on stderr. With `--json` or `--jsonl`, stdout also gets
`{"error": {"code": ..., "message": ...}}`, with `code` one of `usage`, `refused`, `no_server`,
`server_timeout`, `server_error`, `config` or `not_found`. `decide --jsonl` gives every output
row an `index`, the 0-based line of the input it answers, so rows pair with lines across skipped
blanks. A line that starts like JSON and does not parse stops the run at that line (earlier rows
stay valid); a plain-text line is sent as text, with a warning.

With no server answering, a command loads the model itself and says so once on stderr. A server
that accepts the connection and does not answer in time is an error (`server_timeout`), not a
reason to load a second copy of the model beside it.

## Configuration reference

Settings resolve flag > environment > `verdict.toml` (or `~/.config/verdict/config.toml`) >
built-in default. These five override a `[section].key` in that file:

| variable | overrides | default |
|---|---|---|
| `VERDICT_URL` | `[server].url`, the server every command asks | `http://127.0.0.1:8799` |
| `VERDICT_MODEL` | `[model].path`, the checkpoint to load | base Laya |
| `VERDICT_MULTILINGUAL` | `[model].multilingual`, the checkpoint `"model": "multilingual"` answers from | `aac6fef/laya-multilingual-mlx` |
| `VERDICT_BITS` | `[model].bits`, quantization at load (16 or 8, §36) | 16 |
| `VERDICT_LANG` | `[model].lang`, `auto`, `en` or `multi` | `auto` |

`VERDICT_REPO` is different: it names the release source for `verdict update`
(`cli/update.py`), not a `verdict.toml` setting, so it has no flag or config-file form. Default
`https://github.com/iksnerd/verdict.git`.

There is deliberately no `[thresholds]` key: cuts live in the server process, so a file value
would be ignored (see [init](#install-and-update)). `[model.extra]` names further checkpoints
the same way as `[model].path`, each loaded on first request that names it; it is TOML-only, with
no environment-variable form.

## Scripts and agents

Check a question bank before starting inference:

```sh
verdict validate -q examples/support-tickets/bank.json
cat examples/support-tickets/bank.json | verdict validate -q - --json
```

Validation needs no server or model. It checks the question structure, calls a bank invalid when
a question has a shape `decide` would refuse (unless `--allow-unmeasured`), and reports possible
option-truncation warnings; it does not measure answer quality. Presets require the
installed `laya-mlx` package, but do not load its model. With `--json`, stdout contains
`{"valid": true, "questions": {...}, "warnings": [...]}` or
`{"valid": false, "error": "..."}`. Exit status is 0 for valid (warnings included), 2 for invalid.

For predictable agent calls, start `verdict serve` once and use `--server-only` on `ask`, `decide`,
`calibrate` or `bench`. An unavailable server produces an error with exit status 2 instead of loading a
local fallback model. The default behavior still permits fallback.

```sh
verdict ask "Please cancel my order" "Does this request cancellation?" --json --server-only
verdict decide --jsonl -q examples/support-tickets/bank.json --server-only < examples/support-tickets/tickets.jsonl
```

Use `ask --json` for the complete answer under `answers.q`, and `decide --jsonl` for one JSON
result per input line. Diagnostics go to stderr; parse stdout as JSON. JSONL output preserves
input order and starts before EOF. If a later call fails, earlier lines remain valid; check the
process exit status before treating a batch as complete. `ask --json` exits 0 on success;
plain `ask --cut` exits 0 for yes and 1 for no. Errors exit 2.


### Calibration and backend identity

Calibration validates each question's labels before inference. A missing label on an individual
row is allowed; each fitted question needs at least two labels, choices must name valid options,
and a yes/no training split must contain both classes. `--heldout` is in `[0, 1)`; zero disables
the held-out split. Cut fitting includes constant classifiers so an inverted signal cannot force
a worse-than-baseline threshold. A constant cut is a reason to improve the question, not evidence
that it distinguishes the classes. Check the reported held-out metrics before using a fit. The
fit file's `model` is the model that answered, as the server or in-process engine named it, not
the configured one; a run whose answers came from two models is refused.

`verdict init` preserves language, quantization, and extra checkpoint settings when rewriting a
configuration, read from the `--out` file itself when it exists. Backend changes still require measurement: never transfer a calibration file or
confidence threshold from local Laya to hosted Jev without refitting. The
[SDK capability guide](api.md#discovering-backend-differences) explains model selection, context
budgets, and missing upstream confidence/usage fields. Python users can inspect
`backend.capabilities()`; HTTP users can call `/v1/capabilities` without loading weights.
