---
name: verdict
description: >-
  Score many texts at once with the local `verdict` CLI (a Laya encoder, ~30 ms of
  inference and ~0.1 s a call against a running server, no generated text): tag hundreds of transcript turns, log lines, commands,
  issues or files with a yes/no, choice or ordinal answer without reading them all
  into context, rank or filter a JSONL, or run Laya's presets (triage,
  moderation, email). Use when a task needs the same judgment made over roughly 30+ items,
  when a script needs a fast typed decision in a loop, or when asked to use verdict.
  Not for a single judgment you can make by reading the text, for anything that
  needs knowledge or reasoning (which database to pick), or for changing verdict itself.
license: Apache-2.0
---

# verdict: bulk typed decisions

verdict is a System 1 in Kahneman's sense: fast, automatic judgments with a number attached.
You are the System 2. Let it make the same quick call over many items, then spend your own
attention where it flags something. Like any System 1 it is confident whether or not it is
right, so don't gate on a raw score.

It is a CLI on Apple Silicon (`uv tool install --python 3.11 'verdict[mlx,laya] @
git+https://github.com/iksnerd/verdict.git@vX.Y.Z'`). `verdict --help` and `verdict COMMAND
--help` have the full reference; the repo's `docs/guide.md` is the human guide and `examples/`
has real runs.
- verdict runs **base Laya** by default. A plain yes/no is at chance on it (SST-2 0.50), but
  verdict asks every new yes/no as a no/yes choice, where base Laya is within noise of the
  author's private fine-tune (0.94 against 0.96, FINDINGS §40).
- Non-English text needs `--lang multi` or `VERDICT_LANG=multi` (see Reading the numbers).

## Reach for it when, and only when

- **The same question is asked of many items.** 300 transcript turns, every
  command in a log, 80 open issues. You get one number per item and your context
  holds only the numbers.
- **A script needs a decision inside a loop.** `verdict ask ... --cut FILE` exits
  0 or 1.
- **Consistency matters more than brilliance.** The same input scores the same way
  every time, and a fitted cut makes the score a decision.

For one item, read it yourself: you are the stronger model. It reads the surface
of the text, so it cannot say whether Postgres beats SQLite, or which of several numbers is
largest (chance on four; §46: let code compare, and ask it only what one number's text shows). It can say whether a
message is asking a question. In `examples/`, "does this output report a failure?"
scored 12/12, while "must this review comment block a merge?" read an SQL injection as a nit:
the first is on the page, the second needs to know what the code does.

**Pick the right fast model.** Measured head to head on the same held-out items (FINDINGS
§32): on a yes/no question about what the text shows, verdict matched Claude Haiku 4.5 (0.77
against 0.73 AUC) for free, locally and faster. On a seven-way category Haiku won clearly (0.67
against 0.48), and verdict barely beat keyword rules (0.46). So use verdict for bulk surface
yes/no, and anything that must stay on the machine. Use a small LLM for sorting into several
categories or anything needing world knowledge, if the data may leave the machine (ask first
when it's someone's real data). verdict is the right tool only when all five hold: many items, a
short text, a judgment about what it shows, yes/no or a few options, and a need for a number or
for the data to stay local (§41). A good pipeline uses both: verdict over everything, an LLM on
what it flags.

**Where the line sits:** reading is more accurate, and scoring saves your context.
A few dozen short items fit in one read, so read them. From about a hundred up,
score everything with verdict first and read only what it flags. Score even
below that when the same judgment must repeat (a script, a loop) or needs a
consistent number to rank or gate on.

## Before you load it

1. Check the bank before anything loads: `verdict validate -q bank.json --json` needs no model
   and reports a malformed question, failed wording and option truncation (exit 2 if invalid).
2. One model at a time on a laptop GPU: see what is already loaded (`ollama ps`,
   `pgrep -fl 'verdict.*serve'`, which also catches `python -m verdict.cli serve`) and don't
   stack another on top. Short on GPU memory:
   `VERDICT_BITS=8` quantizes at load, about 430 MB instead of 800, with the same answers on
   every bench suite (§36).
3. `verdict serve` in the background, once. Without it every call loads the model
   (about 2 s each). In a script or loop, add `--server-only` to `ask`/`decide`/`calibrate`:
   a missing server then fails with exit 2 instead of loading the model into your process.
4. Score a sample of 200 to 400 items first. `--jsonl` and `calibrate` already
   pause 50 ms between calls; keep it.
5. **Stop the server you started when you are done**: `kill $(lsof -ti TCP:8799 -sTCP:LISTEN)`
   (your `url`'s port). It holds the model in GPU memory. Never kill by name (`pkill -f`): that
   also stops another session's server. If the port was already taken when you began, the
   server is someone else's; use it and leave it running.

## Ask the question that is on the page

**Start from a measured question.** `verdict questions` lists the ones shipped in the tool, with
what each was measured on; use them by name (`-q is_instruction,touches_secret`). Then read
`references/questions.md`: every question asked on a real task so far, including the ones that
failed. Reuse a question that worked; don't retry one that failed for a reason that still holds.
The repo's `examples/README.md` has banks for common agent tasks with labelled rows and their
measured match rates, misses included: did a test run fail (12/12), is a turn an instruction
(10/12), does a snippet hold contact details (9/12), does a passage answer a question. Copy the
closest bank rather than writing one from nothing; `verdict examples NAME` reruns it.

Measured on real agent traffic (FINDINGS §25 to §33):

- Ask about what the text *says*, not about something it can't see.
  "Is this an instruction to perform an action?" predicts tool use at AUC 0.77.
  "How hard is this for a language model?" is at chance (0.48).
- Write the obvious question first. Clever purpose-built wording lost to a plain
  question three times out of three.
- One good question beats a bank: eight combined scored *below* the best one
  alone, at eight times the cost.
- Check who wrote your inputs before scoring them. Filter out bot and linter posts first: in
  one triage run 99 of 137 items were a linter's boilerplate. Then check each question's spread
  (verdict prints it for `--jsonl`). A narrow range (0.58 to 0.65) means it separates nothing.
- **A new yes/no is asked as a no/yes choice for you** and answered as a yes/no (`noul` = P(yes),
  `"asked_as": "choice"`). The plain form is fragile: "Is `text` positive?" ranked reviews at 0.79
  and put no positive over 0.5; as a no/yes choice, 0.96 (§33, §38). Naming both sides yourself (`-o`)
  is at least as good.
- **verdict refuses (exit 2) what measured at chance:** a question about a consequence, difficulty
  or risk ("could this cause harm", 0.53), a yes/no about an absence ("is this ordinary", 0.32 to
  0.64), a choice past 20 options (0.43 at 77), and a state missing a field the question names.
  A state far past the 128-token read gets a warning. Reword to ask what the text says or does;
  don't reach for `--allow-unmeasured` unless you will `calibrate` the question on real labels.
- Give the state named fields and refer to them in backticks:
  `{"command": "..."}` with "Does `command` delete files?".

## The commands you will use

```sh
verdict ask "<text>" "<yes/no question>"                     # 0.72
verdict ask "<text>" "<question>" -o a -o "b=description"    # b 0.61, P(b)
verdict ask "<text>" "<question>" -l low -l mid -l high      # 1.30/2 high
verdict validate -q bank.json --json                         # no model; exit 2 if invalid
verdict decide --jsonl -q bank.json --server-only < items.jsonl > scored.jsonl
verdict decide '{"message": "..."}' -q triage               # presets: verdict presets
verdict decide STATE -q is_instruction,touches_secret         # measured: verdict questions
verdict rank scored.jsonl about_money=1 says_leaving=0.5     # search by named answers, no model
verdict docs findings 38                                     # read a section the help or an error cites
verdict update --check                                       # exit 1 when a newer release exists
verdict config                                               # each setting and where it came from
verdict questions --json                                     # the measured library, parseable
verdict examples --path room-triage                          # an installed example's directory
verdict calibrate labelled.jsonl -q bank.json --out fit.json
verdict decide ... --calibration fit.json                    # adds "decision"
```

**Code written for TypeSafe's Jev runs against verdict.** `verdict serve` also speaks Jev's
`/v1/systemone` protocol, so the official `typesafe-sdk`, LiteLLM's TypeSafe passthrough and
TypeSafe's cookbooks work locally with `TYPESAFE_BASE_URL=http://127.0.0.1:8799` and any
`TYPESAFE_API_KEY`. It is not a full substitute: past about 20 choice options laya degrades
(Jev takes 255), and each extra question costs a full pass here, where Jev's fan-out is one.

**When the tool misbehaves:**
- "not found, so using base laya": `model.path` names a local checkpoint that is not there. Fix
  the path, or remove it to use base Laya on purpose.
- Every answer is 0.5, or every choice is a coin flip: the server is the `uniform` backend (no
  model), usually because it was started as `uvicorn verdict.api:app` instead of `verdict serve`.
  Stop it and run `verdict serve`. Never gate on a uniform answer.
- Each call takes about 2 s: no server is answering, so every call loads the model. Start one.
- "Refused, because answers to this shape measured at chance": the question asks about a
  consequence or an absence, or names a field the state lacks. Reword it or fix the state. Over
  HTTP it is a 422 whose body carries `error.code` `refused`; a 422 without `error` is a
  malformed request. A refusal never succeeds on retry, so do not treat it as an outage.
- A warning says the state isn't English: add `--lang multi` before reading any number.
- A `verdict:` line and exit 2: a usage or settings error, stated in that line. Fix what it
  names; exit 2 never means "no". With `--json`/`--jsonl`, stdout also gets
  `{"error": {"code", "message"}}`: branch on `code` (`refused` means reword the question,
  `no_server` start one, `server_timeout` wait and retry; never load around it).
- "loading the model in this process" on stderr: no server answered, so this call paid for a
  local load. Fine once; in a loop, start `verdict serve` or pass `--server-only`.
- `decide --jsonl` rows carry `index`, the input line they answer: pair by it, not by position.

## Worked example

A user asks which of the 1,500 shell commands their agent ran last month touched a secret.

1. Too many to read well, and each answer is on the page: a yes/no about what the command does.
   That's a job for verdict. Sorting the commands by which service they touch would be a small
   LLM's job instead (FINDINGS §32).
2. `verdict questions` lists `touches_secret` (AUC 0.77 on real blocked commands, Haiku 4.5
   0.73). Use it by name, with states shaped as it says: `{"command": "..."}`.
3. Check what is loaded, then start `verdict serve` in the background, or reuse one already
   answering on the port. Score a sample of 200 first:
   `verdict decide --jsonl -q touches_secret --server-only < commands.jsonl > scored.jsonl`.
4. Sort by the answer and read the top 30 yourself. Then read a slice further down as well: a
   command that pulls secrets without naming one (`vercel env pull`) ranks low, because verdict
   reads the surface (`examples/secret-commands/`).
5. Report the ranking and what you read, not "0.5 means secret". For a repeatable gate, label a
   few dozen and fit a cut with `verdict calibrate`. Stop the server you started.

The repo's `examples/secret-commands/` is this procedure with real output.

## Reading the numbers

- **The ordering carries the signal; the absolute value does not.** An `rm -rf`
  scored 0.37 "destructive". Rank, take the top slice, or fit a cut. Never read
  0.5 as the line.
- Before gating on a score, fit a cut on labelled examples with
  `verdict calibrate`. Real labels are often free: agent transcripts record
  what happened after each turn (tools called, commands rejected). Run a secret scan before using them; one in 2,348 prompts
  in one such corpus held a live credential.
- Report AUC or ranked slices, not accuracy, when the labels are skewed. On 91/9
  data, always saying "no" scores 91% accuracy.
- Non-English state: the English checkpoint stays confident while being wrong.
  verdict warns; pass `--lang multi`, or set `VERDICT_LANG=multi` (`lang = "multi"` in the
  config) for a whole run. On Cyrillic the English checkpoint is near chance (Bulgarian reviews
  0.58, Russian 0.64) and the multilingual one reads them at 0.93 and 0.88, yes/no included
  (§37). Its confidences are uncalibrated too. Keep it for non-English runs: on English it is
  weaker (injection choice 0.65 against 0.83, §35), and under it the library's yes/no questions
  are asked as no/yes choices rather than as measured (§40). `--lang en` overrides a `multi`
  config for one run.

## After a use

If you measured something (a question that worked or failed on a real task, a new failure mode),
it belongs where the next agent will find it: a row in `references/questions.md` (task, labels
or "read", question, result), an issue on the verdict repo for a tool problem, or a
`docs/FINDINGS.md` section for a result worth keeping, with its numbers.
