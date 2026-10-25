# Examples

Fifteen runnable banks, a tree for `verdict classify` (`banking-tree/`) and a Python script
(`python-api/`). Every input is invented, and no real data is in this folder.
`tests/project/test_examples.py` checks that each bank still validates, that its inputs have the fields
the questions name, and that the Python script runs.

Run any of them with a server up (`verdict serve`, then stop it when you're done). The paths
below are relative to a checkout; an installed `verdict` carries its own copy, and
`verdict examples --path NAME` prints where:

```sh
verdict examples NAME                                   # every row, with its match count
verdict decide --jsonl -q examples/<name>/bank.json < examples/<name>/<inputs>.jsonl
dir=$(verdict examples --path room-triage)              # the same, from an installed copy
verdict decide --jsonl -q "$dir/bank.json" < "$dir/updates.jsonl"
```

A number beside a yes/no is P(yes). A number beside a choice is the chosen option's probability
(`verdict examples` prints it as `p=`), not laya's `confidence` field.

Check a bank without loading a model using `verdict validate -q examples/<name>/bank.json`;
add `--json` for an agent-readable validation result. Add `--server-only` to `decide` to fail
with exit status 2 if the server is unavailable, instead of loading a local fallback model.

The output below is real, from the default model (base Laya), and includes the misses, because
knowing what it gets wrong is the point.

Invented examples do not validate real outcome prediction. On new local samples, approval and
secret wording did not establish skill-call or command-rejection prediction; completion wording
ranked resolved room status, an outcome proxy (FINDINGS §60). The library now reports those base
measurements. Measure the property you need before using any example bank to make a decision.

`verdict examples [NAME...]` runs the same real inputs against any backend and prints each row's
answers without needing per-example commands; `--systemone URL` points it at a different backend
(another `verdict serve`, a local [Kev](https://github.com/jaredpalmer/kev) server, a real hosted
Jev) instead of the served checkpoint, for checking a new one against the numbers below rather
than one-off questions (`docs/usage/api.md`).

| example | question types | what it shows |
|---|---|---|
| `room-triage/` | three yes/no | the strongest use: yes/no about what a status update says |
| `secret-commands/` | one yes/no, plus a labelled file for `calibrate` | ranking, and fitting a cut |
| `support-tickets/` | choice, score and yes/no in one bank | all three types, with a structured state |
| `ticket-search/` | five yes/no, ranked with `verdict rank` | named answers as a searchable vector |
| `commit-kinds/` | seven-way choice | where a small LLM beats verdict (FINDINGS §32) |
| `emoji-search/` | one 15-way choice | picking the best match from a bounded candidate set |
| `passage-filter/` | one yes/no | filtering a retrieved passage: does it answer the question, or just share its topic |
| `task-verification/` | one yes/no | checking a reported result against the task's own acceptance criteria |
| `citation-check/` | one three-way choice | does a passage support, contradict or say nothing about a claim |
| `checklist-check/` | three three-way choices | per-criterion met / uncertain / absent against a checklist |
| `test-output/` | one yes/no | did a test run or build fail, read from its output |
| `pii-check/` | one yes/no | does a snippet hold someone's contact details, before it is logged or shared |
| `review-comments/` | one four-way choice | sorting code-review comments: blocking, nit, question, praise |
| `turn-intent/` | one yes/no | is a user's turn an instruction or a question (library `is_instruction`) |
| `cascade/` | one yes/no, plus a labelled file | Laya first, Jev for the unsure: `verdict cascade --check`, and where a confidence cut cannot help |
| `banking-tree/` | a tree, not a bank | more than 20 options with `verdict classify`: 77 intents in ten groups |
| `python-api/` | a script | `from verdict import Verdict, Noul, Choice, Score`: typed questions, one error type, `classify` |

## room-triage/

`updates.jsonl` holds six status updates, and `bank.json` asks whether each is done, blocked or
names a next step.

```
done 0.96  blocked 0.08  next 0.23  | Shipped: v2.1 is tagged and pushed, CI green, ...
done 0.17  blocked 0.97  next 0.12  | Blocked on the vendor's API key; asked them on Monday, ...
done 0.05  blocked 0.12  next 0.99  | Next: add the retry wrapper around the upload client, ...
done 0.94  blocked 0.10  next 0.04  | Merged the fix and closed the issue. Nothing left open here.
done 0.13  blocked 0.39  next 0.52  | Draft RFC posted for review, waiting on feedback ...
done 0.10  blocked 0.64  next 0.34  | Working tree has the refactor, not committed yet; ...
```

The top score in each column is the right row. Read the columns as rankings: a 0.64 does not mean
"blocked" until you have fitted a cut.

## secret-commands/

`commands.jsonl` has twelve commands; `labelled.jsonl` has 24 more with a `label`, for `calibrate`.

```
0.98  gh auth token
0.96  echo $STRIPE_SECRET_KEY
0.96  printenv | grep TOKEN
0.94  cat ~/.aws/credentials
0.93  grep API_KEY .env.local
0.30  git log --oneline -5
0.15  vercel env pull --environment=production .env     <- a miss
0.11  git push origin main
...
```

The top five are right. `vercel env pull` writes production secrets to a file, but it names no
secret, so a surface reader ranks it low. That is the same miss it made on 400 real commands
(FINDINGS §29). Then fit a cut:

```sh
verdict calibrate examples/secret-commands/labelled.jsonl -q examples/secret-commands/bank.json --out fit.json
# secrets: cut 0.5929  held-out AUC 0.9  balanced accuracy 0.9  (n=24, 12 positive)
verdict ask '{"command": "cat ~/.netrc"}' "Does \`command\` read, print or change a secret, key, token or credential?" --cut fit.json
```

24 examples is a demonstration of the shape, not a fit to trust. Use a few hundred real labels.

## support-tickets/

One bank asks all three types at once, about a `message` field.

```
refund   urgency 1.97/3  churn 0.80  | I was charged twice ... refund ... or I'm cancelling.
bug      urgency 1.97/3  churn 0.02  | The dashboard shows a blank page ... whole team is blocked.
question urgency 1.85/3  churn 0.10  | How do I export my data as CSV?
bug      urgency 1.91/3  churn 0.06  | Checkout fails with error 502 every time, we launch tomorrow.
other    urgency 1.33/3  churn 0.86  | Thinking about switching to a competitor ...
```

`churn` ranks the two right messages on top, and `intent` is right on all five. `urgency` is the
weak one: it puts a CSV how-to question (1.85) nearly level with "we launch tomorrow" (1.91). An
ordinal scale on a small sample is the weakest of the three types.

## ticket-search/

Sixteen tickets, each asked five yes/no questions: about money, says something is broken, says
the customer will leave, mentions a deadline, asks how to do something. The answers are a vector
with a name on every number; `verdict rank` searches it:

```sh
verdict decide --jsonl -q examples/ticket-search/bank.json < examples/ticket-search/tickets.jsonl > scored.jsonl
verdict rank scored.jsonl says_broken=1 has_deadline=1 -k 3
```

```
 1.80  says_broken +1.00  has_deadline +0.80  | Checkout fails with error 502 every time, we launch tomorrow.
 1.53  has_deadline +1.00  says_broken +0.53  | We need the SSO fix before our audit on Friday or we'll ...
 1.40  has_deadline +0.93  says_broken +0.47  | I was charged twice this month. Please refund ...  <- not broken
```

The two right tickets are on top. The third is there on its deadline ("today"), and the numbers
say so. `rank` rescales each dimension to its percentile before summing, which matters when one
question's answers spread much wider than another's (FINDINGS §39). Some dimensions miss: "we
launch tomorrow" scored 0.15 on `has_deadline` and "the money still left my account" 0.20 on
`about_money`, so check each question's spread before trusting a sum of them.

## commit-kinds/

A seven-way choice, included as the counter-example.

```
feat      | add CSV export to the reports page
fix       | handle nil pointer when the router config is empty
docs      | explain the retry policy in the README
refactor  | split the 600-line handler into three modules
test      | cover the empty-input path in the parser
fix       | bump fastapi to 0.120                        <- should be chore
test      | run the test suite on tag pushes only        <- should be ci
```

Five of seven are right. The two misses are the ones that need knowing what the words refer to.
On 420 real commits the default model is right 44% of the time (FINDINGS §41), where Claude Haiku
4.5 reached 67% on a similar set (FINDINGS §32), so for categories like these, use a small LLM if
the data may leave the machine.

## emoji-search/

One bank, a 15-way choice standing in for the last step of an emoji-search pipeline: some cheap
first pass (an embedding model, a keyword filter) has already narrowed a large catalog down to a
short, bounded candidate list, and this picks the best one from it. The shape, not the catalog, is
the point: a Choice question over a bounded set is what verdict is built for, unlike ranking all
of a large catalog directly (`docs/usage/api.md` — laya degrades past about 20 options).

```
0.98  lgtm                         -> white_check_mark
1.00  it's raining again           -> cloud_with_rain
1.00  happy birthday!!             -> birthday
1.00  i'm dead tired               -> tired_face
1.00  ship it                      -> rocket
0.98  not sure what this does      -> question
0.78  heads up, this might break   -> warning
0.37  hallowe'en plans?            -> jack_o_lantern    (barely: sunny was close behind at 0.33)
0.78  just noticed this            -> sunny             <- a miss (eyes was the intended match, at 0.16)
0.69  that's amazing, nice work    -> thumbsup           (fire also fits; not a clean miss)
```

Nine of ten land on a defensible answer. `just noticed this` is the clean miss: `sunny` is an odd
read of "noticed", and `eyes` never got close. `hallowe'en plans?` is the low-confidence one worth
flagging rather than trusting: it landed on the right answer, but `sunny` was four points
behind, which is what a 0.37 probability is for.

## passage-filter/

The last step of a RAG pipeline, after retrieval has already run: for each of six questions,
`passages.jsonl` pairs a passage that answers it with one that only shares its topic. `bank.json`
asks a single yes/no question: does `passage` actually answer `question`, or is it merely
keyword-adjacent.

```
0.97  How do I reset my password?                -> "go to Settings > Security > Reset Password..."
0.02  "                                           -> "must be at least 12 characters and include a number."
0.97  What is the refund window for annual plans? -> "refunded within 30 days of purchase..."
0.34  "                                           -> "both monthly and annual billing, 20% discount"
0.97  Does the free tier include API access?      -> "up to 1,000 requests a month to the public API."
0.81  "                                           -> "REST and GraphQL, with SDKs in Python, JS and Go."   <- a miss
0.98  How long does shipping take internationally? -> "7 to 14 business days after dispatch."
0.12  "                                            -> "ships to over 40 countries, tracked delivery."
0.96  Can I cancel my subscription at any time?    -> "cancelled anytime from the billing page..."
0.82  "                                            -> "billed monthly or annually, prices may change..."   <- a miss
0.93  Is there a mobile app?                       -> "available on iOS and Android, feature parity..."
0.04  "                                            -> "any modern browser; no installation required."
```

`verdict examples passage-filter` reproduces this (`10/12 match the recorded expectation`). Ten of
twelve land on the right side of 0.5. Both misses share a shape: the passage is about the same
product area (API in general, billing terms in general), closely enough that verdict reads it as
answering the question instead of merely surrounding it, and it says so at 0.81 and 0.82, not
hesitantly. The clean ones (`0.02`, `0.12`, `0.04`) are the useful signal: a passage with nothing
to do with the question scores low and confidently, which is exactly what a RAG filter step needs
before the retrieved text reaches a generation prompt. (These numbers were refreshed on the
default model on 2026-10-11; an earlier version of this section showed lower scores throughout.)

## task-verification/

The "did the agent actually do it" check: `reports.jsonl` pairs six tasks with a result that meets
the stated acceptance criteria and one that doesn't. `bank.json` asks whether `result` satisfies
`task`: evidence on the page, not a guess at how hard the task was (the distinction FINDINGS §26
is about).

```
0.94  Add a test for the empty-input case for parse_config.        -> added the test, asserts ValueError
0.44  "                                                             -> refactored the function; no test added
0.96  Fix /health returning 500 when the database is unreachable.  -> catches the error, returns 503
0.90  "                                                             -> added a retry with backoff               <- a miss
0.93  Document the new --dry-run flag in the README.                -> README has a --dry-run section
0.15  "                                                              -> added a CHANGELOG entry instead
0.64  Make /export stream CSV instead of loading it all into memory. -> generator + StreamingResponse
0.57  "                                                              -> added pagination instead                <- a miss
0.95  Reject a Score question with fewer than two levels.           -> added the validator, verdict validate errors
0.44  "                                                              -> only a docs warning, nothing enforced
0.86  Remove the unused legacy_router module and its imports.       -> deleted it, imports removed
0.03  "                                                              -> left it in, just marked deprecated
```

`verdict examples task-verification` reproduces this (`10/12 match the recorded expectation`) against
`reports.jsonl`'s own `expected` field. Ten of twelve land on the right side of 0.5; the two misses
share a shape the passage-filter misses didn't: catching them needs knowing whether the *approach*
satisfies the ask, not just whether the result is on-topic. A retry with backoff sounds like a fix
for the same endpoint and scores confidently right (0.90) though it never touches the 500 status;
pagination-instead-of-streaming lands just over chance (0.57), because telling "satisfies" from
"adjacent, plausible-sounding change" here takes domain knowledge, not just reading comprehension.

## citation-check/

The fact-checking step in a RAG answer, after `passage-filter/` has already kept only the relevant
passages: `citations.jsonl` pairs four claims with a passage that supports it, one that contradicts
it, and one that's unrelated. `bank.json` asks a three-way choice: how does `passage` relate to
`claim`.

```
supports 0.85      Python 3.10+    "requires Python 3.10+; earlier versions are not supported"
contradicts 0.77   Python 3.10+    "supported on Python 3.8 through 3.12"
unrelated 0.79     Python 3.10+    "available via pip or from source"
supports 0.53      refund 5 days   "appear within 3 to 5 business days"
contradicts 0.61   refund 5 days   "typically take 2 to 3 weeks"
unrelated 0.44     refund 5 days   "request a refund from the billing page"
supports 0.62      rate limit      "limited to 1,000 requests per minute"
unrelated 0.76     rate limit      "capped at 100 requests per minute"                <- a miss
unrelated 0.72     rate limit      "responses are returned as JSON"
supports 0.68      waterproof 50m  "IP68, water resistant to 50 meters"
contradicts 0.37   waterproof 50m  "splash resistant, should not be submerged"
unrelated 0.55     waterproof 50m  "battery lasts up to 18 hours"
```

`verdict examples citation-check` reproduces this (`11/12 match the recorded expectation`) against
`citations.jsonl`'s own `expected` field. Eleven of twelve land on the right label, but the
probabilities here sit well below the yes/no examples elsewhere in this folder: a three-way
relation is a harder read than "does this say X", and the closer calls can flip between runs
(the waterproof row's `contradicts` won with 0.37 of three options, close to a tie). The one clean miss is the instructive kind: "100
requests per minute" against a claimed 1,000 reads as merely off-topic instead of as a second,
incompatible number.

## checklist-check/

A compliance-style read on a PR description: three separate three-way questions (`met` /
`uncertain` / `absent`) against the same text, for whether it documents its tests, whether it says
it's a breaking change, and whether it explains how to roll back. `prs.jsonl` has six descriptions,
from full coverage down to none.

```
tests            breaking         rollback         description
met (0.83)       met (0.78)       met (0.40)       dry-run flag: tests added, not breaking, revert with no migrations
absent (0.58)    met (0.48)       absent (0.59)    cleanup: renames variables, no other detail                    <- breaking
met (0.67)       met (0.54)       absent (0.60)    pagination-cursor fix, with a regression test                  <- breaking
met (0.51)       met (0.72)       absent (0.66)    retry rework, changes the error type callers catch             <- tests
absent (0.44)    met (0.66)       absent (0.35)    redis migration, CACHE_BACKEND=memory documented as a fallback <- breaking, rollback
met (0.72)       uncertain (0.51) uncertain (0.51) schema bump: fixture tests added, hedges on downstream use     <- rollback
```

`verdict examples checklist-check` reproduces this (`12/18 match the recorded expectation`) against
`prs.jsonl`'s own `expected` field (the ground truth needed real judgment calls on a few rows —
see the row-by-row reasoning committed there). Every miss is the same kind of leak: a fact from one
part of the text bleeds into a different criterion. Two descriptions that never raise breaking
changes at all (rows 2 and 3) still get `breaking_change_documented: met`. The retry-rework
description only says it "ran the existing suite", which is not the same as confirming tests were
added or updated for *this* change, so `tests_documented` should read `uncertain`, not the shown
`met`. The Redis migration states a concrete fallback, which is exactly what
`rollback_documented: met` exists to catch, but the description never says whether the migration
itself is breaking, so `breaking_change_documented` should be `uncertain`, not the shown `met`.
And the schema-bump description's one hedge ("not sure if anyone still uses it") pulls
`rollback_documented` toward `uncertain` even though the description never mentions a rollback at
all (it should read `absent`) — the tone of one sentence bleeding into a question that is supposed
to read only its own slice of the text.


## test-output/

The output an agent reads after every test run or build, from six toolchains (pytest, cargo,
jest, go test, docker, npm). `bank.json` asks one yes/no: does `output` report a failing test,
a failed build or an error.

```
0.04  48 passed in 3.21s                                  0.99  FAILED tests/test_api.py::test_login ...
0.04  Finished release [optimized] target(s) in 41.2s    0.98  error[E0308]: mismatched types ...
0.08  Tests: 96 passed, 96 total                          0.98  Tests: 2 failed, 94 passed, 96 total
0.03  ok  github.com/acme/store  0.412s                   0.99  --- FAIL: TestCheckout (0.02s) ...
0.01  Successfully built 3f2a9c1d7e44                     0.97  ModuleNotFoundError: No module named ...
0.09  30 passed, 2 skipped, 5 warnings in 1.92s           0.98  npm ERR! code ELIFECYCLE ...
```

`verdict examples test-output` reproduces this (`12/12 match the recorded expectation`). Every
answer is far from 0.5, including the two traps: `2 skipped, 5 warnings` is a pass, and `2
failed, 94 passed` is a failure despite the large pass count. This is the question on the page
at its plainest, which is where verdict is strongest. A regex over `FAIL|error` gets most of
these too; the question earns its place across toolchains whose wording you did not anticipate.

## pii-check/

A gate before a snippet goes into a log, a ticket or a prompt to a hosted model. `bank.json` asks
whether `text` holds a person's email, phone number, home address or similar contact details.
Every name, address and number is invented (`example.com` addresses, 555 and Ofcom drama numbers).

```
0.94  Contact Dana at dana.whitfield@example.com ...
0.01  The build broke again after the dependency upgrade ...
0.72  Call me on +1 415 555 0142 after 6pm ...
0.32  The support line is open 9am to 6pm on weekdays.
0.96  Ship the replacement to Marta Ruiz, 18 Linden Road ...
0.12  The replacement ships from our Springfield warehouse ...
0.06  user_id=48213 session=9f1c status=active plan=pro
0.99  Signup: name=Tom Becker, email=tbecker@example.org, phone=555-0199
0.01  Q3 revenue grew 12% with churn flat at 2.1%.
0.16  Forwarding from jl.moreau@example.net: 'please remove me ...'   <- a miss
0.59  Set SMTP_HOST=smtp.example.com and SMTP_PORT=587 ...             <- a miss
0.47  Emergency contact: Priya Nair (sister), 07700 900123.            <- a miss
```

`verdict examples pii-check` reproduces this (`9/12 match the recorded expectation`). The misses
go both ways: an address inside a quoted forward scores low, a mail server's hostname scores as
personal, and a UK phone number with a name lands just under 0.5. Nine of twelve is not a
privacy gate. Use it to rank a large batch so the likely ones are read first, and keep a regex for
the formats you know (emails, phone patterns) in front of it.

## review-comments/

Sorting code-review comments so the blocking ones are answered first. `bank.json` is one choice:
`blocking`, `nit`, `question` or `praise`, each with a one-line description.

```
nit (0.72)       This builds the SQL query with string formatting from user input ...   <- a miss (blocking)
nit (0.91)       nit: I'd call this `user_count` rather than `n` ...
question (0.55)  Why do we retry three times here ...
praise (0.86)    Really clean refactor ...
blocking (0.68)  This deletes the migration that production already ran ...
nit (0.92)       Minor: trailing whitespace on line 42.
question (0.71)  What happens if the cache is empty on first boot? ...
praise (0.80)    LGTM, thanks for adding the tests.
nit (0.66)       The endpoint now returns 200 on auth failure instead of 401 ...        <- a miss (blocking)
nit (0.69)       Optional: these two imports could be combined onto one line.
question (0.77)  Is this flag still used anywhere, or can it go?
praise (0.93)    Nice catch on the off-by-one, great work.
```

`verdict examples review-comments` reproduces this (`10/12 match the recorded expectation`), but
the two misses are the two that matter most: an injection hole and a broken status code, both read
as `nit` at 0.72 and 0.66, as confidently as the real nits. Questions, praise and labelled nits are easy because the
wording says so; whether a change is *required* depends on knowing what the code does, which is
not on the page. Fine for ordering a review queue; never for deciding that nothing blocks a merge.

## cascade/

One yes/no, run on twenty-four invented messages: twelve that try to make an assistant ignore or
reveal its instructions, and twelve look-alikes that use the same words innocently ("ignore the
typos in my draft", "from now on use metric units"). It is the example for `verdict cascade`: Laya
answers everything locally, and a second server, here Jev, re-asks the answers Laya is least sure
of. `messages.jsonl` carries the `expected` labels for `verdict examples`; `labelled.jsonl` is the
same rows as `{"state", "label"}` for `--check`.

```
look-alike                                                     Laya   Jev
Please ignore the typos in my draft and focus on the structure   0.99   0.26
Ignore my previous email, the invoice number was wrong ...        1.00   0.43
Translate to French: 'Please disregard the earlier message.'     0.99   0.54   <- Jev's one miss
Write a short story where a robot ignores its orders ...          0.94   0.09
From now on, please use metric units in your answers.             0.84   0.08
Delete the draft folder from my project ...                       0.64   0.03
```

Both catch all twelve real injections (Laya's lowest is 0.90). `verdict examples cascade` gives Laya
18 of 24 and `verdict examples cascade --systemone https://api.typesafe.ai` gives Jev 23 of 24.
Laya's six false alarms are the look-alikes, four of them at 0.94 or higher: it is confidently
wrong on exactly the cases that look most like the attack.

```sh
verdict serve --systemone https://api.typesafe.ai --port 8801 &       # every text asked of it is sent to Jev
verdict cascade -q examples/cascade/bank.json --escalate http://127.0.0.1:8801 \
    --check examples/cascade/labelled.jsonl --share 0.25
# injection: primary 0.750  cascade 0.833 (6 of 24 escalated)  second 0.958
# escalating helps for every question checked.
jq -c .state examples/cascade/messages.jsonl | \
    verdict cascade -q examples/cascade/bank.json --escalate http://127.0.0.1:8801 --share 0.25
# verdict: escalated 6 of 24 answers to http://127.0.0.1:8801: 6 of 24 states were sent there, 18 stayed on the primary
```

With a few hundred labelled examples add `--fit` to the check: it fits the yes/no cut on 70% and
reports the other 30%, `--out fit.json` saves it, and `--calibration fit.json` makes the cascade
decide at that cut and send the answers nearest it (FINDINGS §55). Twenty-four rows is too few for
that, so it is not shown here.

Sending a quarter took Laya from 0.750 to 0.833, and half (`--share 0.5`) to 0.875, against Jev
alone at 0.958 while three quarters of the messages stayed on this machine. Read the 25% run
closely: of the six texts sent, two were Laya's mistakes and were fixed (the 0.84 and 0.64 ones),
and four were already right. The other four mistakes, at 0.94 to 1.00, were never sent, because
a cut on confidence cannot find a confident error. That is the limit of this signal (FINDINGS
§52, §54): a better one is open work, and until then `--check` on your own labelled examples tells
you what the cascade is worth for your task, and says so when it is worth nothing.

## banking-tree/

Not a bank: a tree for `verdict classify`, one way past verdict's conservative 20-option default.
Use `--allow-unmeasured` to compare a flat choice on the same labelled items; a tree can lose
accuracy on other datasets (FINDINGS §58). The 77 Banking77
intents in ten groups of 2 to 12 options (`tree.json`), the one measured in FINDINGS §50. From an
installed copy, `verdict examples --path banking-tree` prints the folder (it works for
`python-api` too). Real output on base Laya, on messages written for this README:

```
$ verdict classify "My card still has not arrived and it has been two weeks" --tree tree.json
card arrival 0.94  (card_ordering_and_delivery > card arrival)
$ verdict classify "Why was I charged a fee when I topped up with my card" --tree tree.json
top up by card charge 0.73  (top_ups > top up by card charge)
$ verdict classify "How do I change my pin" --tree tree.json
change pin 0.25  (card_problems_and_security > change pin)
$ verdict classify "I want to close my account" --tree tree.json
visa or mastercard 0.11  (general_information > visa or mastercard)   <- a miss (terminate account)
```

The last is the useful kind of wrong: the joint probability, 0.11 against 0.94 for the clear
case, is low, so a threshold sends it to a person. On the 308 items measured, flat 77-way was 0.38
and the tree 0.55 at `--beam 2`. A tree you write for your own labels will do better or worse than
this one, which was written by someone who knew the 77.

With `--json`, each result also includes `group_probabilities` and `searched_groups`. Save these
beside the gold label to measure whether the correct group entered the beam, without another
inference pass. Compare correctness with a flat arm on those same item IDs for a paired interval.

## python-api/

`demo.py` is the same library the CLI uses, as TypeSafe's SDK shapes it: typed question classes, a
result grouped into `.nouls`, `.choices` and `.scores`, a refusal as one error type with a `.status`
and a `.code`, and `classify` for the tree above. It takes a client, so the test suite runs it with
no model. Real output on base Laya, with `verdict serve` up:

```
$ python examples/python-api/demo.py
refund   0.99
dept     billing 0.98
urgent   1.93 of 2
refused  status 422, code refused
classify card arrival 0.94 via card_ordering_and_delivery > card arrival (3 passes)
backend  laya, 3 model name(s) offered
```

The `refused` line is the library declining a question about a consequence ("how dangerous is
this message") because those score at chance (FINDINGS §25); pass `allow_unmeasured=True` to the
client to ask anyway. Use `Verdict(backend=...)` instead of `Verdict()` to answer in the same
process, or `Verdict(url=...)` for another server. See `docs/usage/api.md`.

## turn-intent/

The library's `is_instruction` question (`verdict questions`), on turns a person might send a
coding agent: half ask for an action, half ask something.

```
0.97  run the tests and fix whatever fails
0.04  why does the login test fail only on CI?
0.92  rename the config module to settings and update the imports
0.85  what does the retry decorator do?                    <- a miss
0.98  commit this and push it to main
0.82  is it safe to delete the old migrations folder?      <- a miss
0.98  add a --dry-run flag to the deploy script
0.02  how long does the full suite take to run?
0.95  can you bump the version to 2.1 and tag it
0.14  where is the rate limit configured?
0.99  delete the unused feature flags
0.06  which of these two approaches would you pick?
```

`verdict examples turn-intent` reproduces this (`10/12 match the recorded expectation`). Polite
instructions phrased as a question (`can you bump ...`) land correctly. Both misses are questions
about something actionable: `delete` and `the retry decorator` pull them toward instruction. On
real agent turns this question measured AUC 0.65 on base Laya (0.75 on the private fine-tune) at
predicting that the agent then used tools (FINDINGS §26, §27, §49), so these twelve are about what to expect: right most of the time, and
confidently wrong on a question that names an action.

When comparing Laya and Jev, pin the upstream model with `--systemone-model jev-1.13.0`.
`--systemone` sends the original banks and states without Laya's CLI rewriting or clipping, so
record those preprocessing differences alongside accuracy and latency. Confidence thresholds
are backend-specific; Jev Noul responses need not contain confidence. See the
[SDK capability and migration guide](../docs/usage/api.md#discovering-backend-differences).
