# Examples

Five runnable examples. Every input is invented, and no real data is in this folder.
`tests/test_examples.py` checks that each bank still validates and that its inputs have the fields
the questions name.

Run any of them with a server up (`verdict serve`, then stop it when you're done):

```sh
verdict decide --jsonl -q examples/<name>/bank.json < examples/<name>/<inputs>.jsonl
```

Check a bank without loading a model using `verdict validate -q examples/<name>/bank.json`;
add `--json` for an agent-readable validation result. Add `--server-only` to `decide` to fail
with exit status 2 if the server is unavailable, instead of loading a local fallback model.

The output below is real, from the default model (base Laya), and includes the misses, because
knowing what it gets wrong is the point.

| example | question types | what it shows |
|---|---|---|
| `room-triage/` | three yes/no | the strongest use: yes/no about what a status update says |
| `secret-commands/` | one yes/no, plus a labelled file for `calibrate` | ranking, and fitting a cut |
| `support-tickets/` | choice, score and yes/no in one bank | all three types, with a structured state |
| `ticket-search/` | five yes/no, ranked with `verdict rank` | named answers as a searchable vector |
| `commit-kinds/` | seven-way choice | where a small LLM beats verdict (FINDINGS §32) |
| `emoji-search/` | one 15-way choice | picking the best match from a bounded candidate set |

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
of a large catalog directly (`docs/api.md` — laya degrades past about 20 options).

```
0.96  lgtm                         -> white_check_mark
1.00  it's raining again           -> cloud_with_rain
1.00  happy birthday!!             -> birthday
1.00  i'm dead tired               -> tired_face
0.99  ship it                      -> rocket
0.95  not sure what this does      -> question
0.68  heads up, this might break   -> warning
0.38  hallowe'en plans?            -> jack_o_lantern    (barely: sunny was close behind at 0.33)
0.74  just noticed this            -> sunny             <- a miss (eyes was the intended match, at 0.16)
0.72  that's amazing, nice work    -> thumbsup           (fire also fits; not a clean miss)
```

Nine of ten land on a defensible answer. `just noticed this` is the clean miss: `sunny` is an odd
read of "noticed", and `eyes` never got close. `hallowe'en plans?` is the low-confidence one worth
flagging rather than trusting: it landed on the right answer, but `sunny` was one point behind,
which is what a 0.38 confidence is for.
