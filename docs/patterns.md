# Patterns

TypeSafe's docs describe four ways to build with typed answers: fan-out, confidence routing,
composite scoring and intent routing. verdict supports each, with Laya's limits stated where they
change the advice. All of them rest on one idea from Jev's design: ask atomic questions, and
combine the answers in code, not in a prompt.

## Atomic questions, composed in code

One narrow question beats one question that weighs several things. Score market size, feasibility
and differentiation separately, then weight them yourself. On Laya this is stronger than elsewhere:
eight questions combined in one bank scored below the best single one (FINDINGS §27), and a model
that cannot compare numbers (§46) can still read each one. See `verdict docs rough-edges`.

## Fan-out: many questions, one state

```sh
verdict decide '{"message": "Duplicate charge on invoice 4411, refund please"}' -q bank.json
verdict decide --jsonl -q bank.json < items.jsonl > scored.jsonl     # many states
```

A bank answers every question about the state in one command. Jev evaluates them in parallel in one
request; Laya runs a forward pass per question, about 30 ms each, and batching states does not help
(§34). Keep banks to the questions you will use.

## Confidence routing: act only when sure

Gate on a cut you fitted, not on 0.5. Laya's probabilities rank but are not calibrated (§2).

```sh
verdict calibrate labelled.jsonl -q bank.json --out fit.json      # a cut, on a held-out split
verdict decide --jsonl -q bank.json --calibration fit.json < items.jsonl   # adds "decision"
verdict ask "$CMD" "Does \`command\` touch a secret?" --cut fit.json && echo flagged
```

Send what falls near the cut, or what `classify` scores low, to a person or a larger model.

## Composite scoring: weights set in code

Score dimensions once, then change the weights without asking again.

```sh
verdict decide --jsonl -q bank.json < tickets.jsonl > scored.jsonl
verdict rank scored.jsonl urgency=2 churn=1 spam=-3
```

Each dimension is rescaled to its percentile before the weighted sum, because raw dimensions have
different ranges, and never ranked by cosine, which matched an item that scored low on everything
(§39).

## Intent routing: code, a specialist, or a person

Ask a choice over your own handlers, with a "none of these" option, and branch on the answer. Keep
it to 20 options; past that use `verdict classify` with a tree, or Jev.

```sh
verdict ask "$MSG" "Which handler fits?" -o "billing=refunds, invoices" -o "tech=bugs, outages" -o other
verdict classify "$MSG" --tree handlers.json --beam 2     # more than 20 handlers
```

verdict's own big-or-small model router (`verdict route`) is a worked example, not a router to
trust: it is at chance on real traffic (§25). It is refused when Jev or another model sits behind
`verdict serve`, since its thresholds were fitted to Laya.

## Hierarchical classification: more than 20 labels

```sh
verdict classify "my transfer never arrived" --tree examples/banking-tree/tree.json
```

Groups first, then options inside the best few. 77 Banking77 intents went from 0.38 flat to 0.55
with `--beam 2` (§50). The joint probability is low when the tree lost the item, which makes a
good review queue. The tree has to be written by someone who knows the labels: one clustered
mechanically scored 0.34 (§53). **For topical labels, an embedding model beats this**: the nearest
label by `embeddinggemma:300m` scored 0.89 on the same items, and Laya added nothing on top. Reach
for `classify` when you want no second model, or when the labels are not self-describing.

## Cascade: Laya first, Jev for the items Laya is least sure of

Ask Laya about everything, locally and free, and send only the least-sure share to Jev. Measured on
the 12 bench suites (§52): 10% sent closes 39% of the gap to Jev, 20% closes 56%, half closes 92%,
and the same shares sent at random close 12%, 22% and 52%. The text of the other items never leaves
the machine. It is a per-task call: on AG News Laya beats Jev and every item sent makes it worse.


```sh
verdict serve --systemone https://api.typesafe.ai --port 8801 &       # the second backend (Jev)
verdict cascade -q bank.json --escalate http://127.0.0.1:8801 --check labelled.jsonl   # does it help here?
verdict cascade -q bank.json --escalate http://127.0.0.1:8801 --share 0.3 < rows.jsonl
```

`--check` asks the second server about your labelled examples and reports, per question, the primary,
the cascade and the second on their own, and names the questions where escalating would not help.
Each output row marks every answer `primary` or `escalated`, and a line on stderr says how many
states were sent out. Live on 200 public prompt-injection items: Laya 0.805, Jev 0.845, the cascade
with 60 of 200 sent 0.845; on AG News the check said escalating would not help (§54).

## Select instead of generate, and verify evidence

Find candidates in code, then ask whether each one is what you want; check a claim against its
source. The `examples/` folder has working banks: `citation-check`, `passage-filter`,
`task-verification`, `commit-kinds`, `ticket-search`. `verdict examples` scores them against a
backend, Laya or `--systemone`.

## When Jev is the better backend

Bulgarian and Russian, yes/no sentiment and injection, and flat banks past 20 options (Jev takes
255). Laya is better on topic classification, cost and latency, and keeps every text on the
machine. `verdict serve --systemone https://api.typesafe.ai` swaps the backend with the same
commands; see `verdict docs models`.
