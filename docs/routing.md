# The switch, and the router built on it

← [Back to the README](../README.md)

> **The router is at chance on real agent traffic** (AUC 0.478 on 2,312 real turns, FINDINGS
> §25). It is kept as the worked example of a switch, not as something to put in front of real
> traffic. For real bulk questions, see the [guide](guide.md); for intent routing that is measured
> (a choice over your own handlers, a tree past 20 options, a cascade from Laya to Jev for the
> unsure), see `verdict docs patterns`.

## The idea: declare a decision as a switch

Write down the branches you are willing to take and which one to fall back to, and a branch
comes back. The mechanism that picks it is separate from the declaration (`src/verdict/judge/switch.py`),
so it can be measured and swapped without touching a call site. The separation matters: the obvious
mechanism, one exclusive choice question over the branch names, was the *worst* one available on
this project's data (§3, §9, §14). A switch always has a default branch, taken whenever the model is
unsure or absent. A `uniform` backend therefore always lands on the default, instead of pretending
to decide.

## `verdict route`

Two cases, big and small, with two questions behind them (upstream's `laya.router_questions()`):

```
$ verdict route "what is 17% of 340"
branch     SMALL  (70 ms via server http://127.0.0.1:8799)
reason     easy (difficulty 1.46/3) and not sensitive
scores     difficulty=1.46  sensitive=0.25
```

- `verdict route "$PROMPT" --quiet` prints only `small` or `big`. The exit status is 0 for small
  and 1 for big, and the calling script decides what to do with it. verdict runs nothing.
- `verdict route --batch < prompts.txt` streams NDJSON over one connection. 72 prompts took 2.5 s,
  against 5.4 s calling once each and about 151 s with no server; 34 ms a prompt is the inference
  floor (§18).
- `verdict cases` prints the switch: its branches, the default, and the questions behind it.

## How well it routes

On 72 hand-written prompts in `bench/router_prompts.jsonl`, fitted on 48 and reported on the 24 the
fit never saw:

| | train | held-out |
|---|---|---|
| accuracy | 83.3% | 70.8%, 95% CI 54.2 to 87.5 (chance 50%) |
| latency | | 34 ms p50, 2 questions |

**That number does not transfer.** The 72 prompts read like textbook examples. Against 2,312 real
turns labelled by what the agent actually did, the router is at chance. 92% of real turns call
tools, and more than half are few-word fragments that only mean something inside the session (§25). The same checkpoint scored
0.753 on those turns when asked a question about what is on the page, "Is this an instruction?"
(§26; that checkpoint was the private fine-tune, and base Laya scores 0.65, §49), which is why the rest of verdict asks surface questions.

Two things about the 70.8% still hold as method:

- It is fitted on **expected cost, not accuracy**. A hard prompt sent to the small model counts
  three times an easy one sent to the big model. Tuning on plain accuracy scored four points
  higher while nearly tripling the expensive mistake (§15).
- The questions are upstream's, not invented here. Phrased the way the checkpoint was trained,
  they scored AUC 0.88 against 0.81 for invented ones (§14).

Reproduce with `uv run python -m verdict.evaluation.router_eval --refit`, or gate it with
`apol validate --all --no-run`.
