# Concepts

← [Back to the README](../../README.md)

Four words carry the whole tool: a **state**, a **question**, an **answer** and a **bank**. This page
says what each is. For the commands, see the [guide](guide.md); for the wire format, the
[API reference](api.md).

## State

A state is the thing you are asking about. It is text, or, better, a JSON object with named fields:

```json
{"message": "I was charged twice", "plan": "pro"}
```

Laya serializes an object as JSON, and a question refers to a field in backticks: "What does the
customer want in `message`?" A question that names a field the state does not have is refused,
since the model would answer from nothing (`--allow-unmeasured` asks anyway). The same goes for a
question about a field on a plain-text state, which has none: pass the text as `{"message": ...}`.

The model reads the first 128 tokens of a state. Put the field that decides first. A state past
that is cut, verdict warns once per run, and every answer of a cut call carries `"truncated": true`
(the HTTP API and `decide` both; `/v1/systemone` leaves it out, since Jev's answer types are
strict). See [rough edges](rough-edges.md).

## Question

A question has a type and, usually, instructions:

| Type | Asks | `criteria` |
|---|---|---|
| `noul` | does the statement hold (yes/no) | optional; leave it off (see below). `labels` can name the two sides, unmeasured |
| `choice` | which one of these options | the options, as names or `name: description`; at least 2 |
| `score` | how much, on an ordered scale | the levels, lowest first; at least 2 |

Each question is atomic: one narrow thing about what the text shows. Score several narrow questions
and combine the answers in code, rather than writing one question that weighs several things. Eight
questions folded into one scored below the best single one (FINDINGS §27), and each question costs a
forward pass, about 30 ms. See [patterns](patterns.md).

Two limits are enforced, not advised. A flat `choice` takes at most 20 options; past that, try
`verdict classify` with a tree, and check it against the flat choice on a labelled sample, since a
tree is not always better (FINDINGS §58). Questions measured at chance (consequences, difficulty, an absence
instead of a property) are refused. The reasons are in [rough edges](rough-edges.md).

Describing both sides of a yes/no with `criteria` dropped `is_sensitive` from AUC 0.813 to 0.633
(§22). Leave it out unless you have measured otherwise.

## Answer

An answer is typed, never generated text, so there is nothing to parse:

- `noul`: P(yes), 0 to 1.
- `choice`: the picked option and a probability for each.
- `score`: the expected level and a probability for each level.

Every Laya answer also carries a `confidence`. The probabilities rank items; they are not calibrated, and
0.5 is not the line between yes and no for a yes/no. Fit a cut on your own labelled examples with
`verdict calibrate` before gating anything on a number (§55). `verdict` answers and never acts on an
answer: what to do with a number is the caller's code.

## Bank

A bank is a set of questions, keyed by an id of your choosing, asked of the same state:

```json
{"urgent": {"type": "noul", "instructions": "Is `message` urgent?"},
 "wants":  {"type": "choice", "instructions": "What does the customer want in `message`?",
            "criteria": {"refund": "money back", "info": "an answer", "other": ""}}}
```

`verdict decide STATE -q bank.json` answers every question of the bank for one state;
`decide --jsonl` streams the bank over many states. `verdict questions` lists the measured library,
and `-q is_instruction,touches_secret` asks questions from it by name. `verdict validate -q bank.json`
checks a bank with no model loaded.

## Where the pieces run

The CLI, the localhost HTTP API and the Python client all take the same states and banks and give
the same answers. A backend turns a state and a bank into answers: local Laya, TypeSafe's Jev, or
any `/v1/systemone` server. The picture is in [architecture.mmd](../design/architecture.mmd).
