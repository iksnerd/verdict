# HTTP API

← [Back to the README](../README.md)

`verdict serve` exposes the same decisions over HTTP on `127.0.0.1:8799`. It binds to localhost and
has no auth, so do not expose it beyond the machine. The wire format mirrors Laya's
`Agent.predict(state, questions)`, so a Laya checkpoint drops in as a backend with no translation,
and Laya's mirrors TypeSafe's Jev, which `/v1/systemone` serves.

## `POST /v1/decide`

```json
{
  "state": {"message": "I was charged twice, please refund"},
  "questions": {
    "intent": {"type": "choice", "instructions": "What does the customer want in `message`?",
               "criteria": {"refund": "money back", "info": "a question", "other": "none fit"}},
    "urgent": {"type": "noul",   "instructions": "Is there time pressure in `message`?"},
    "anger":  {"type": "score",  "instructions": "How angry is `message`?", "criteria": ["calm", "annoyed", "angry"]}
  }
}
```

The response has one answer per question:

- a `choice` answer carries `choice`, `probabilities` per option and `confidence`
- a `score` answer carries `score` (the expected level), `legend`, `probabilities` and `confidence`
- a `noul` answer carries `noul`, the probability the statement holds

Plus `model`, the checkpoint that answered.

- `state` is a string, or better an object with named fields that the questions name in backticks.
- Each state is clipped to the server's token budget (128 by default; `verdict serve --budget 0`
  reads it whole). A state within budget passes through still structured.
- Optional `"model": "multilingual"` answers from Laya's multilingual checkpoint, loaded on first
  use. It is named as in Laya's `Router.predict(model=...)`. A server without one configured returns
  400.
- `[model.extra]` in `verdict.toml` names further checkpoints the same way (`support = "org/support-mlx"`),
  each loaded on first request that names it and cached after. An unconfigured or misspelled name
  is also a 400. `GET /v1/models` lists whatever is configured. These are a `/v1/decide` feature
  only: they are verdict's own vocabulary, not something Jev's protocol has a name for, so
  `/v1/systemone` cannot reach them (below).
- A `noul` question may describe its sides with `criteria: {"true": ..., "false": ...}`. On this
  project's data that hurt (§22), so leave them out unless you've measured otherwise.

## `POST /v1/systemone` and `GET /v1/models`: Jev's protocol

The same answers on [TypeSafe's Jev](https://docs.typesafe.ai/api.md) wire protocol, so tools written
for Jev can use verdict instead: the official `typesafe-sdk` (Python) and JavaScript SDKs, LiteLLM's
TypeSafe passthrough, TypeSafe's cookbooks. Point them here and give any key:

```sh
verdict serve &
export TYPESAFE_BASE_URL=http://127.0.0.1:8799 TYPESAFE_API_KEY=local
```

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

r = TypeSafeClient().system_one(
    state={"message": "I was billed twice. Refund the duplicate or I cancel."},
    questions={"dept": Choice(instructions="Which team?", criteria={"billing": "refunds", "tech": "bugs"}),
               "refund": Noul(instructions="Does the customer ask for money back?")},
)
# dept billing 0.78, refund 0.74 (real output, typesafe-sdk 0.7.1)
```

- The request is `/v1/decide`'s plus a required `model`. `jev-latest`, and any name verdict does not
  know, selects the served checkpoint; `english` and `multilingual` select those. `GET /v1/models`
  lists them.
- The response adds `usage`. `input_tokens` counts what the model read, which is the state once per
  question (laya re-reads it for each, §11), so it is not Jev's billing figure. `output_tokens` is 0.
- `instructions` may be text, JSON, or left out, as Jev allows. The bearer key is ignored: the
  server binds to localhost.

What does not carry over: Jev takes up to 255 choice options, and laya degrades past about 20
(its options share a 192-token budget). Jev's fan-out pattern, many questions in one call, is one
pass there and one pass per question here (§11). `verdict bench --systemone URL` scores any
`/v1/systemone` endpoint on the same suites, Jev included with `TYPESAFE_API_KEY`.

## Swapping the backend to another /v1/systemone server

`SystemOneBackend` (`backend_systemone.py`) is a `Backend` that forwards `decide()` to any other
`/v1/systemone`-compatible server instead of running a local checkpoint: another `verdict serve`,
a real hosted Jev, or an independent open model whose server speaks the same contract (validated
against [Kev](https://github.com/jaredpalmer/kev), a real Qwen3.5-based decision model with its
own `/v1/systemone` server — a genuinely different architecture from laya's, not another laya
checkpoint). It needs no translation layer, only typing the raw JSON, since the wire shape is
already ours:

```python
from verdict.api import create_app
from verdict.backend_systemone import SystemOneBackend

backend = SystemOneBackend("http://127.0.0.1:8009", model="kev-latest", key="local")
uvicorn.run(create_app(backend), host="127.0.0.1", port=8799)
```

A `noul` answer missing `confidence` (Jev's own contract has none there, and Kev's matches it) is
synthesized as `abs(noul - 0.5) * 2` — Kev's own documented Choice-confidence formula,
`(p_max - 1/K)/(1 - 1/K)`, read at K=2, not an invented number.

`GET /v1/models` forwards the upstream server's own list rather than the generic
`jev-latest`/`english` cards, since those describe laya/Jev's fixed vocabulary and would
misdescribe whatever this is actually proxying to; a listing failure degrades to an empty list
rather than breaking the endpoint.

A server that just started can take longer than one request to answer while it loads weights, and
at least one (Von, on its very first request against a cold cache) has been observed to drop the
connection entirely rather than answering slowly. A timeout or connection failure is retried the
same as a 429/529, with a line on stderr each time ("...it may still be loading -- retrying
(N/5)") so a slow load reads as progress, not a hang. Once retries are exhausted, or the upstream
answers with a bad status, `decide()` raises a plain message naming the possibility rather than
the raw exception; `verdict serve` turns that into a 502, not a 500 with a traceback, since it is
a proxy here and the failure is the upstream's. `verdict examples`/`verdict bench --systemone`
both turn the same failure into a clean CLI error (exit 2) naming which row it happened on,
instead of an unhandled Python traceback.

This is not yet wired into `verdict serve`'s CLI or `verdict.toml`: today it is a `Backend` you
construct yourself in a custom entry point, the same way `MlxBackend` is. `verdict examples
NAME... --systemone URL` runs `examples/`'s own real inputs against it, reports ms/call, and where
a row carries an `expected` label (citation-check, checklist-check, task-verification and
passage-filter all have one) prints how many match it — a new backend gets a real accuracy and
latency number against the same rows `examples/README.md` already documents for Laya, not a
one-off question and an eyeballed guess at whether it's "close enough".

## Other endpoints

| endpoint | does |
|---|---|
| `GET /healthz` | `{"status": "ok", "backend": "laya-mlx:..."}`. A backend of `uniform` means no model: treat it as absent, never as "unsure" |
| `POST /v1/route` | `{"prompt": ...}` in, the big-or-small branch out ([routing.md](routing.md)) |
| `POST /v1/route/batch` | `{"prompts": [...]}` in, one result per prompt with its `index`; at most 512 |

`uvicorn verdict.api:app` serves the `uniform` backend, not the model. Use `verdict serve`.
