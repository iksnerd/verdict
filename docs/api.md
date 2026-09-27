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
  disables this preprocessing cap, but the checkpoint still has a finite context). A state within
  budget passes through still structured. These are local Laya settings, not upstream Jev limits.
- Optional `"model": "multilingual"` answers from Laya's multilingual checkpoint, loaded on first
  use. It is named as in Laya's `Router.predict(model=...)`. A server without one configured returns
  400.
- `[model.extra]` in `verdict.toml` names further checkpoints the same way (`support = "org/support-mlx"`),
  each loaded on first request that names it and cached after. An unconfigured or misspelled name
  is also a 400. `GET /v1/models` lists whatever is configured. Both `/v1/decide` and `/v1/systemone` can select them.
- A `noul` question may describe its sides with `criteria: {"true": ..., "false": ...}`. On this
  project's data that hurt (§22), so leave them out unless you've measured otherwise.

It answers as `verdict decide` does, so the same question gets the same number from either:

- A question naming a field the state lacks (a plain-text state has none), or shaped like the
  questions that measured at chance (a consequence, difficulty or absence; more than 20 options),
  is refused with a 422 and the reason. `?allow_unmeasured=true` asks anyway.
- A new yes/no is asked as a choice between `no` and `yes` and answered as a `noul`, P(yes), which
  ranked as well or better on every set measured (§38, §40). `?yesno=true` sends it as a plain
  yes/no. A library question keeps the shape it was measured in on the checkpoint it was measured on.
- A proxy backend (`SystemOneBackend`) forwards the request as given: these findings are about Laya.

## `POST /v1/systemone` and `GET /v1/models`: Jev's protocol

The same answers on [TypeSafe's Jev](https://docs.typesafe.ai/api) wire protocol, so tools written
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

- The request is `/v1/decide`'s plus a required `model`. On a local Laya backend, `jev-latest`,
  `jev-preview`, `jev-1.13`, and `jev-1.13.0` are explicit compatibility aliases for the served
  checkpoint. They do **not** load Jev; the response names the actual Laya checkpoint. `english`,
  `multilingual`, and configured extras select local checkpoints. Unknown names return 400.
  An explicitly configured extra takes precedence over a compatibility alias.
  A SystemOne proxy forwards the requested model unchanged, including version-pinned Jev IDs.
- Yes/no questions are rewritten as on `/v1/decide`, so the two endpoints return the same answers.
  Nothing is refused here: Jev's protocol has no way to pass the opt-out, and its SDKs expect an
  answer, so check a bank with `verdict validate` first.
- The response adds `usage`. `input_tokens` counts what the model read, which is the state once per
  question (laya re-reads it for each, §11), so it is not Jev's billing figure. `output_tokens` is 0.
- Local `instructions` may be text, JSON, or left out (Laya receives an empty string). The Jev
  profile requires instructions as TypeSafe's current HTTP reference specifies. The local bearer
  key is ignored: the server binds to localhost.

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
checkpoint). It validates the shared wire shape while retaining provider-specific values:

```python
import uvicorn

from verdict.api import create_app
from verdict.backend_systemone import SystemOneBackend

with SystemOneBackend("http://127.0.0.1:8009", model="kev-latest", key="local") as backend:
    uvicorn.run(create_app(backend), host="127.0.0.1", port=8799)
```

Use the backend directly as a typed Python SDK, with a context manager to close its connections:

```python
import os

from verdict.backend_systemone import SystemOneBackend
from verdict.schema import DecideRequest
from verdict.answers import max_probability

with SystemOneBackend("https://api.typesafe.ai", model="jev-1.13.0",
                      key=os.environ["TYPESAFE_API_KEY"]) as backend:
    print(backend.capabilities().model_dump())
    result = backend.decide(DecideRequest(
        state={"message": "Please refund the duplicate charge."},
        questions={"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    ))
    answer = result.answers["refund"]
    print(result.model, answer.noul, answer.confidence)  # confidence is None when unreported
    print(max_probability(answer))  # explicitly derived max(p, 1-p), not provider confidence
```

`request.model` overrides the constructor's default. The state and question types are forwarded
without clipping, Noul-to-Choice rewriting, or fallback to a local model. The official
`api.typesafe.ai` host selects the Jev capability profile automatically. For a custom Jev gateway,
pass `profile="jev"`; other SystemOne servers use `profile="systemone"`, with unknown limits.
A model named `jev-latest` on localhost does not by itself establish that the server runs Jev.

Missing Noul `confidence` remains `None` in Python and is omitted from HTTP responses. Values
actually reported by the provider are preserved. Missing token counts also remain unknown and
are omitted over HTTP, rather than becoming fabricated zeros. Choice/Score confidence is
provider-specific; thresholds do not transfer between Laya and Jev. `max_probability()` has a
consistent mathematical definition across backends, but still requires calibration on your data.

`GET /v1/models` forwards the upstream list; an unavailable or malformed listing becomes an empty
list. Inference failures never become empty answers: transport failures and transient statuses
(408, 429, 500, 502, 503, 504, 529) retry with exponential backoff, honoring `Retry-After` seconds
or HTTP dates. Authentication and validation failures are not retried. Responses must contain
exactly the requested answer IDs and types, finite probabilities matching the criteria, normalized
distributions (allowing four-decimal rounding), valid selections, and scores consistent with their
rubric and probabilities. Malformed responses and exhausted retries raise `UpstreamError`, a
`ValueError` subclass; the HTTP proxy returns 502. Locally detected Jev question-limit violations
return 422. Error messages omit upstream bodies to avoid echoing request data.

The backend owns its HTTP client; use `with` or `close()`. `create_app(backend)` leaves ownership
with its caller. The `bench` and `examples` commands close their clients on both success and error.

## Discovering backend differences

`GET /v1/capabilities?model=support` is a Verdict extension, kept separate from Jev's model-card
format. In Python, use `backend.capabilities(model)` or
`verdict.client.capabilities(url="http://127.0.0.1:8799", model="support")`. Discovery does not
load local weights or send upstream inference. `null` means unknown, **not unlimited**. A direct
TypeSafe endpoint or older Verdict may not expose this extension; the HTTP helper raises
`NoServer` when discovery is unavailable, without triggering an inference fallback.

| Property | Local Laya | Hosted Jev profile |
|---|---|---|
| State handling | Default server preprocessing cap: 128 tokens; checkpoint may truncate further | Sent unchanged; upstream enforces token limits |
| Context | Checkpoint-specific; reported only when available from a loaded model | 64k total, 32k for state plus longest question |
| Choice options | 20 recommended by Verdict's measurements, not a protocol maximum | 255 maximum |
| Score levels | No additional Verdict maximum | 10 maximum |
| Question execution | One pass per question | Parallel against shared state |
| Noul confidence | `max(p, 1-p)` | Absent |
| Choice/Score confidence | `1 - normalized entropy` | Provider-defined distribution statistic |
| Usage | State counted per question; output tokens zero | Provider-reported counts, not interchangeable with local counts |

Jev's limits are a documentation snapshot checked **2026-09-27**, not a local Jev tokenizer or a
promise about future versions. Unknown SystemOne implementations do not inherit Jev's limits.
The 20-option Laya recommendation is exposed separately from hard limits; the Python/HTTP SDK
reports it, while the Laya-oriented CLI refuses larger choices unless `--allow-unmeasured` is set.

Sources: [TypeSafe models](https://docs.typesafe.ai/models),
[API contract](https://docs.typesafe.ai/api), [confidence](https://docs.typesafe.ai/confidence).
Protocol compatibility does not establish equal accuracy, calibration, language support, or
resistance to adversarial text. Benchmark the same labelled states on both backends and refit
thresholds for the actual model version, question shape, and preprocessing budget.

### Migration from earlier Verdict versions

- Fix misspelled local model names: they now fail instead of silently choosing the default.
- A proxied Noul's `confidence` can be absent. Use `answer.noul` for P(yes), or explicitly call
  `max_probability(answer)` if you need that statistic. Local Laya output remains unchanged.
- Unknown upstream token counts are omitted; do not interpret absence as zero usage.
- Malformed upstream decisions are rejected instead of flowing into downstream thresholds.
- Ordinary `ask`/`decide` CLI paths still apply Laya-specific rewrites and warnings. For Jev,
  use `SystemOneBackend` directly or the explicit `bench/examples --systemone` paths.

The upstream backend is not yet wired into `verdict serve`'s CLI or `verdict.toml`: today it is a `Backend` you
construct yourself in a custom entry point, the same way `MlxBackend` is. `verdict examples
NAME... --systemone URL` runs `examples/`'s own real inputs against it, reports ms/call, and where
a row carries an `expected` label (most banks do; `examples/README.md` gives each one's
count) prints how many match it — a new backend gets a real accuracy and
latency number against the same rows `examples/README.md` already documents for Laya, not a
one-off question and an eyeballed guess at whether it's "close enough".

## Other endpoints

| endpoint | does |
|---|---|
| `GET /healthz` | `{"status": "ok", "backend": "laya-mlx:..."}`. A backend of `uniform` means no model: treat it as absent, never as "unsure" |
| `POST /v1/route` | `{"prompt": ...}` in, the big-or-small branch out ([routing.md](routing.md)) |
| `POST /v1/route/batch` | `{"prompts": [...]}` in, one result per prompt with its `index`; at most 512 |

`uvicorn verdict.api:app` serves the `uniform` backend, not the model. Use `verdict serve`.
