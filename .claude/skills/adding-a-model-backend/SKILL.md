---
name: adding-a-model-backend
description: >-
  Evaluate and wire in a new decision model as a verdict Backend — a genuinely different
  architecture, not another laya checkpoint (for that, see [model.extra] in docs/api.md). Use
  when asked to add, try, compare, or benchmark an alternative model against verdict's own laya
  backend, especially one found via a blog post, a "Jev alternatives" list, or a link someone
  pasted in. Covers verifying the model is real before spending effort on it, sizing the resource
  cost before downloading anything, choosing whether it needs a translation layer or can reuse
  SystemOneBackend, and comparing it against examples/ honestly.
---

# Adding a model backend

A `Backend` is anything with `name` and `decide()` (`src/verdict/backend.py`). Two ship today:
`UniformBackend` (no model, for tests) and `MlxBackend` (laya). This skill is for adding a third
kind — a different architecture entirely, not a different laya checkpoint. `[model.extra]` in
`verdict.toml` already covers "another laya checkpoint"; reach for this skill only when the model
itself is not laya.

## 1. Verify it is real before spending any more effort on it

Every session that has gone looking for "Jev alternatives" has found at least one fabricated
entry. The failure mode is specific: a blog or listing page returns a bot-challenge or 403, and
whatever fetched it (a summarizer, a scraper) produces a fluent, plausible-looking table anyway
instead of reporting the failure. The table can mix one real, well-known name with several
invented ones, which is what makes it convincing.

Do not trust a fetched summary. Check primary sources directly:

- `curl -sL -A "Mozilla/5.0 ..." <url> -o page.html` and read the actual status/title. A "Just a
  moment..." title or a 403 means whatever summarized it never saw the real page either.
- Cross-check the named model against the real Hugging Face API: `curl -s
  "https://huggingface.co/api/models?search=<name>"` — a real model has real download/like
  counts and, usually, independent community forks or quantizations (ONNX/GGUF/CoreML conversions
  by people who are not the original author). A fabricated name returns nothing, a parse error,
  or unrelated fuzzy matches.
- Check the real GitHub org/repo the same way (`gh api users/<org>`, `gh api
  repos/<org>/<repo>`), and read the actual README with `curl`, not a fetch tool's summary of it.
- A real, actively maintained project reads like one: dated changelogs, PRs, an issue tracker, a
  citation or reference to where the idea came from, honest caveats about what got *worse* in a
  new version. Extremely detailed, self-critical model cards (confidence intervals, "this number
  is unexplained, not a skill") are a good sign, not a bad one — compare the tone to this repo's
  own `docs/FINDINGS.md`.

If it doesn't check out, say so and stop. Don't build anything on an unverified claim.

## 2. Size the resource cost before downloading anything

Check what disk, RAM, and process footprint the model actually needs — not what a marketing
paragraph implies. Read the model card and the repo's own quickstart for real numbers:

- Base weights size (unquantized), and whether a merge step needs *both* the base and the merged
  copy on disk at once (roughly doubles the transient disk need).
- Whether it runs natively on Apple Silicon (MLX, or PyTorch's MPS backend) or is CUDA-first with
  Apple support bolted on separately, or missing.
- Params as a multiple of laya's own footprint (322–421M, ~800 MB) gives a quick gut check: a few
  hundred MB to low single-digit GB is comparable to what this repo already runs; a 9B-parameter
  model with an 18 GB base and a separate merge step is a different scale of commitment and
  deserves an explicit go-ahead before downloading, not just a mention in passing.
- `ollama ps`, `pgrep -fl 'verdict serve'`, and `lsof -iTCP:<port> -sTCP:LISTEN` before starting
  anything, matching "one model at a time" (CLAUDE.md, CLAUDE.local.md). This applies to whatever
  process serves the new model too, even though it runs outside verdict's own process.

When the honest answer is "this is a much bigger commitment than anything loaded so far," say
that plainly and let the user decide, rather than starting a multi-gigabyte download on a vague
"try it" instruction. Two real examples from one session: a 9B/18GB Nimble checkpoint (declined,
too big) against a 0.8B/1.8GB Kev checkpoint (built and live-tested) for the same request.

## 3. Decide: translation layer, or reuse `SystemOneBackend`?

Check whether the model's own server already speaks TypeSafe's `/v1/systemone` contract (`state`,
`model`, typed `questions` in; `choice`/`score`/`noul` answers out). Some Jev-inspired open models
do, because that is the point of the exercise for their authors too.

- **If it already speaks `/v1/systemone`**: no translation layer is needed. Run its own server
  locally and point `SystemOneBackend` (`src/verdict/backend_systemone.py`) at it —
  `SystemOneBackend(base_url, model=..., key=...)` reuses `bench.py`'s already-tested
  `systemone_asker` for the HTTP (auth, retries) and only adds typed-answer construction. Check
  whether its `noul` answer omits `confidence` the way Jev's real contract does (ours requires
  one; `_answer()` synthesizes `abs(noul - 0.5) * 2` when it's missing — Kev's own documented
  Choice-confidence formula read at K=2, not an invented number) and whether it has its own
  `GET /v1/models` to forward (`SystemOneBackend.models()`, api.py's `/v1/models` route already
  prefers `backend.models()` over the laya-specific default cards when the backend has one).
- **If it does not** (a raw scoring library, a different wire shape, something that returns
  logits instead of a typed JSON answer): that is a real, separate integration project — writing
  a new `Backend` with its own state/questions translation, the way `backend_mlx.py` translates
  for laya. Scope and confirm that separately; it is not a smaller version of the same task.

## 4. Test before you run anything real

TDD, at the `Backend`'s own seam (`create_app(backend)` via `TestClient`, or `backend.decide()`
directly), mocked first:

- Reuse `httpx.MockTransport` the way `tests/test_bench.py`'s `jev_transport` and
  `tests/test_systemone_backend.py` already do — a fake server that returns the raw JSON shape,
  not a fake model.
- Cover the adaptation logic specifically: confidence synthesis, `usage` passthrough, which
  server name gets reported (`raw.get("model", ...)`, so the caller sees the real upstream
  checkpoint, not just this repo's own label for it), and any answer type the upstream is known
  to omit fields from.

Then, once mocked tests pass, run the real thing once and confirm live — this is not optional.
Every fix this session that mattered was confirmed against a real server, not just mocks: a
synthesized confidence value that happened to equal the mock's hand-picked number would prove
nothing; matching a formula on live, unscripted model output is the actual check.

**A cold server can crash or hang on its first real request.** `bench.systemone_asker` (shared by
`SystemOneBackend`, `verdict examples --systemone`, and `verdict bench --systemone`) already
retries a timeout or connection failure the same way it retries a 429/529, printing a line on
stderr each time ("...it may still be loading -- retrying (N/5)") so this reads as progress, not
a hang; `verdict serve` reports a persistently failing upstream as a 502, and the CLI commands
fail cleanly (exit 2, naming the row) rather than an unhandled traceback. You get this for free by
building on `systemone_asker`/`SystemOneBackend` rather than a bespoke HTTP client — don't
re-implement retry logic in a new backend; a `Backend` that talks to a fundamentally different
wire shape (step 3's translation-layer case) still needs its own equivalent handling, deliberately,
not by accident. If a cold start is still slow enough to matter, run one throwaway request first
(any single question) before the real comparison, so the retries in the log belong to warm-up,
not to the run you're about to report.

## 5. Compare it against `examples/` honestly

`verdict examples [NAME...] --systemone URL --systemone-model NAME` runs the real, invented
inputs in `examples/*/bank.json` + `*.jsonl` against any `/v1/systemone` backend, the same real
inputs `examples/README.md`'s own Laya numbers came from, and reports ms/call. Use it instead of
one-off ad hoc questions — a handful of hand-picked questions can't tell you where a new model's
actual weaknesses are; the existing example banks were built (and are checked by
`tests/test_examples.py`) precisely to cover several different question shapes at once.

Where a row carries an `expected` label (most banks do; `examples/README.md` gives each one's
documented count), it prints how many answers match it — grade against that, not
against how confident the answer looks. **Low confidence is not the same as wrong.** One session
read a backend's uniformly low confidence on `checklist-check` as "not working" without checking
it against the recorded labels; graded, it was actually *more* accurate than Laya's own documented
number on that example, just less confident about being right. A model that is right 80% of the
time at 0.3 confidence and one that is right 50% of the time at 0.3 confidence look identical
until you grade them.

Report what you find plainly, including where the new model does *worse*, not just where it's
comparable — that is this repo's own convention (`secret-commands/`'s miss, `commit-kinds/`'s
counter-example) and it is more useful than a clean-looking summary. A real result from one
session: a candidate model defaulted to "unrelated" on 12/12 rows of a three-way relation task
where laya got 10/12 right with a real mix of answers — a genuine, specific weakness, not a
close-enough approximation, and worth saying so.

## 6. Update docs in the same change

- `docs/api.md`: what the backend is, what it needs, what carries over and what doesn't (see the
  "Swapping the backend to another /v1/systemone server" section as the template).
- `docs/guide.md`'s "running it without cooking the laptop" list: the new process's own resource
  footprint, since it runs outside verdict's own memory but "one model at a time" still applies to
  it.
- `examples/README.md`: a line pointing at `verdict examples --systemone` if that's how it was
  validated.

## 7. Clean up

Stop whatever server process you started, by the PID you captured or the one listening on its
port (never by name: that also stops another session's), confirm the port
is free, and leave cloned repos in the session scratchpad rather than inside this checkout unless
asked to keep them. Nothing from this exercise should still be running, or still be downloading,
after you report the result.
