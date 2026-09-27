# Project: verdict

A typed decision API and the pipeline that trains its model. See `README.md` for what is built
and `docs/FINDINGS.md` for what has actually been measured.

Public on GitHub as `iksnerd/verdict`, Apache-2.0. The fine-tuned weights are not: they stay in the private Hugging Face
repo `iksnerd/verdict-v1-mlx` (trained on Gemini labels). Never attach them to a GitHub release.
History is pushed, so do not rewrite it without asking.

## Stack
- Python 3.11, `>=3.11.4,<3.12` (pinned: the ML wheels this project needs have no 3.14 build), managed by uv
- pydantic 2, FastAPI, pytest

## Conventions
- The wire format in `src/verdict/schema.py` mirrors laya's `predict(state, questions)`.
  Change it only if laya's changes; the dataset, teachers, model and API share it. laya's in turn
  mirrors TypeSafe's Jev, and `POST /v1/systemone` + `GET /v1/models` serve Jev's protocol so its
  SDKs work via `TYPESAFE_BASE_URL`. `tests/test_systemone.py` pins that shape (read from
  `typesafe-sdk` 0.7.1's wire models, whose answer types are strict: floats must be floats).
- Every backend implements `Backend` (`name`, `decide`). Anything new plugs in there.
- verdict is a System 1 (Kahneman): fast typed judgments for a System 2, a person or an agent,
  to act on. It answers and never acts on an answer: no dispatch, no model-to-model routing,
  nothing run because of a verdict. Anyone who wants an answer to launch something writes that in their own script.
  The only commands that run anything are maintenance: `verdict update` (in a checkout, `git pull
  --ff-only` plus `uv sync --extra mlx --extra laya`; in a `uv tool` install, a reinstall of the
  newest `v*` tag).
- Tests first. `uv run pytest` is the gate; nothing needs the network or a model.

## The one that matters

The shipped router (`difficulty` + `is_sensitive`) is **at chance on real agent traffic**, AUC 0.478,
against 70.8% on the 72 hand-written prompts it was tuned on (§25). The diagnosis is not the model
and not the fine-tune: the same checkpoint scores **0.753** on the same turns asked
`is_instruction` instead (§26). Ask about what is on the page, not about how hard a task would be
for a different model. Real outcome labels are free in agent transcripts: each one records which
tools the agent called after each turn.

## What has been ruled out

Do not re-derive these. Each is measured, in `docs/FINDINGS.md`.

- **Swapping to another Laya checkpoint** (§23). `laya-typed-decisions` and `laya-multilingual`
  both score at or below our own fine-tune on the real ledger task, and both have 1024-token
  context, so truncation is not what §10 was about. On the public bench (§35) the fine-tune wins
  only on yes/no; base laya ties it on choices and is the fallback when the weights are missing.
- **Batching several states per forward pass** (§34). At best 1.5x, noisy, sometimes 0.5x: on
  an Apple Silicon laptop the forward pass is the whole cost, so throughput is set by the checkpoint and budget.
- **Core ML and the Neural Engine** (§24). Slower than MLX for this encoder by the port's own
  published table, and any call mixing two question types costs about 850 ms against 31 to 42 ms
  for a same-type pair. Our bank is `score` + `noul`, the mixed case. Filed upstream as
  `mizorewww/laya-coreml#5`.
- **Connection reuse** (§20). Measured at 3% *slower*: a loopback handshake is free against 33 ms
  of inference.
- **Describing the `noul` sides** (§22). It drops
  `is_sensitive` AUC from 0.813 to 0.633.

## Constraints
- Never gate a decision on a `uniform` backend answer. It means "no model", not "unsure".
- Teacher labels come from Gemini only. Local Ollama models generate states, never labels.
- Real data (private message logs, agent transcripts) is a sanity test set, never training data,
  and is not sent to a hosted API without asking. Never quote it in tracked files: FINDINGS reports
  aggregates, and example inputs are invented.
- Do not commit `.env`, `data/`, `models/`, `runs/`, `apol.db` or `experiments/`. The last two
  are not obvious: `runs/` holds large binary `.pt` items, and `experiments/` holds scripts that
  read private data.
- The repository is public. Machine-specific notes go in `CLAUDE.local.md` (gitignored), and
  `tests/test_public_repo.py` fails on home paths and private tool names in anything that ships.
- Keep the API bound to localhost.
- Commit and push only when asked.

## Commands
- `uv sync --extra mlx --extra laya` then `uv run pytest`. Plain `uv sync` *removes* both extras,
  which silently skips the laya-dependent tests (including the guard that the vendored router bank
  still matches `laya.router_questions()`) and breaks the CLI and the scorer.
- `git config core.hooksPath .githooks` once per clone enables the pre-commit hook. It blocks the
  never-commit paths below, runs gitleaks on the staged diff, and runs the suite against an export
  of the index (not the working tree, and without stashing, so peers are undisturbed). About 4 s.
- The agent skill ships from this repo as a plugin: `.claude-plugin/marketplace.json` and
  `plugins/verdict/skills/verdict/`. `tests/test_plugin.py` ties the plugin version to
  the package version and refuses private paths in what ships.
- Releases: a `vX.Y.Z` tag with a committed scorecard (`bench/scorecards/vX.Y.Z.json`, run at
  normal priority); `.github/workflows/release.yml` runs the suite on macOS arm64 on tags only
  (macOS is the only runner with the mlx and laya extras, and downloading model weights plus the
  full suite is real wall-clock time to spend on every push; the pre-commit hook gates day-to-day
  changes instead), checks the tag against the version and refuses a missing or regressed
  scorecard. The default model is base Laya (`aac6fef/laya-mlx`); the fine-tune is private, in
  the Hugging Face repo `iksnerd/verdict-v1-mlx`, used only where `model.path` points at it.
  Scorecards come from the release machine's configured model (the fine-tune).
- Docs: README (landing), `docs/guide.md` (usage), `docs/api.md`, `docs/routing.md`,
  `docs/pipeline.md`, and
  `examples/` (synthetic inputs, real outputs, checked by `tests/test_examples.py`). When a command
  or measured number changes, update the guide and the examples README in the same change.
- `verdict update` fast-forwards this checkout and re-syncs with both extras, refusing on
  uncommitted changes; `verdict update --check` only reports (exit 1 when behind). Prefer it to a hand-typed `uv sync`,
  which drops the extras.
- `[model].lang = "multi"` (`$VERDICT_LANG`, overridden by `--lang`) reads every state with laya's
  multilingual checkpoint; the English one is near chance on Cyrillic (§37). Bench suites with
  `lang: multi` (bg-, ru-) are asked with it, so a release scorecard loads both checkpoints.
- `[model].bits = 8` (`$VERDICT_BITS`, `serve --bits`) quantizes the fine-tune at load, as §36
  measured it (group 64, `act_head` left FP16, in `engine.quantize`): half the memory, same answers,
  no speedup. Only 16 and 8 are accepted. Scorecards record `bits`.
- `uv run verdict init` writes `~/.config/verdict/config.toml` from what the machine actually has
  (`--out verdict.toml` for a per-directory one, gitignored), and fetches the weights if missing.
  Settings resolve flag > environment > file > built-in default, and there is deliberately no
  `[thresholds]` key: the cuts live in the server process, so a file value would be ignored.
- `uv run verdict serve` first: it loads the checkpoint once, and `verdict route` then costs
  70 ms instead of 2.1 s. **`uvicorn verdict.api:app` is not the same thing**: it serves the
  `uniform` backend, which is worse than no server, because the client treats it as absent and
  falls back after paying a round trip.
- `uv run verdict route "<prompt>"`, `uv run verdict cases`,
  `uv run verdict route --batch < prompts.txt`
- `verdict ask "<text>" "<question>"` (`-o` choice, `-l` score, `--cut N|FILE`), `verdict decide`
  (JSON state; `--questions` preset, library names comma-separated, JSON or file; `--jsonl`;
  `--calibration`; `--server-only` to fail instead of loading locally), `verdict validate -q BANK
  [--json]` (checks a bank with no model), `verdict questions` (the measured library), `verdict presets`,
  `verdict calibrate`, `verdict bench [--verify CARD]`, `verdict examples [NAME...] [--systemone URL]`
  (scores `examples/` banks with a backend, real or `--systemone`, for comparing one against another),
  `verdict rank`, `verdict docs`. The plugin's `verdict` skill says
  when an agent should reach for it.
- The everyday `verdict` on PATH is a `uv tool` install of a release tag, not this checkout:
  `uv tool install --python 3.11 'verdict[mlx,laya] @ git+https://github.com/iksnerd/verdict.git@vX.Y.Z'`.
  `verdict update` moves it to the newest tag. Its weights live in `~/.local/share/verdict/models/`,
  and `models/verdict-v1-mlx` here is a symlink to them, so the pipeline and `uv run verdict` in
  this repo still find them. `~/.config/verdict/config.toml` points at the absolute path. Changes
  here reach the global CLI only through a release (a `v*` tag).
- The CLI's output is an interface: scripts outside this repo run `verdict decide --jsonl` and
  parse `answers.<qid>.noul` from each stdout line. A change to that output must keep the shape.
- `uv run verdict decide "<state>" --questions <file.json>` is the general path: your own typed
  questions, nothing routed and nothing dispatched. `--check` tokenizes the options to report exact
  truncation and costs about 1.3 s, so it is opt-in; without it a character pre-filter warns.
- `uv run python -m verdict.router_eval --refit`


## Running models locally

A laptop GPU is shared by everything on it, and several models loaded at once plus thousands of
back-to-back calls can overheat it.

- **One model at a time.** Before `verdict serve`, a scoring run or anything that loads weights,
  check what is already resident (`ollama ps`, `pgrep -fl 'verdict serve'`, anything listening on
  the server port). If something else is loaded, wait for it or ask. Do not stack another model.
- **Sample before sweeping.** Score 200 to 400 states, look at the numbers, and only then decide
  whether more are worth it. 400 labelled items settled every question in §26 to §28.
- **Throttle long runs.** Pause between calls (`time.sleep(0.05)` or more) and run the script
  under `taskpolicy -b`, so the machine stays usable and the fans stay quiet.
- **Stop what you started.** Kill `verdict serve` and any scoring process when the measurement is
  done. Nothing is left running in the background at the end of a session.
