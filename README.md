# verdict

Fast, local answers to typed questions about text: yes/no, pick one, or a level, each with a
probability, in tens of milliseconds on Apple Silicon. Nothing is generated and nothing is run on
your behalf. verdict answers; you, or your agent, act.

Use it when the same judgment has to be made over hundreds or thousands of short texts (commands,
tickets, messages, agent turns) and a person or an LLM should read only what it flags.

## Quick start

Apple Silicon, Python 3.11.

```sh
uv tool install --python 3.11 'verdict[mlx,laya] @ git+https://github.com/iksnerd/verdict.git@v0.6.1'
verdict init        # picks a free port and writes ~/.config/verdict/config.toml
verdict serve &     # holds the model, so each call costs milliseconds

verdict ask "commit the fix and push it" "Is this an instruction to perform an action?"
# 0.94
verdict ask "why does the build fail on CI" "Is this an instruction to perform an action?"
# 0.05

kill $(lsof -ti TCP:8799 -sTCP:LISTEN)   # holds the model in GPU memory; stop it when done
                                          # (8799 unless `verdict config` shows another url)
```

The model, [Laya](https://huggingface.co/convaiinnovations/laya) (`aac6fef/laya-mlx`,
Apache-2.0), downloads itself on first use. `verdict update` moves to the newest release.

## When it is the right tool

It fits a task with many items, short texts, a question about what the text shows, a yes/no or a
few named options, and a need for a number to rank or cut on, or for data to stay on the machine.
On public labelled sets the default model ranks sentiment at 0.94, spam at 0.99 and prompt
injection at 0.88 AUC ([FINDINGS §40](docs/FINDINGS.md)).

It is the wrong tool for categories that need world knowledge (seven commit types: 0.44 accuracy,
where an LLM does better) or a maintainer's judgment (bug report or support question: 0.59, and two
rounds of fine-tuning did not fix it; [FINDINGS §41, §45](docs/FINDINGS.md)). It is also wrong for
more than 20 options, for long documents (it reads the first 128 tokens), and for questions about
consequences or difficulty, which score at chance. verdict refuses those questions and more than
20 options rather than answer them badly, and warns when a text runs past what it reads.

## Features

- **Three question types:** `ask` inline, `decide` with a JSON bank of questions. A yes/no gives
  P(yes), a choice a probability per option, a score an expected level.
- **Bulk runs:** `decide --jsonl` streams one bank over any number of states and flags a question
  whose answers barely vary.
- **Calibration:** `calibrate` fits a cut or a temperature on your own labels and reports it on a
  held-out split.
- **Measured questions by name:** `verdict questions` lists questions with their measured results;
  `-q is_instruction,touches_secret` uses them.
- **Guard rails:** question shapes that measured at chance are refused (`--allow-unmeasured` asks
  anyway), and `validate` checks a bank with no model loaded.
- **Search by named answers:** `rank` orders a `decide --jsonl` run by a weighted sum of its
  answers and shows what each one contributed.
- **Other languages:** `--lang multi` switches to Laya's multilingual checkpoint.
- **An HTTP API** on localhost ([docs/api.md](docs/api.md)), including TypeSafe Jev's
  `/v1/systemone` protocol, so Jev's SDKs work against it.
- **Backend capabilities:** `GET /v1/capabilities` reports Laya/Jev limits and confidence
  semantics separately from the shared protocol ([SDK guide](docs/api.md#discovering-backend-differences)).
- **Swap the backend:** `[model.extra]` names further local laya checkpoints; `SystemOneBackend`
  swaps to any other `/v1/systemone` server instead (a real, differently-trained model, not another
  laya checkpoint). `verdict examples --systemone URL` checks it against the same real inputs
  `examples/README.md` documents for Laya, with a match count and ms/call, not one-off questions.
- **The docs ship with the tool:** `verdict docs`, `verdict docs guide`, `verdict docs findings N`.

## How it works

Laya is a ModernBERT encoder trained to answer typed questions in one forward pass, run through
MLX. `verdict serve` loads it once; without a server, each command loads it itself. Every question
costs one pass over the state, so a bank should hold only the questions you need. The scores are
uncalibrated: trust their order, and fit a cut before gating anything on one.

## For agents

A Claude Code plugin in this repo teaches an agent when to reach for verdict and how to read the
numbers:

```sh
claude plugin marketplace add iksnerd/verdict
claude plugin install verdict@verdict
```

## Documentation

- [docs/guide.md](docs/guide.md): writing questions, bulk runs, calibration, recipes,
  troubleshooting, and the command reference.
- [examples/](examples/README.md): fourteen runnable banks with invented inputs and their real
  measured output, including what each one gets wrong.
- [docs/api.md](docs/api.md): the HTTP API.
- [docs/FINDINGS.md](docs/FINDINGS.md): every measurement behind these docs, with sample sizes
  and intervals.
- [docs/pipeline.md](docs/pipeline.md): building a checkpoint.

## Developing

```sh
uv sync --extra mlx --extra laya     # plain `uv sync` drops both extras and breaks the CLI
uv run pytest                        # no network and no model needed
git config core.hooksPath .githooks  # pre-commit: blocked paths, gitleaks, tests on the staged snapshot
```

A release is a `vX.Y.Z` tag with a committed answer-quality scorecard (`verdict bench`, 12 pinned
public suites); CI refuses a tag whose scorecard is missing or below the previous one.

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
