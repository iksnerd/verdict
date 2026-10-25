# Pipeline

← [Back to the README](../README.md)

Goal: a small, fast, calibrated decision model for one domain at a time, trained on synthetic
data that we generate and label ourselves. Today's domain is the engineering-room ledger
(what kind of entry is this, does it record a decision, and so on). The pipeline is
domain-agnostic; the domain is a question bank plus a state generator.

## Stages

```
gen ──> label ──> split ──> export ──> train ──> convert ──> eval
 local   Gemini    by topic   Laya     PyTorch   laya-mlx    scorecards
 LLM     teachers             format   (cloud)   + parity    (APOL)
```

| stage | does | tool | status |
|---|---|---|---|
| gen | writes synthetic states from seeded prompts (topic, author, length, ambiguity) | Ollama `gemma3:4b` | built |
| label | 2 cheap teachers x 2 runs, plus a 3.1-Pro reference on a subset; soft distributions | Gemini API | built |
| split | by room topic, hash-assigned, so a topic never straddles train and test | local | built |
| export | our labels to the layout Laya's fine-tune notebook reads | local | built |
| train | RLCD fine-tune of Laya | upstream PyTorch project, Modal A10G | built (v1, FINDINGS §10) |
| convert | `laya-mlx convert`, then a torch-vs-MLX agreement check on the test split | laya-mlx | built; parity 160/160 (§13) |
| eval | student accuracy, macro-F1, ECE against the teacher ceiling | `verdict bench`, `verdict calibrate` | built except ECE |

Run: `uv run python -m verdict.pipeline --config training/configs/v1.json --run-dir runs/v1 --stage all`
(`training/configs/dev.json` is the smaller, cheaper one).

`uv run python -m verdict.pipeline.compile runs/v1 runs/issues_pilot ... --out data/compiled/synthetic.jsonl`
gathers runs into one file: a row per state per run, with its split and every teacher's raw
labels, nothing averaged, so any training target can be rebuilt without relabelling.

A converted checkpoint goes to the private Hugging Face repo `iksnerd/verdict-v1-mlx`; a machine
uses it by pointing `[model].path` at a local copy. verdict itself defaults to base Laya.
## Rules every stage follows

- **Resumable.** Rerun a stage and only the missing work happens. Rows are appended and fsynced
  one at a time; a torn last line from a kill is dropped on reopen; whole-file writes go through
  a temp file and rename. A four-hour labeling run must survive `Ctrl-C`.
- **Cost-capped.** The label stage sums spend from the rows already on disk, so a resume cannot
  forget what an earlier run spent, and it stops before crossing `max_cost_usd`.
- **Manifested.** Every stage ends by writing `<stage>.manifest.json`: config digest, git commit,
  sha256 of each input and output, counts, spend. APOL scorecards pin these hashes, so a scorecard
  fails when its data drifts instead of quietly scoring something else.
- **Intended type is metadata, never a label.** A 4B generator does not write the type it was
  asked for (10 to 12 of 28 for every teacher, Pro included), so the teacher labels the text that
  was actually produced.
- **Teachers only from Gemini; the local model never labels.** A 3 to 4B model's distributions are
  too noisy to train on.
- **Real data is a held-out sanity set, never training data, and never sent to a hosted API
  without asking.** Real messages can hold personal data and secrets, so they stay out of the
  repository and out of any prompt to a hosted model.

## Open decisions

- A second cheap teacher.
- Whether to keep `kind` as one exclusive question or lean on the yes/no properties, which the
  teachers agree on more (0.86 to 0.96 against 0.75 for `kind`, n=28).
