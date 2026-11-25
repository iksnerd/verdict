# Models

verdict answers with one of these, chosen by `[model]` in `verdict.toml`, `--lang` and
`--systemone`. The numbers are the 12 pinned public suites of `verdict bench` (200 to 240 items
each, 95% intervals in the scorecards) unless stated; `bench/scorecards/v0.10.1.json` is base Laya
and `bench/head-to-head/` holds Jev's run (FINDINGS §48).

| model | what it is | use it for | limits |
|---|---|---|---|
| **base Laya** `aac6fef/laya-mlx` (the default) | ModernBERT-large, 421M, English, 512 context, run through MLX | English text, short texts, a yes/no or a few named options, free and local | 20 options, 128-token read, plain yes/no collapses (see `rough-edges`) |
| **Laya multilingual** `aac6fef/laya-multilingual-mlx` | mmBERT-base, 322M, 100+ languages, 1,024 context (up to 8,192 upstream) | Cyrillic and other non-English text, `--lang multi` | weaker on English; rarely picks the first `score` level |
| **Laya typed-decisions** (upstream) | Laya fine-tuned on Laya's own typed-decisions workflows | not recommended here | no better than base on our real task (§23, §35) |
| **verdict-v1 fine-tune** (private) | base Laya trained on Gemini labels | an archive; opt in with `--model` | within noise of base on public suites (§40, §41); labels not redistributable |
| **Jev** (hosted) | TypeSafe's System One model, 255 options | Bulgarian and Russian, yes/no sentiment and injection, flat banks past 20 options | every state is sent to the service and billed; about 0.5 s a call |

## Base Laya against Jev on the bench

| suite | base Laya | Jev |
|---|---|---|
| SST-2, yes/no / choice | 0.500 / 0.910 | **0.989** / 0.960 |
| spam, yes/no / choice | 0.982 / 0.960 | 0.986 / 0.965 |
| injection, yes/no / choice | 0.900 / 0.795 | **0.992** / 0.865 |
| AG News, 4 topics | **0.940** | 0.825 |
| emotion, 6 labels | 0.475 | 0.500 |
| Bulgarian, choice / yes/no | 0.925 / 0.963 | **0.985** / **1.000** |
| Russian, choice / yes/no | 0.880 / 0.944 | 0.895 / 0.979 |

Bold marks a gap whose intervals do not overlap. AUC for yes/no, accuracy for the rest. AG News is
in Laya's training mix, so Laya's figure there is probably flattering; Jev's training data is not
published. About 0.03 s a call local, about 0.5 s hosted.

## Long states

Laya reads 512 tokens (the multilingual checkpoint 1024). `[model].max_len` raises that, up to
8192, at about the cost of what it reads (0.09 s a call at 512, 0.41 s at 2048, 3.2 s at 8192). It
is opt-in: the checkpoint was trained at 512, so accuracy beyond that is unmeasured and calibrated
cuts do not carry over. `prompt_token_budget` clips first, so raise it too. See the guide.

## Choosing

- Text stays on the machine, or you have thousands of items: **base Laya**.
- Not English: **Laya multilingual**, or Jev.
- A yes/no where Laya is weak (sentiment, injection), or more than 20 options in one flat choice:
  **Jev**, or `verdict classify` with a tree on Laya.
- Many topical labels (intents, topics) to pick among: not Laya, and not Jev either. An embedding
  model's nearest label scored 0.89 on Banking77 against Jev's 0.78 and Laya's 0.38 to 0.55 (§53).
- One fixed question and a hundred or more labels per class for it: compare with a trained
  classifier before settling on any of these. A logistic regression on `embeddinggemma:300m`
  vectors matched zero-shot Laya on sentiment and spam at 400 labels per class, and beat it on the
  six emotions (0.59 against 0.475) and on prompt injection (0.88 against 0.795, with 20 per class).
  Laya stayed ahead on AG News (0.94 against 0.90), and LightGBM was no better than the logistic
  regression (§63). It answers that one question only; Laya answers a new one with no labels.
- Laya for most items and Jev for the ones Laya is unsure of: half the items sent closes 92% of the
  gap, with the rest of the text staying local (§52). `verdict cascade` does it, and `--check` says
  whether it helps for your task.
- Never gate on a `uniform` backend answer; it means "no model", not "unsure".

Pick the backend per run: `verdict serve --systemone https://api.typesafe.ai` (key in
`$TYPESAFE_API_KEY`), or `[systemone]` in `verdict.toml`. `verdict questions` lists each library
question with the model it was measured on.
