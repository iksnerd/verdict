# Findings

The lab notebook behind verdict: dated, append-only experiments. It is here because every number
the docs and the CLI cite comes from a section in it, and each section says its n. For how to use
verdict, read the [guide](guide.md); for what is true now, read the latest sections, which
override earlier ones. The research scripts most sections name are not published; the
public-dataset results can be rerun with `verdict bench`.

§1 to §9 are the base `laya` checkpoint zero-shot through the `laya-mlx` port (`aac6fef/laya-mlx`, FP16),
plus teacher-label agreement. From §10 on the model is our fine-tune, `verdict-v1`, unless a
section says otherwise. Sections are append-only: a later finding that overturns an earlier one says so in its own section (§26
corrects §25).

## 1. What the model sees

`[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`. There is no system
prompt. The levers are the instructions string, the option descriptions and the state. Options
share a fixed token budget (`head_max_len`, 192 for the base checkpoint, 512-token total).

## 2. Cross-task survey (`experiments/task_survey.py`, n=60 per task, balanced)

| task | in Laya's training mix? | chance | exclusive choice (defs) | exclusive choice (names) | one yes/no per class, argmax | single yes/no @0.5 (AUC) |
|---|---|---|---|---|---|---|
| ag_news, 4-way topic | yes | 0.25 | 0.967 | 0.950 | 0.983 | n/a |
| emotion, 6-way | no | 0.167 | 0.567 | 0.550 | 0.567 | n/a |
| prompt injection, binary | no | 0.50 | 0.650 | 0.617 | 0.717 (AUC 0.82) | 0.717 (AUC 0.84) |
| sst2 sentiment, binary | not stated | 0.50 | 0.917 | 0.883 | 0.750 (AUC 0.81) | 0.533 (AUC 0.78) |
| hub message typing (ours), 6-way | no | 0.167 | 0.400 | 0.350 | 0.333 | n/a |

Reading it:

- **In-mix task is near ceiling; held-out tasks are not.** ag_news 0.95 to 0.98, against 0.55 to 0.57
  on emotion, which matches the model card's 0.573 for emotion.
- **The best mechanism depends on the task.** Prompt injection: yes/no questions carry the signal
  (AUC 0.82 to 0.84) while the exclusive choice predicts "benign" for 51 of 60. Sentiment: the reverse,
  the choice is best.
- **Rank signal and calibration are different things.** sst2 single yes/no has AUC 0.78 but accuracy
  0.533, because at the 0.5 threshold it calls 58 of 60 negative. The model ships over-confident and
  uncalibrated, so a threshold or temperature must be fitted on own data before probabilities are used.
- Interval width: at n=60 a 95% interval is about +-12 points. Gaps under that are noise.

## 3. Our task, message typing on a real multi-agent message log

The log is the author's own: agents post typed messages (action, decision, thought and so on) into
topic rooms. It is private, and nothing from it is published beyond the aggregates here. Labels are the type the author-agent assigned when posting (mostly claude agents), so they are a noisy
gold, not ground truth.

- 48 short snippets, 8 settings (`experiments/sweep_hub_typing.py`): accuracy 0.35 to 0.48, macro-F1
  0.32 to 0.46, chance 0.167. Nothing separates the settings (interval about +-14).
- 360 full messages, baseline setting: accuracy 0.325, 95% interval 0.28 to 0.38.
- Definitions, a context sentence, plain-string state and a larger token budget made no measurable
  difference. The larger budget gave outputs identical to the baseline, so options were not being clipped.
- **Consistent bias:** `action` is predicted for 19 to 26 of 48 to 60 in every setting, when the true
  count is one sixth. Reversing option order moved accuracy by several points (positional bias).
- Confidence is informative: the more confident half was right 46% of the time against 29%.

## 4. Teacher agreement (`python -m verdict.label`, 28 synthetic states, $0.39)

Two cheap teachers, two runs each, plus `gemini-3.1-pro-preview` once as a reference.

| measure | kind (7-way) | yes/no properties |
|---|---|---|
| self-agreement, same model run twice | 0.96 and 1.00 | 0.93 to 1.00 |
| cheap vs cheap | 0.75 | 0.75 to 0.93 |
| cheap vs Pro reference | 0.75 to 0.82 | 0.82 to 0.96 |

- Teachers are consistent with themselves and least consistent with each other on the exclusive `kind`
  question. That is the ceiling: a student cannot be expected to beat the disagreement between teachers.
- The generator does not write the type it is asked for: every teacher, Pro included, matched the
  intended type on only 10 to 12 of 28. Intended type is therefore metadata, never a label.
- Pro is 90% of the spend ($0.35 of $0.39) because its long output is billed. Use it only as a small
  reference. Two cheap teachers at two runs each cost about $0.0012 per state.
- Prices used: `PRICES` in `verdict/label/teacher.py`, from the Google pricing page on 2026-09-19.
  Re-verify before a large spend.

## 5. Not established

- **Speed.** ~~Not established.~~ **Resolved in §11 (2026-09-22).** The port claims 13 ms median on an M3 Max. Every timing here was taken on a machine that was
  overloaded (load average 8 to 27, swap nearly full, a runaway process, Docker). Ranges were 50 ms to
  seconds per call and are not a benchmark. Re-measure on a quiet machine before quoting anything.
- ~~**MLX vs PyTorch parity** on our data.~~ **Resolved in §13 (2026-09-22): 160/160, max diff 0.0011.**
- **Whether the task is learnable.** The teacher-vs-author-label ceiling on real messages has not been
  measured, because sending real messages to Gemini was not approved. Only synthetic states were labelled.
- Fine-tuning has not started; there is no dataset yet beyond the 28-state pilot.

## 6. What to take from this

1. Zero-shot Laya is a base to fine-tune, not a drop-in classifier. It works where its training mix
   covers the task and is mediocre elsewhere.
2. For our domain, pose the problem as several yes/no properties and calibrate them, rather than one
   exclusive 7-way type. The teachers agree with each other more on properties, and Laya's yes/no
   signal held up on the held-out injection task.
3. Measure the teacher ceiling on the real task before generating at scale.

## 7. Modal fine-tune path: timing probe (2026-09-20)

`cloud/train_modal.py` on an A10G, 175 proxy items (Pro-labelled, median 194 tokens), micro-batch 8,
27 timed micro-batches per arm, one run each (not replicated):

| precision | ms per micro-batch | peak memory |
|---|---|---|
| fp16 (upstream default) | 343.6 | 8.4 GB |
| bf16 | 354.2 | 8.4 GB |
| tf32, no autocast | 539.7 | 8.4 GB |

fp16 is fastest, bf16 is within noise of it, tf32 is about 57% slower. Projection, not measured on a
real run: about 43 s per 1,000 items per epoch, so roughly 1.9 h and $2 for 40k items over 4 epochs.
8.4 of 24 GB is used, so a larger micro-batch is untested headroom. Loss values after 30 micro-batches
say nothing about learning.

## 8. APOL optimizer trial on the teacher prompt (2026-09-20)

Target: the opening paragraph of the labeling prompt. Score: agreement of `gemini-2.5-flash-lite`
(temperature 0, one run) with the Pro reference, as a share of question decisions. Data: 86 synthetic
states, split by room topic into train 54, holdout 19, frozen test 13. Proposer `gemini-2.5-flash`,
swarm 2, 8 iterations, patience 3, run in a throwaway git sandbox.

| | train (54) | holdout (19) | frozen test (13) |
|---|---|---|---|
| default preamble | 86.30 | 89.47 | 84.62 |
| best candidate found | 86.67 | not scored for the winner | 86.15 |

- **No real improvement.** +0.37 on train is one question decision out of 270; +1.5 on test is one
  decision out of 65. Candidate scores across the run ranged 84.07 to 86.67, which is the noise band
  (one standard error of a share near 0.86 at 270 decisions is about 2 points). The trial cannot tell
  a better preamble from the default.
- **The winner is sensible but not evidenced.** It added one-line definitions of each type and each
  yes/no property to the paragraph. Kept at `prompts/candidates/`; the default preamble stays in use.
- **The weak question is `kind`** (74 to 85 across splits), the exclusive 7-way type. Wording did not fix it,
  which agrees with the teacher-agreement result that `kind` is where teachers disagree most.
- **The prediction shows "missed" because of the config, not the result.** `success_score` was set to 100
  (a perfect score) for a claim that was "beats the default", so it could never be met. It should have
  been the baseline plus a margin fixed in advance.
- **Cost:** scoring $0.156, proposer $0.019, baselines $0.012, about 22 minutes wall. Total Gemini spend
  for the project so far: about $1.31 (pilot $0.39, dev Pro labels $0.74, optimizer trial $0.19).
- **APOL's engine fixes held on a real run.** Candidates were applied and varied (the old symptom was
  one value fifteen times); the file left on disk is byte-identical to the best candidate although the
  last candidate explored was worse; the per-iteration proposal count was persisted.
- **Split audit result** (borrowed from an earlier corpus pipeline): the frozen test split is 13 states
  and has no critique, draft, note or review states; holdout has no review states. Nothing leaks across
  splits. Splitting by topic over only 25 topics is too coarse at this size.

**Reproduced exactly on 2026-09-22**, two days later and through a rebuilt binary, as the
`teacher-prompt` APOL scorecard: train 86.30, holdout 89.47, frozen test 84.62, the same three
numbers to two decimals. All 86 states were cache hits, so the run cost $0.00 and the spend
ledger did not move. That is the first end-to-end run this scorecard has had; before it the
config was `apol init` scaffolding pointing at an `eval.py` this repo never had (§16).

What this says for the pipeline: the teacher prompt is not the lever at this scale. More states,
a stronger or more varied teacher, or reframing `kind` as yes/no properties are more likely to matter
than another round of prompt search.

## 9. v1 dataset (2026-09-20)

`configs/v1.json`: 363 synthetic states (400 requested; the generator's near-duplicate filter and retry
limit stopped it at 363), written by `gemini-2.5-flash-lite` over 60 room topics and 6 authors, labelled by
two cheap teachers (`gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`) with two runs each, split by topic,
exported in the Laya training schema. No Pro reference on this run.

| | states | export items (states x 5 questions) |
|---|---|---|
| train | 259 | 1,295 |
| holdout | 72 | 360 |
| frozen test | 32 | kept apart, never a training file |

- **Audit passed.** All 10 intended types appear in both holdout and test (split seed 0 was enough), zero
  training states duplicate a held-out one, 60 topics. Some types have a single test state
  (critique 1, synthesis 1), so the test set is still small: 32 states is 160 decisions, about +-7 points.
- **JSON mode held up:** 0 failures in 1,452 labelling calls, so no schema is needed for now.
- **Teacher agreement on this set** (cheap vs cheap, means of two runs each): `kind` 79.9, the four yes/no
  properties 88.2 to 94.2. Self-agreement 94.4 to 98.5. So the exclusive `kind` question has a ceiling near
  80, and the properties near 90 to 94; a student should not be expected to beat those.
- **Cost:** labelling $0.425 (manifest, from token counts), generation $0.048. Total Gemini spend for the
  project so far: about $1.78.
- **Projection revised.** The fine-tune estimate of about $2 assumed 40k items. This set is 1,295 training
  items: about 162 micro-batches per epoch at 0.34 s, roughly a minute per epoch, so a 4-epoch run is
  minutes and cents, plus container start and model download. Not yet run.

## 10. First fine-tune (v1) and how it does (2026-09-20)

Base `convaiinnovations/laya`, RLCD fine-tune on the 1,295 v1 training items, A10G, fp16, 4 epochs, 238 s
(avg loss 0.61, 0.39, 0.37, 0.38: flat after epoch 2). Calibration temperatures fitted on the 360 held-out
items: choice 1.007, yes/no 1.032 (the score temperature stayed at its 1.2 default, no score questions).
Converted with `laya-mlx convert` from the local directory without trouble. Scripts: `scripts/eval_student.py`,
`scripts/eval_real.py`, `scripts/eval_fair.py`.

**On synthetic held-out states** (agreement with the mean of the two cheap teachers; the student was trained
toward that mean, so this flatters it against the teacher-vs-teacher row):

| | frozen test (32) | holdout (72) |
|---|---|---|
| base Laya, zero-shot | 54.4 | 57.5 |
| fine-tuned student | 93.8 | 89.5 |
| teacher A vs teacher B | 90.0 | 89.2 |

Per question the student is at 80.6 (`kind`, holdout) and 87.5 to 96.9 on the yes/no properties.

**On real messages from that log** (360, `kind` only, six types, author-assigned labels, all local):

| | accuracy (95% interval) | macro-F1 | predicts `action` |
|---|---|---|---|
| base Laya | 0.300 (0.26 to 0.34) | 0.263 | 101 of 360 |
| fine-tuned student | 0.381 (0.33 to 0.43) | 0.332 | 211 of 360 |

- **Not usable for typing real ledger entries.** +8 points over base on real data, against +32 to +37 on
  synthetic data, and the intervals nearly touch. Chance is 0.167. The student calls 59% of real messages
  `action`; the true share is one sixth.
- **The gap is a distribution shift, not a training failure.** The training states are short messages written
  by a Flash-Lite model; the real ones are long agent posts with markdown headings and commit hashes, and the
  model only sees roughly the first 320 tokens of them. The real labels are also noisy (author-assigned by agents).
- **Untested:** whether the yes/no properties transfer better than `kind`, and how well the teachers agree with
  the real author labels (needs sending real messages to a hosted API, which was not approved).
- `scripts/eval_fair.py` (student against each teacher's mean separately, the fair comparison to the
  teacher-vs-teacher row) had not finished when this was written.

## 11. MLX latency, measured on a quiet machine (2026-09-22)

Replaces the "not established" row in §5. `scripts/bench_router_latency.py`, fine-tuned
`models/verdict-v1-mlx` (FP16) through `laya-mlx`, MLX 0.32.2 on an M3 Pro, 40 reps after 8 warm-up
calls per cell. Load average was 4.22 falling to 2.97 and swap was 5.8 of 7.2 GB used, so the machine
was *not* idle, but p50 sits within 5% of the per-cell minimum in every cell, so the Metal path was
not contended and these numbers stand. Cold model load is 0.34 s.

| state | state tokens | 1 question | 3 questions | 5 questions |
|---|---|---|---|---|
| short prompt | 9 | 15.3 ms | 33.7 ms | 46.6 ms |
| medium prompt | 64 | 26.1 ms | 56.2 ms | 87.1 ms |
| long prompt + code | 1456 (capped at 512) | 93.4 ms | 249.1 ms | 409.4 ms |

- **The port's 13 ms claim is roughly right, for its best case only.** 15.3 ms is a 9-token state and
  one question. Every realistic router call is bigger than that.
- **Cost is `n_questions x f(state_tokens)`: each question re-encodes the whole state.** Marginal cost
  of one more question is about 15 ms at a 64-token state and about 82 ms at 512. There is no shared
  encode across the questions in a `predict()` call, which is the single largest optimisation left on
  the table upstream.
- **`batch_size` does not help.** default 87.0 / 422.4 ms (medium/long, 5 questions), `batch_size=8`
  86.0 / 423.2, `batch_size=64` 87.8 / 422.8, `pad_to_multiple=16 + batch_size=8` 91.9 / 422.7. All
  within noise of the default. The option batches across calls, not across questions within one.
- **The 512-token cap is real and is a latency floor, not just a quality ceiling.** State budget
  against 1 question: 32 tok 23.2 ms, 64 tok 27.1, 128 tok 36.1, 256 tok 55.5, 512 tok 86.9,
  1456 tok 88.7, 5818 tok 99.6. Flat past 512, so an oversized state costs tokenisation only.

**Budget for a router.** Latency is bought with question count and state budget, and nothing else.
Three properties over a 128-token prompt head is 83 ms; five properties over 512 tokens is 409 ms.
Against a hosted LLM call at 0.5 to 3 s time-to-first-token, the first is clearly worth it and the
last is not. Design target: **at most 3 questions over at most 128 tokens of prompt head, about 83 ms.**

Raw output: `data/bench_router_latency.log` (gitignored).

## 12. Big-vs-small routing, zero-shot (2026-09-22)

> **Superseded by §15.** The bank here was invented rather than taken from upstream, and the
> accuracy below was fitted and scored on the same 24 prompts. Kept because the two failures
> it records are the reason §14 and §15 look the way they do. `examples/route_big_small.py`
> was replaced by `evals/router_prompts.jsonl` plus `scripts/eval_router.py`.

Can the v1 student route a prompt to a big model (gemini) or a small one (ollama)? It was never
fine-tuned for this; it is being asked questions it did not see in training. `src/verdict/router.py`,
24 prompts labelled by one engineer's judgment (12 each way), so the
95% interval is about +-20 points. A direction, not a number.

**First attempt scored exactly chance.** A three-question bank (`trivial`, `needs_code_edit`,
`high_stakes`) with the obvious 0.5-ish thresholds sent all 24 prompts to BIG: 12/24, and the same
majority-class collapse §3 found on the ledger. The properties were not the problem.

| property | AUC (toward small) | observed range | fitted threshold | accuracy at it |
|---|---|---|---|---|
| `trivial`, "could a small model answer this?" | 0.60 | 0.23 to 0.46 | 0.32 | 66.7% |
| `needs_code_edit` | 0.75 | 0.15 to 0.58 | 0.38 | 75.0% |
| `high_stakes` | 0.81 | 0.18 to 0.61 | 0.26 | 83.3% |

- **The signal was there; the thresholds were wrong.** No property ever crossed 0.5 in either
  direction, so any 0.5 cut is unanimous by construction. This is §2's "rank signal and calibration
  are different things" exactly: good AUC, useless accuracy at the default threshold.
- **Asking the model to judge its own kind does not work.** `trivial` is near chance and was dropped.
  It asks about model capability; the other two ask about the text, which is all the encoder sees.
- **Two questions, fitted thresholds:** 83.3% in-sample, **79.2% leave-one-out** against 50% chance,
  1 of 24 in the expensive direction. Dropping `trivial` also cut latency: **26 ms p50** for the
  2-question bank against 33 ms for 3, at a 128-token prompt budget.
- **The one dangerous miss is instructive:** "review this PR for security issues, it touches session
  handling and CSRF" routed small. It reads as a review request, and nothing in the surface text
  says the stakes are high. That is the class of error a threshold cannot fix.
- **`confidence` is not independent evidence for a yes/no question.** Laya reports
  `confidence == max(p, 1 - p)`, so gating on confidence and thresholding the probability are the
  same test twice. The router uses an explicit `is_uniform` check instead, which is the thing a
  threshold genuinely cannot express: "there is no model here".

**What this says.** Zero-shot, on properties, with thresholds fitted on 24 prompts, routing is
already well above chance, where zero-shot ledger typing was 0.30 (§10). Routing is the easier
task, and unlike ledger typing its labels come from outcomes rather than from a teacher's opinion:
run a prompt through both models and record which was sufficient. That is real training data, in
the real distribution, which is precisely what §10 says the synthetic pipeline could not provide.

Next, in order: collect real prompts, label by outcome, refit the thresholds on a held-out split
large enough to have an interval worth quoting, and only then consider a fine-tune.

## 13. Which Laya are we running, and does the MLX port agree? (2026-09-22)

**Both, at different stages, and yes.**

| stage | runtime | artefact |
|---|---|---|
| base checkpoint | n/a | `convaiinnovations/laya` (upstream, 2152 likes on HF) |
| zero-shot probing (§2, §3) | MLX | `aac6fef/laya-mlx`, pre-converted FP16 of that exact revision `c5d7873` |
| fine-tune (§10) | PyTorch 2.14, A10G on Modal | `models/verdict-v1-torch` |
| everything served locally | MLX 0.32.2 via `laya-mlx` 0.1.0 | `models/verdict-v1-mlx` |

The two local checkpoints carry byte-identical `rl_agent_config.json`, so they are the same
fine-tune in two runtimes.

**Parity, which §5 left open.** `verdict.train.parity` over the 32 frozen-test states x 5 questions,
PyTorch original against the MLX port:

| | |
|---|---|
| decisions compared | 160 |
| argmax agreement | **160/160 (1.00)** |
| max probability difference | 0.0011 (tolerance 0.02) |
| mean probability difference | 0.00019 |

So the MLX port is safe to serve, on our data and not only on its own 63 fixtures. Cold load is
0.4 s against 25.7 s for PyTorch, a 64x difference that matters for a process that starts per call.

**A naming correction.** `mizorewww/laya-mlx` is the port's **GitHub** project, which is what
`pip install laya-mlx` builds from. `aac6fef/laya-mlx` is the **Hugging Face** repo with the
pre-converted weights. The `backend_mlx.py` docstring named the first as though it were a HF model
repo; that path 401s. Corrected.

## 14. Upstream already ships a router question bank, and it beats ours (2026-09-22)

`laya.router_questions()` exists in the installed `laya` 0.3.4 and was not noticed before §12 was
written: `difficulty` (a 4-level score), `domain` (6-way choice), `needs_tools` and `is_sensitive`.
There is also a `laya.Router` with a `system_one()` method, though that routes between *Laya
checkpoints* by language and task, not between LLMs. The question bank is the reusable part.

On the same 24 labelled prompts as §12:

| signal | AUC (toward small), base | AUC, v1 student |
|---|---|---|
| **`difficulty`** (official) | **0.90** | **0.92** |
| `is_sensitive` (official) | 0.82 | 0.88 |
| `needs_tools` (official) | 0.41 | 0.34 |
| `high_stakes` (ours, invented) | 0.62 | 0.81 |

- **`difficulty` is the best single signal found so far**, and it is one question, so it is also the
  cheapest: 23 ms against 26 ms for our two. It works nearly as well on the *base* checkpoint
  (0.90) as on the fine-tuned one (0.92), which is the §2 pattern exactly: a question inside Laya's
  training mix runs near ceiling while an invented one does not. Our `trivial` (§12, AUC 0.60) was
  invented; `difficulty` asks the same thing in the phrasing the model was trained on.
- **`needs_tools` is below chance here** (0.41 and 0.34), i.e. mildly anti-correlated. It is asking
  about the request, not about which model should serve it, and our prompt set does not vary on it.
- **Accuracy after threshold-fitting cannot separate these at n=24.** Leave-one-out under one
  consistent procedure: difficulty-only 75.0%, difficulty+`is_sensitive` 75.0%, ours 70.8%. Those
  differences are one prompt wide. AUC is the more sensitive statistic and it is not close.

**What this says.** Prefer the official bank over an invented one, for provenance rather than for
the accuracy delta: it is in the training mix, it is one question instead of two, and it is cheaper.
The §12 policy should be rebuilt on `difficulty` with `is_sensitive` as the safety escalation.
Check what a library already ships before inventing a bank; this one was in `dir(laya)` all along.

## 15. The router, rebuilt on upstream's bank and scored honestly (2026-09-22)

§12's router was fitted and scored on the same 24 prompts, which means its 83.3% said nothing.
This replaces it: `evals/router_prompts.jsonl` is 72 prompts (36 each way) with the train/test
split written into the file rather than re-derived, `scripts/eval_router.py` fits on train only
and reports on a test split the fit never saw, and model outputs are cached so a threshold sweep
is repeatable without re-running anything.

The bank is now `laya.router_questions()`'s `difficulty` and `is_sensitive` (§14), not the
questions this project invented. Over all 72 prompts, **`difficulty` AUC 0.88, `is_sensitive`
AUC 0.81**, confirming §14's 0.90 to 0.92 on a set three times the size.

| | train (48) | **held-out (24)** |
|---|---|---|
| accuracy | 83.3% | **70.8%**, 95% CI 54.2 to 87.5 (chance 50%) |
| `hard->small` | | 3 |
| `easy->big` | | 4 |
| cost per prompt | 0.167 | 0.542 |

**The 12-point train-to-test drop is the honest cost of fitting two thresholds on 48 prompts.**
The held-out interval still clears chance, but only just. More prompts is the fix, not more knobs.

**Fitting on accuracy optimises the wrong thing, and it is not a small effect.** The first refit
maximised accuracy and produced thresholds scoring **75.0%** held-out, four points *better* than
the ones shipped, while taking `hard->small` from 2 to 5. Accuracy treats both mistakes as equal.
They are not: an easy prompt sent to the big model wastes money, a hard one sent to the small
model produces a wrong answer someone has to catch. `scripts/eval_router.py` now minimises
expected cost with `hard->small` weighted 3x, the same asymmetry the base checkpoint carries in
`rl_agent_config.json` (`act_costs.escalate` 0.5 against `cost_wrong_act` 3.0). The shipped
thresholds are the cost-minimising pair: lower accuracy, 0.542 cost per prompt against 0.667.
**A router tuned on accuracy is a worse router that reports a better number.**

**The remaining held-out mistakes are mostly one kind.** Four of six are long analytical prompts
scored just under the difficulty cut ("work out the blast radius of deleting this column",
"reconcile these two conflicting specs"). They read as short instructions; the work is implied
rather than stated. The one in the other direction is "what's the current VAT rate in Germany",
a pure lookup escalated by `is_sensitive` because it mentions money. That one is arguably the
policy behaving as designed rather than an error.

**A trap worth naming.** A uniform `difficulty` answer scores 1.5, which is *below* the 1.6
threshold. A backend with no model behind it would therefore route every prompt to the small
model, the expensive direction, unless something stops it. `router.is_uniform` checks the
distributions for flatness rather than reading `confidence`, because `UniformBackend` reports
confidence 0.0 for a score question and 0.5 for a yes/no one, so no single confidence cutoff
catches both. Covered by `test_the_uniform_backend_can_never_route_small`, which asserts the trap
is real before asserting the guard closes it.

## 16. All three scorecards run, and the sweep that said so was lying (2026-09-22)

`apol bench` on each, recorded in `apol.db`:

| scorecard | score | gate | prediction |
|---|---|---|---|
| `router-accuracy` | 70.83 held-out | 60 | met, +20.8 over chance |
| `teacher-agreement` | 88.57 | 85 | met |
| `teacher-prompt` | 86.30 train, 89.47 holdout, 84.62 frozen test | 80 | met |

`teacher-agreement` reproduced its 2026-09-20 pilot number to two decimals, and `teacher-prompt`
reproduced all three §8 numbers exactly, both through a binary rebuilt in between. Total spend for
the lot: **$0.00**. Every one of `teacher-prompt`'s 86 states was a cache hit, so the ledger at
`runs/dev/optimizer_spend.json` did not move from $0.16748.

**`teacher-prompt` had never produced a score in its life.** The config was `apol init` scaffolding
that was never filled in: a placeholder task description, a baseline with no number, contamination
notes reading "replace with real exposure notes before sharing results", `success_score` 100 (the
same unmeetable-prediction mistake §8 records making), and a `test_command` naming an `eval.py`
this repo does not contain. The real judge script had been committed all along and nothing
pointed at it.

**The sweep reported it conformant.** That is the part worth keeping:

```
apol validate --all --no-run   ->  3/3 conformant, exit 0
apol validate --all            ->  2/3 conformant, exit 1
```

`--no-run` skips the judge dry-run, and `--no-run` is what APOL's own `docs/ci.md` recommends for
CI, so the configuration most likely to run automatically was the one that could not catch a judge
naming a script nobody wrote. Fixed upstream: the validator now
checks `test_command`, `holdout_test_command` and `final_test_command` statically. Commands already
run as whitespace-split argv, so the tokens are literal paths and need no shell to resolve; tokens
carrying shell or glob syntax are skipped rather than guessed at.

**The lesson is the one APOL's own room keeps re-learning:** the mechanism existed and nothing asked
it. Check 5b already refuses a judge that cannot emit `[APOL_SCORE]`, by running it. A check that
has to be remembered is a check that eventually is not.

Two smaller things worth not re-deriving:

- **A Go binary's mtime can lie about what is in it.** The installed `apol` was dated after the
  engine fix and built from a commit two weeks before it. `go version -m $(which apol)` reports the
  real `vcs.revision`; the filesystem cannot.
- **`stability_samples` earns its keep on a hosted judge.** APOL warned that `teacher-prompt` had a
  non-deterministic eval with no repeat sampling, which is exactly the weakness §8 spent 22 minutes
  and $0.19 discovering by hand. It is set to 2 now, free on a warm cache.

## 17. The CLI was 2.1 s for a 26 ms decision (2026-09-22)

A one-shot `verdict route` took **2.1 s** to make a decision the model answers in 26 ms, 80x
overhead, against a premise that the whole thing beats a 0.5 to 3 s hosted call.

| component | cost |
|---|---|
| `import transformers` | **1277 ms** |
| `import laya_mlx` | 158 ms |
| tokenizer load | 142 ms |
| model load | 99 ms |
| the decision | 26 to 75 ms |

**The clip cost 25x what it saved.** `transformers` was imported for one reason: to clip the
prompt to 128 tokens, worth about 50 ms of inference (§11). In a process that exits immediately
that is a catastrophic trade; in a server that loads once it pays for itself on every later call.
The same line is right or wrong depending only on process lifetime, which is not something the
code could say about itself.

**Fixed by asking a server first.** `verdict serve` loads the checkpoint once; `verdict route`
POSTs to `/v1/route` and falls back to in-process when nothing answers.

| path | wall |
|---|---|
| in-process (before) | 2100 ms |
| in-process, no tokenizer | 340 ms |
| **via a warm server** | **70 ms** |
| of which the service itself (curl) | 35 ms |
| of which bare Python startup | 10 ms |

30x on the warm path. The floor is the 35 ms round trip, which is almost entirely model
inference: uvicorn and FastAPI add single-digit milliseconds to it.

Three things this turned up that were not the point:

- **`uvicorn verdict.api:app` serves the `uniform` backend**, because that is what `create_app()`
  defaults to. A server answering `uniform` is *worse* than no server for routing: the client
  treats it as absent and falls back, so you pay a round trip and then load the model anyway.
  Hence `verdict serve`, which loads a real checkpoint eagerly so the first caller is not the one
  who pays for it.
- **The two front doors disagreed.** The CLI clipped and the HTTP endpoint did not, so a long
  prompt could route differently depending on how you asked. `/v1/route` clips now; both paths
  return `difficulty 1.8947` on the same long prompt.
- **`uv run` adds about 10 ms** over the installed console script. Small, but it is a tenth of the
  warm-path budget, so a hot loop should call `.venv/bin/verdict` directly.

The client uses `urllib.request` rather than the already-present `httpx`: 13 ms to import against
48 ms, which is not a rounding error when the decision is 26 ms. It imports nothing heavy, and a
test parses its AST to keep it that way.

## 18. The batch endpoint: 34 ms a prompt, which is the inference floor (2026-09-22)

`POST /v1/route/batch` and `verdict route --batch` (prompts on stdin, NDJSON out). Measured on the
72 prompts of `evals/router_prompts.jsonl`:

| how | wall | per prompt |
|---|---|---|
| one CLI call each, no server | about 151 s (72 x 2.1 s) | 2100 ms |
| one CLI call each, warm server | 5.40 s | 75 ms |
| batch, in-process, no server | 4.62 s | 34 ms |
| **batch, warm server** | **2.50 s** | **34 ms** |

Both sources returned the same 27 small / 45 big.

**34 ms is the floor**, not a batching win: it is one forward pass per prompt. The port has no
cross-state batching (`predict` takes a single state, and the `batch_size=16` attribute is the
micro-batch for questions *within* one state, which §11 already measured as a no-op on a
two-question bank). So this removes per-call overhead and nothing else, and per-call overhead was
41 of the 75 ms.

Reaching into `prepare`/`forward` to concatenate several states into one pass might win more on
GPU utilisation. Not done: it is surgery on a third-party port whose value is that §13 verified it
at 160/160 against PyTorch, and that verification does not cover a path we assembled ourselves.

**Batch helps the no-server case most**, which was not the reason for building it. In-process it
loads the checkpoint once for the whole set instead of once per prompt: 151 s to 4.6 s, 33x. The
server is still better, but a caller with no server running is no longer punished 60-fold.

Two bugs this shook out, both from code that had no test:

- `decide_many` called `branch.is_small`, which existed on the old `router.Route` and not on
  `switch.Branch`. It is a router concept, not a switch one, so the fix was at the call site. The
  batch path had no test, which is exactly why it reached a real run broken.
- The in-process timer started before the lazy checkpoint load, so a batch of three reported 608 ms
  a prompt. Warming the engine first also makes the two sources comparable, since the server has
  already paid that cost.

## 19. An audit pass, and a port collision that would have been silent (2026-09-22)

Six things found by going back over what had been noticed and not fixed.

**The default port collided with another local service.** A speech engine on the development
machine listens on 8765, the port this project also chose. It started partway through a measurement session
and the symptom was two lies at once: `verdict serve` printed `serving ... on http://127.0.0.1:8765`
*before* binding and then died on `EADDRINUSE`, while `verdict route` would happily POST to whatever
else held the port. Default moved to 8799, `serve` now probes the port before loading 400 MB of
weights and exits 2 with the `lsof` line that names the holder.

**The client now checks it is talking to a verdict server.** A 200 with a JSON body was enough to be
believed. It requires `model` plus the shape it asked for (`branch`, or `results`) and treats
anything else as no server. Ports get reused, as the 8765 collision showed.

**`scripts/bench_router_latency.py` carried a private copy of the router bank**: the five invented
`noul` questions dropped in §14, including `trivial` at AUC 0.60 and `needs_tools` below chance. So
the latency benchmark measured a bank the project no longer used, and cited a "§6.2" that does not
exist. It imports `verdict.router.ROUTER_BANK` now, plus explicitly-labelled filler questions,
because the sweep varies the *count* (§11: cost is `n_questions x f(state_tokens)`) and not the
content.

**`switch.Branch` stopped being a dataclass.** `dataclasses` pulls `inspect`, together 3.7 ms of the
CLI's 21.8 ms of imports, to generate three fields' worth of behaviour. Written out by hand with
`__slots__`, `__eq__` and `__repr__`, both modules leave the import graph entirely. With `subprocess`
deferred to `--exec`, the CLI's fixed cost is 21.8 ms to 18.7 ms.

**The floor is `urllib.request` at 10.3 ms**, almost all of it `http.client` pulling the `email`
header machinery. Not fixed, deliberately: the alternative is owning a hand-rolled HTTP client, with
its status codes and chunked encoding, to save 10 ms. `httpx` is worse here at 48 ms to import.

**Three docs were stale.** `docs/pipeline.md` told you to run the pipeline with a `pipeline.json`
that does not exist (the configs are `configs/dev.json` and `configs/v1.json`). `CLAUDE.md` still
recommended `uvicorn verdict.api:app`, which serves the `uniform` backend and is worse than no
server. `skill/verdict/SKILL.md`, which is what a model reads, predated both `verdict serve` and the
batch endpoint, so an agent following it would have looped single calls at 60x the cost.

The pattern is §16's again, one level up: a check that has to be remembered is a check that
eventually is not. Every item here was something a person had noticed and moved past.

## 20. Connection reuse: measured, not worth having (2026-09-22)

The last item on the CLI shape list, and the answer is no. 40 sequential single routes against a
warm server, in one process, new connection each against one reused `http.client.HTTPConnection`:

| | p50 | min |
|---|---|---|
| new connection per call | 33.31 ms | 32.32 ms |
| reused connection | 34.19 ms | 31.66 ms |

**Reuse was 0.88 ms slower**, which is noise in both directions. A loopback TCP handshake costs
nothing against 33 ms of inference, so pooling would add state, lifecycle and failure modes to buy
a negative number. Not built.

Worth recording rather than leaving as an untried idea: "add connection pooling" reads like an
obvious optimisation, and the only thing separating it from the two that *did* pay (asking a server
at all, §17; batching, §18) is that someone measured. The two that paid were 30x and 60x. This one
is -3%.

## 21. Using it as a user, and the bug that only that found (2026-09-22)

Everything up to here measured the decision. Nothing had ever run `verdict route --exec`, which is
the part that acts on it.

**`CLIS[SMALL]` named `ollama run qwen2.5:7b`, and that model was not pulled on the development machine.**
`--exec` would have failed on every `small` route. Both binaries were present (`ollama`, `gemini`),
so a shallower check would have passed: the missing thing was the model, one level down. No test
could have caught it, because the correct value is a property of the machine rather than of the
code.

Three changes, in order of how much they matter:

1. **`verdict init`**, which asks the machine instead of guessing: it lists the models `ollama`
   actually reports, warns when the configured one is absent and suggests an installed one, checks
   each dispatch binary is on `PATH`, finds a free port when the configured one is taken, and
   checks the checkpoint directory exists. Idempotent, so re-running offers the current values back.
2. **`verdict.toml`** (`config.py`), with precedence flag > environment > file > built-in default.
   Gitignored: model paths, ports and which local models you happen to have pulled are properties
   of a machine.
3. **`--exec` preflights the binary** and exits 2 naming the variable to set, rather than raising.

**There is deliberately no `[thresholds]` table.** The cuts are applied by the switch inside the
server process, so a value in the file would be read and ignored on the server path, and a config
key that silently does nothing is worse than no key.

**`verdict decide`** was added at the same time, for the case that has nothing to do with routing:
your own typed questions against any state, one pass, nothing dispatched. Measured at 37 to 66 ms
for a three-question bank. Worth reporting honestly, since it is the same §2 lesson a third time:
asked whether shell commands are destructive, it ranked `rm -rf` above `ls -la` but scored `ls -la`
at 0.48, and put `git push --force` above `rm -rf ~/Documents` on blast radius. The ranking carries
signal and the absolute numbers do not. Anyone using `decide` on invented questions must fit cuts
on their own labelled examples before trusting a threshold, exactly as `skill/verdict/SKILL.md` says.

**ollama is not in the chain.** It is a destination for `--exec` and nothing else. `POST /v1/decide`
and `verdict route --quiet` hand back a decision and let the caller act, which is the path an agent
uses and the one where none of this configuration matters.

## 22. What the format actually lets you say, and one negative result (2026-09-22)

Read laya's `build_sequence` rather than inferring from behaviour. The sequence is

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

There is **no system prompt**, so the only places context can go are the instructions, the option
descriptions, and the state. That makes the option descriptions a context channel, and this project
had been ignoring it.

**Our schema was stricter than the format it claims to mirror.** `CLAUDE.md` says the wire format
mirrors laya's `predict(state, questions)`. It did not:

| input | laya | our schema, before |
|---|---|---|
| `noul` with `criteria: {"true": ..., "false": ...}` | renders it | rejected (`extra: forbid`, no field) |
| a structured criterion (`{"means": ..., "signals": [...]}`) | renders as compact JSON | rejected (strings only) |

Both are accepted now. Without `criteria` a yes/no question shows the model the literal strings
`"false: no, the statement does not hold"` and `"true: yes, the statement holds"`, which carry no
information about the question being asked.

**Describing the two sides does change answers, and on this task it makes the router worse.** A
three-prompt probe looked promising: on `is_sensitive`, describing both sides moved
"what's the current VAT rate in Germany" from 0.574 to 0.384, which is §15's known bad escalation.
Measured over all 72 evaluation prompts it does not hold:

| bank | AUC `difficulty` | AUC `is_sensitive` | held-out accuracy | held-out cost | `hard->small` |
|---|---|---|---|---|---|
| bare (shipped) | 0.875 | **0.813** | **70.8%** | 0.542 | 3 |
| described | 0.875 | 0.633 | 62.5% | **0.458** | **1** |

Described wins on expected cost, which is the objective the thresholds are fitted on, and it halves
the expensive error. It is not adopted, for two reasons. The cost difference is 11 against 13 cost
units on 24 prompts, which is noise. The AUC drop is measured on all 72 and is large, and AUC is the
threshold-free statistic, so it is the one that survives a refit.

**The likely cause is §14 again, one level down.** "Phrase the question the way the model was
trained" applies to the *criteria*, not just the instructions. `"yes, the statement holds"` is
laya's own default wording and therefore in the training distribution; a hand-written description,
however sensible it reads, is not. An invented criterion lands the same way an invented question
does.

So: the lever is real, it is now reachable, and for this bank it is the wrong lever. A refit
confirms the shipped cuts are still the cost-optimal pair on train: `T_DIFFICULTY = 1.6`,
`T_SENSITIVE = 0.3`, held-out 70.8%.

**A silent limit, now surfaced.** `build_sequence` truncates every rendered option to 48 tokens
with no error. A 61-token criterion loses its last 13 tokens mid-sentence. `Engine.option_overflow`
reports which options cross it and what the model never reads, and `verdict decide` prints those
warnings before spending the call. Not enforced: a long criterion still works, and only its author
can say whether the dropped clause mattered.

**Two levers exist in `build_sequence` that no `predict` exposes.** `option_order` is the
positional-bias knob (§3 measured several points from reversing options), and `truncate_left`
decides which end of an over-long state survives. The state is truncated from the **right** by
default, so a long prompt whose actual request comes last loses it, and our own 128-token clip
takes the head as well. Reaching either would mean bypassing `predict`, which §13's parity result
does not cover.

`max_prefixes: 6` in `rl_agent_config.json` is not a context mechanism. The only "prefix" in laya's
code is a weight-key prefix check.

## 23. The specialist checkpoint is not a shortcut for the ledger task (2026-09-22)

The Laya family on Hugging Face is larger than this project had assumed: three official checkpoints
and around thirty community conversions (GGUF, ONNX, CoreML/ANE, LiteRT).

| checkpoint | encoder | context | for |
|---|---|---|---|
| `convaiinnovations/laya` | ModernBERT-large | **512** | English, general. What we build on. |
| `convaiinnovations/laya-multilingual` | mmBERT-base | **1024** | 100+ languages |
| `convaiinnovations/laya-typed-decisions` | ModernBERT-large | **1024** | four synthetic workflows |

**The 512-token ceiling is a property of our checkpoint, not of Laya.** §11 measured 512 as both a
latency floor and a quality ceiling and treated it as fixed. Two of the three official checkpoints
have twice that.

`laya-typed-decisions` looked like a free win. It is fine-tuned on `LocalLLaMA/typed-decisions`,
which is the exact schema `pipeline/export.py` already targets; it scores 0.766 on that benchmark
against base `laya`'s 0.362; one of its four workflows is *agent-trace observability*, which is
essentially §10's ledger task; and `aac6fef` had already published an MLX conversion. Its card
warns plainly: *"This is a specialist. Expect it to behave like the base `laya` checkpoint, or
worse, on anything else."*

**It does.** On the same 360 real log messages as §10, `kind` only, chance 0.167:

| checkpoint | accuracy | 95% CI | macro-F1 | most-predicted label |
|---|---|---|---|---|
| `aac6fef/laya-mlx` (base, zero-shot) | 0.300 | 0.256 to 0.344 | 0.263 | synthesis, 147 of 360 |
| `models/verdict-v1-mlx` (ours) | **0.381** | 0.333 to 0.433 | **0.332** | action, 211 of 360 |
| `aac6fef/laya-typed-decisions-mlx` | 0.311 | 0.264 to 0.356 | 0.263 | synthesis, 158 of 360 |

The specialist lands within noise of the base checkpoint and below our own fine-tune, and it
collapses the same way: 158 of 360 messages called `synthesis`. Being trained on *a* workflow
called agent-trace observability did not transfer to *our* agent traces, which is §10's distribution
shift arriving from the other direction.

**And the 1024-token context does not rescue it either, which kills §10's other explanation.**
§10 blamed the real-message gap partly on truncation. That part is real and now quantified: the
360 real states run to a median of 356 tokens against room for about 317, so **76% of them lose
their tail at `max_len=512`, a median of 53 tokens each**. At 1024 nothing is truncated at all.

| checkpoint | context | truncated | accuracy | macro-F1 | collapses onto |
|---|---|---|---|---|---|
| `laya-mlx` (base) | 512 | 76% | 0.300 | 0.263 | synthesis, 147 |
| **`verdict-v1-mlx` (ours)** | 512 | 76% | **0.381** | **0.332** | action, 211 |
| `laya-typed-decisions-mlx` | 1024 | 0% | 0.311 | 0.263 | synthesis, 158 |
| `laya-multilingual-mlx` | 1024 | 0% | 0.172 | 0.222 | other, 196 |

Both checkpoints that see the whole message score *worse* than ours, which sees 76% of them
truncated. `laya-multilingual` is barely above the 0.167 chance line. It is a different and smaller
encoder (mmBERT-base, 322M against ModernBERT-large's 421M), so it is not a clean isolation of
context, but a general-purpose 1024-token checkpoint landing at chance is not a subtle result.

**So truncation is not why §10 failed.** Fine-tuning on our own data, even the synthetic kind, is
worth more than seeing the whole message: our 512-token student beats every off-the-shelf
checkpoint including the ones with no truncation at all. That removes the cheapest of §10's two
explanations and leaves the expensive one, distribution shift, which is what §10 said to fix and
what no checkpoint swap can fix for us.

There is no shortcut here.

**A latent bug this turned up.** `Engine.tokenizer` joined `f"{model_id}/tokenizer"`, correct for a
local directory and an invalid repo id for the Hub, where the tokenizer is a `subfolder` argument.
So `Engine("aac6fef/laya-mlx")`, this project's own `DEFAULT_MODEL`, raised the moment anything
clipped a prompt. Nothing caught it because every caller passes a local path.

## 24. Core ML and the Neural Engine: not for this workload (2026-09-22)

Three questions, three answers, all no.

**Does Core ML CPU+GPU beat MLX for our model?** No, and the port's own published benchmarks say so
before you measure anything. For Laya 421M, the encoder `verdict-v1` is fine-tuned from:

| backend | 1 question | 3 questions | 10 questions |
|---|---|---|---|
| Core ML CPU+GPU | 13.71 ms | 40.11 ms | 129.80 ms |
| MLX GPU | 13.33 ms | **26.24 ms** | **74.38 ms** |

Those live in `BENCHMARKS.md` in the `mizorewww/laya-coreml` GitHub repo, not on the Hugging Face
page, which is why they are easy to miss.

**Is the Neural Engine variant usable?** No, for two independent reasons. It exists only for
`laya-multilingual`, which §23 measured at 0.172 on the real ledger task against a 0.167 chance
line. And its fast exports are **L96**: a 96-token capacity, against the 128-token prompt budget
§11 settled on, and the card says the short ANE exports reject over-capacity prompts. The headline
numbers are real (4.98 ms P50, 2.78x better energy per decision) and they belong to a model that
cannot do our task at a capacity below our own.

**Does it reproduce here?** Partly, and the part that does not is worth reporting upstream.
Measured on this M3 Pro against `aac6fef/laya-coreml`, one process, 10 warmups then 50 measured
calls a row. The first version of these numbers mixed a p95 from one run with a p50 from another
and published a p95 *below* its own p50, which a blind read caught after it was already filed
upstream. Re-measured in one run, min/p50/p95 per row:

| call | min | p50 | p95 |
|---|---|---|---|
| `score` | 20.45 | 20.75 | 22.07 |
| `noul` | 15.33 | 15.82 | 16.72 |
| `choice` | 10.89 | 11.44 | 14.91 |
| `noul` + `noul` | 30.98 | 31.52 | 32.71 |
| `score` + `score` | 41.01 | 41.68 | 42.77 |
| **`score` + `noul`** | 831.91 | **851.55** | 907.23 |
| `noul` + `score` | 831.72 | **853.35** | 897.66 |
| `choice` + `noul` | 832.96 | **850.10** | 962.16 |
| `score` + `score` + `noul` | 883.44 | **961.48** | 1082.95 |

Same-type pairs cost about twice a single question, which is the expected shape. **Any call mixing
two question types costs about 20 times that**, and three properties narrow it down: order does not
matter, it is not specific to `score` (a `choice` + `noul` pair is equally slow), and it is not a
cold plan cache, because those are 50 consecutive calls with identical questions and the *minimum*
is 831 ms. Every call pays it. The published table shows 40 ms at three questions, which reproduces
here only when all three share a type.

**This project's router bank is exactly `score` + `noul`**, so it lands on the pathological case.
Core ML would take a 34 ms decision to 851 ms. Not adopted, and `laya-coreml` is not added as an
extra.

Reported upstream as [mizorewww/laya-coreml#5](https://github.com/mizorewww/laya-coreml/issues/5).
It is invisible to a benchmark suite that varies question *count* while holding type constant.

## 25. The router is at chance on real traffic (2026-09-22)

§15 reports 70.8% held-out. That number is real and it does not transfer. On a developer's actual
agent traffic the router is a coin flip, and this is the third time §10's lesson has arrived.

**The labels come from outcomes, not judgment, and cost nothing.** Every Claude Code transcript
records what the agent did after each user turn. 563 session files across 61 project directories
gave 2,548 prompt-to-outcome pairs; one prompt carrying a live credential was dropped whole rather
than redacted, leaving 2,312. The label is what actually happened: a turn the agent
answered with text alone could plausibly have gone to a small model, a turn that called tools could
not.

**The base rate alone reframes the product.** Only **8.8%** of real turns were answered with text
alone. The median turn used **8 tool calls over 16 assistant turns** (p90: 41 and 78). So
prompt-level big-versus-small routing can save at most one call in eleven here, and "always big"
scores 91.2% accuracy, which is why this section reports AUC and not accuracy.

**Every subset is at chance:**

| subset | n | small | AUC `difficulty` | AUC `is_sensitive` |
|---|---|---|---|---|
| all real prompts | 2312 | 8.8% | 0.478 | 0.519 |
| self-contained only | 1010 | 12.3% | 0.486 | 0.495 |
| self-contained, 8+ words | 706 | 12.2% | 0.484 | 0.499 |
| longest quartile | 293 | 9.6% | 0.426 | 0.484 |
| **the 72 hand-written prompts (§15)** | **72** | **50%** | **0.88** | **0.81** |

Restricting to self-contained prompts does not rescue it, so this is not a missing routability
gate. `difficulty` is *below* 0.5 on the longest quartile, which is worse than useless.

**Why the 72 prompts flattered it.** The router's author wrote them, and they read like textbook
examples: "what does chmod 755 mean" against "design a migration plan to move 40 services off a
shared postgres". Real turns are short, terse and full of typos, in the manner of "ok do that" or
"now fix the other one". **56% are not self-contained at all**: continuations, deictic references,
or three words or fewer. A curated set built by the same person who built the router measures that
person's imagination.

**Two smaller findings from the same pass.** Asked to judge its own traffic, verdict ranked a
two-word instruction to open and merge a pull request as more question-like (0.60) than a short
yes/no question about a tool (0.24): the ordering is inverted, which is the same out-of-distribution failure from the other side. And
the transcripts contain credentials, one in 2,348 prompts here, so any pipeline touching them needs
a secret scan before anything else.

**What this retires.** Prompt-level big-versus-small routing, for this kind of traffic. Not because
the model is weak, but because the decision is not in the prompt: 92% of turns need tools, and more
than half are fragments that only mean anything given the session.

**What it suggests instead.** The transcript is a free, real, outcome-labelled corpus, and the
useful questions are the ones whose answers it already contains: does this turn need tools, is it
self-contained, is it destructive, which skill should fire. Those have labels with no judge, no
hosted call and no money. The thing this project never had was real labels, and agent transcripts
had them all along.

The corpus is not published, in this repository or anywhere else: it comes from 61 project
directories, including other people's work. Only these aggregates are reported.

## 26. Correction to §25: the question was wrong, not the task (2026-09-22)

§25 concluded that prompt-level routing is retired for this traffic "because the decision is not in
the prompt". That was too strong, and the same 2,312 real turns disprove it. Asked a different
question, the same checkpoint reads them fine:

| question | AUC, predicting the turn needed tools |
|---|---|
| **`is_instruction`** ("is this an instruction to perform an action, rather than a question?") | **0.753** |
| `needs_tools` ("does this require running a command, reading or editing a file?") | 0.672 |
| `is_followup` | 0.516 |
| `is_answerable` | 0.510 |
| `difficulty` (what we shipped) | 0.478 |
| `is_sensitive` (what we shipped) | 0.519 |

On a 400-prompt sample, `is_instruction` scores AUC 0.765, and the 100 highest-scoring turns needed
tools **100%** of the time against a 91.2% base rate. The extremes are sensible: terse orders to
commit, push and update the docs at the top; short "why is this slow"-style questions and open-ended
musings at the bottom.

**The distinction that matters is surface against latent, not present against absent.**
`is_instruction` asks about the grammatical form of the sentence in front of it, which is on the
page. `difficulty` asks the model to simulate how hard a *different* model would find the task,
which is not on the page and never was. Both questions are "about the prompt"; only one is
answerable from it. That is why §2's in-mix results held for sentiment and topic and collapsed
here: sentiment is surface, "how hard is this for a language model" is not.

**What this does and does not rescue.** The signal is real, and the practical ceiling still is not
large: 91.2% of real turns need tools, so the best achievable saving is about one call in eleven,
and the bottom decile by `is_instruction` is 83% tool-needing against the 8.8% base rate, roughly a
2x lift on the minority class. A useful filter, not a transformation.

**§25 stands on everything else**: the shipped bank is at chance on real traffic, the 70.8% does
not transfer, the 72 hand-written prompts measured their author, and the transcript is a free
outcome-labelled corpus. Only the diagnosis changes, and it changes in the direction that matters:
this is fixable by asking a better question, not by a bigger model or a longer context.

## 27. Asking properly: one surface question, on real traffic (2026-09-22)

Eight questions, all about what is visible in the sentence rather than about how hard a task would
be, scored against real outcome labels from the transcripts. 600 prompts, 400 train and 200
held-out.

| question | needed tools (base 91%) | long task (base 49%) |
|---|---|---|
| **`is_instruction`** | **0.759** | **0.694** |
| `is_approval` | 0.673 | 0.681 |
| `is_multi_step` | 0.570 | 0.650 |
| `asks_to_change` | 0.548 | 0.621 |
| `is_destructive` | 0.637 | 0.585 |
| `is_why_question` | 0.389 (0.611 inverted) | 0.405 (0.595 inverted) |
| `names_artifact` | 0.562 | 0.500 |
| `needs_lookup` | 0.466 | 0.516 |
| *for contrast, the shipped bank* | *`difficulty` 0.478* | |

**Held out, with a logistic combiner fitted on train:**

| label | best single | all eight | accuracy at the median cut |
|---|---|---|---|
| needed tools | `is_instruction` **0.773** | 0.761 | n/a, base rate 93.5% |
| long task | 0.634 | **0.661** | **66.5%**, 95% CI 60.0 to 73.0 |

**Seven extra questions do not pay for themselves.** On the tool question the single
`is_instruction` *beats* all eight combined, 0.773 against 0.761. On effort the combination wins by
0.03 AUC and costs eight times the latency, because every question re-encodes the state (§11). The
recommendation is one question.

**`is_approval` is the surprise and it is not a mistake.** "ok", "yes do it", "go ahead" predict
*more* tool use and *longer* tasks, not less. An approval is the turn that unleashes the work the
previous turn proposed. Obvious in hindsight, invisible to anyone reasoning about prompts without
looking at what followed them.

**`is_why_question` is inverted and informative.** Asking why or how something works predicts *not*
needing tools, which is the one reliable signature of a turn a small model could serve.

**What is actually usable.** "Will this be a long task" is a balanced label, so its 66.5% is a real
number rather than an artefact of a 91/9 split, and the interval excludes chance. One question,
about 15 ms. That is an effort estimate available before starting work, which is a decision an
agent genuinely makes.

**What is still not usable.** Big-versus-small routing. 91% of turns need tools, so the ceiling is
one call in eleven no matter how good the question gets, and `is_instruction` at 0.773 does not
change that arithmetic.

The lever was never the model, the checkpoint, the context length or the fine-tune. It was the
question, and finding a better one took an afternoon against real labels that were free all along.

## 28. Skill firing is predictable; the latency scare was not what it first looked like (2026-09-22)

**Two results from the same pass, one of them a retraction.**

**Predicting whether a skill fires.** Same free-label trick as §25: every transcript records which
`Skill` call followed a user turn. 2,151 prompts, a skill fired after 260 of them (12.1%), across
109 distinct skills. On a 700-prompt sample:

| question | AUC |
|---|---|
| `is_approval` | **0.693** |
| `is_instruction` | 0.674 |
| `asks_to_change` | 0.666 |
| `is_multi_step` | 0.665 |
| `is_workflow` | 0.599 |
| `names_artifact` | 0.542 |

**The question written for the job came last but one.** `is_workflow` ("does this ask for a named,
repeatable process such as reviewing, committing, deploying") was designed specifically to predict
skill use, and it is beaten by the generic `is_approval`. That is now the third time a
purpose-written question has lost to a plain one (§14 stock `difficulty` over invented `trivial`,
§22 bare `noul` over described sides, and this). The pattern is consistent enough to plan around:
**write the obvious question first and only reach for a clever one when it measurably loses.**

**The latency retraction.** A batch run reported 831 ms a call for 6 questions where §11 predicts
about 90 ms. The first guess was that varied prompt shapes force MLX to re-plan per sequence length, by
analogy with §24, where mixing question types cost Core ML 20x and was invisible to a suite that
varied question count on fixed inputs. That analogy is wrong. Measured directly, 200 calls of 3
questions each:

| arm | min | p50 | p95 |
|---|---|---|---|
| one prompt repeated 200 times | 53.8 | 68.7 | 158.8 |
| 200 distinct prompts (40 distinct token lengths) | 39.0 | **64.2** | 161.7 |
| 200 distinct prompts padded to a single length | 67.7 | 81.0 | 125.3 |

Varied shapes are the *fastest* arm and padding to a uniform shape is the slowest, so shape variety
is not the mechanism and §11 is not invalidated.

**What the same table does show is worth more.** Three questions cost 31 ms in §11's clean run and
64 ms here, on the same machine and the same workload. A 2x swing with nothing changed but what
else was running. The 831 ms figure came from a run with other jobs on the GPU, which is exactly
what §11 was written to stop, so the lesson is §5's, learned twice: **a latency number without
its machine conditions is not a measurement.** p50 is robust; means and p95 are not, and this
session produced a 741 ms max on an 8-question call that p50 put at 75 ms.

## 29. The CLI on a real bulk task: blocked commands (2026-09-22)

The first end-to-end use of `verdict calibrate` as a tool rather than as an experiment. The labels
are free and real. Of 35,232 distinct Bash commands in one developer's agent transcripts, 81 were blocked
by the Claude Code auto-mode classifier (67) or rejected by the person (14). All 81 went in, with 319
ordinary commands sampled beside them, and each was asked two questions. Fit on 280, reported on
the 120 held out:

| question | held-out AUC | balanced accuracy at the fitted cut |
|---|---|---|
| "Could running \`command\` cause harm that is hard to undo?" | **0.529** | 0.495 |
| "Does \`command\` read, print or change a secret, key, token or credential?" | **0.742** | 0.705 |

**§26 a fourth time.** Harm is a prediction about consequences, and nothing on the page shows it.
A secret is on the page: `.env`, `CLERK_SECRET_KEY`, `~/.aws/credentials`. An earlier partial run
of 411 commands gave 0.607 and 0.733, so the ordering is stable. At the fitted cut (0.35) the
secrets question catches `echo $STRIPE_SECRET_KEY` and `cat ~/.aws/credentials`. It misses
`vercel env pull` and flags `git log`, which is what 0.74 looks like one command at a time.

**Two defects the run found, which no test had.**

- **Language warnings fired on shell.** laya's detector guesses a Latin-script language from word
  frequencies, and 41 of 3,000 commands read to it as Portuguese, French or Spanish. The warning
  now ignores a Latin-script guess on text that is mostly code punctuation (threshold 0.05:
  flagged commands started at 0.054, non-English prose topped out at 0.037). Over all 35,232
  commands, 11 are still flagged (0.03%). Non-Latin scripts always warn.
- **`decide` did not clip, so long states were slow.** 400 calls took 172 s, about 380 ms each,
  against 65 ms for a short `ask`. `route` clips the state to 128 tokens (§11) and `decide` did
  not, so a 600-character command paid the full context for each question. Fixed the same day:
  the server now clips each `decide` state to `prompt_token_budget`. A structured state is
  serialized exactly as laya serializes it and clipped only when it is over budget. Measured on the
  same 400 commands and the same split, one server per arm, back to back:

  | budget | secrets AUC | harm AUC | per call |
  |---|---|---|---|
  | full context | 0.742 | 0.529 | 312 ms |
  | 256 tokens | 0.751 | 0.527 | 323 ms |
  | 128 tokens | 0.771 | 0.563 | 165 ms |

  The full arm reproduced the first run's 0.7415 exactly. At 128 the calls are 1.9x faster with no
  loss. 0.771 against 0.742 is within noise on 24 held-out positives, so the claim is "no worse",
  not "better". 256 buys nothing because few commands exceed it. `verdict serve --budget 0` reads
  the whole state for the rare job that needs it.

The run used one model at a time, 50 ms between calls and the client under `taskpolicy -b`, and the
machine stayed usable throughout; an unthrottled 1,600-call attempt earlier the same day did not.

## 30. Triage of every open room in a multi-agent log, and the input that almost fooled it (2026-09-22)

A second real bulk task, and the first with no labels at all: a private multi-agent message log,
where agents post into topic rooms. 137 open rooms (116 active, 21 paused) were scored with three questions about each room's latest message: does it say the work is
finished, blocked, or name a next step. The state is `{"latest": ..., "topic": ...}` with `latest`
first, because states are clipped to 128 tokens (§29). The DB was read `mode=ro`. 137 rooms took
48 s through a server and 41 s through the in-process fallback with none.

**The first pass scored boilerplate.** The log's automated linter posts as author `system`, and
it had written the latest message in **99 of the 137 rooms** ("This room has been inactive for over
7 days..."). Nothing failed. The run produced plausible rankings, and the linter's posts also
made every room's last activity look recent. Two things gave it away: reading the top of the
ranking, and one question's spread.

| question | range, linter included | range, linter excluded |
|---|---|---|
| `done` | 0.28 to 0.75 | 0.28 to 0.75 |
| `blocked` | **0.58 to 0.65** | 0.12 to 0.63 |
| `next` | 0.16 to 0.56 | 0.14 to 0.70 |

`blocked` over 137 rooms spanned seven hundredths. Uncalibrated scores are only useful for their
ordering, and an ordering that tight is noise. With the linter excluded, the real idle time came out
at a median of 58.5 days, and 72 rooms had been idle over 30 days.

**How good the rankings were, read by hand.** There are no labels, so this is judgment, not AUC:

- `done`, top 11 after the fix: about 9 were finished work. The misses were a room whose latest
  post was a roadmap and one with uncommitted work.
- `blocked`, top 20: about half were real blockers. The rest were status posts that mention
  waiting.
- 22 rooms read as finished and had been idle 35 to 171 days: finished work nobody closed.

The model did what a System 1 is for: it cut 137 rooms to about 30 worth reading. Reading them was
still necessary, which is why the triage procedure is written as a reading list and never changes a
room by itself.

**What changed because of it.** `decide --jsonl` now prints each yes/no question's range over the
batch to stderr and flags a range under 0.1, so the next boilerplate-shaped input announces itself
instead of relying on someone to look. The triage procedure filters `author != 'system'` at the
source.

## 31. A choice question on free labels: what kind of commit is this (2026-09-23)

The first real test of a `choice` question and of temperature fitting. The labels are free:
conventional-commit prefixes across one developer's repos. There are 3,639 prefixed commits over 48
repos. A balanced sample of 418 (60 each of feat, fix, docs, refactor, test, chore; all 58 ci) had
the prefix stripped, and each was asked one plain question: "What kind of change does `subject`
describe?", with a short description per option. Chance is 1 in 7, 14%.

| | |
|---|---|
| held-out accuracy (126) | **0.48**, 3.4x chance |
| fitted temperature | 0.84 (slightly *under*-confident; NLL 1.434 to 1.407) |
| 418 calls through a server | 70 s |

**The single number hid everything worth knowing.** `calibrate` now also reports recall per option
and the top confusions, computed over every example; the argmax ignores the temperature, so the
fit cannot leak into them:

| option | recall | |
|---|---|---|
| feat | 0.70 | |
| test | 0.62 | |
| ci | 0.53 | |
| refactor | 0.50 | refactor to feat x16 |
| docs | **0.28** | |
| fix | 0.27 | fix to refactor x14 |
| chore | **0.17** | **chore to fix x40**, the largest confusion |

**`docs` is §26 from the label side.** Commits here labelled `docs:` describe what was *found* ("the
router is at chance on real traffic"), not that documentation changed. With the prefix stripped,
"this is a docs change" is not on the page, so no question can recover it. **`chore` against
`fix` is the labels overlapping**: 40 of 60 chores read as fixes, and many are ("chore: pin the
action"). Both are limits of the task, not the model. The usable part is the ranking for `feat`,
`test` and `ci`, and knowing that is the reason to report per-option recall at all.

## 32. Head-to-head: verdict against Claude Haiku 4.5 and a regex (2026-09-23)

§29 and §31 each gave verdict a number and no rival. Here both labelled sets are scored three
ways on the same held-out 30% (`calibrate.split`, seed 0). The contestants are verdict, Claude
Haiku 4.5, and a regex or keyword baseline written before scoring. Haiku ran through `claude -p`,
50 items per call, with the data owner's consent to send it.

**Yes/no, what the text shows** (§29 set: does the command touch a secret, scored against "was it
blocked"; 120 held out, 20 positive):

| | AUC | balanced accuracy | 400 items | cost |
|---|---|---|---|---|
| verdict | **0.771** | 0.725 | 62 s | $0 |
| Haiku 4.5 | 0.727 | 0.715 | 403 s | $0.28 |
| regex | 0.640 | 0.640 | instant | $0 |

**Seven-way category** (§31 set: commit type from the subject, prefix stripped; 125 held out,
chance 0.14):

| | accuracy | feat | fix | docs | refactor | test | chore | ci | 418 items | cost |
|---|---|---|---|---|---|---|---|---|---|---|
| verdict | 0.480 | 0.76 | 0.23 | 0.14 | 0.62 | 0.68 | 0.14 | 0.65 | 57 s | $0 |
| Haiku 4.5 | **0.672** | 0.76 | 0.54 | 0.43 | 0.71 | 0.53 | **0.90** | 0.70 | 363 s | $0.21 |
| keyword rules | 0.464 | 0.82 | 0.08 | 0.21 | 0.38 | 0.74 | 0.52 | 0.35 | instant | $0 |

**Each wins a different job.** On the yes/no question verdict matches Haiku: 0.77 against 0.73 is
inside the noise of 20 positives. It does so free, locally and several times faster. On the
seven-way category Haiku wins by 19 points, well outside the noise at n=125. verdict barely clears
hand-written keyword rules (0.480 against 0.464). Its worst option, `chore` at 0.14, is Haiku's
best at 0.90. Telling a chore from a fix takes knowing what the words refer to, which a surface
reader does not have. §31 said docs and chore were unrecoverable once the prefix was stripped;
Haiku's 0.43 and 0.90 show that was verdict's limit, not the task's.

**Caveats that cut toward Haiku.** The timings and costs are for `claude -p` with a 50-item
prompt. Direct API calls in parallel would finish in seconds for less. So "several times faster"
is real but overstated, and at about $0.25 per 400 items cost barely separates them.

**Routing, as measured:** yes/no about the text's surface, in bulk, especially data that should
stay local: verdict. Categories with several options, or anything needing world knowledge: a small
LLM. A regex is a floor to beat, not a contender, at 0.64 and 0.46.

## 33. Yes/no wording is fragile; a named choice is not, and upstream's `labels` fix works (2026-09-24)

Upstream documents a known limit (laya #156): on the English checkpoint a `noul` can follow its
`false:` / `true:` option labels instead of the state. A smoke test reproduced it on our
checkpoint: a plainly positive review scored 0.01 on "Is this review positive?". §2 had already
seen the symptom at n=60 (SST-2, 58 of 60 called negative at 0.5). Here it is measured properly.

**Setup** (`scripts/eval_noul_lean.py`). Three public, human-labelled sets from the local Hugging
Face cache, English only, 100 items per class, seed 0, states clipped to 128 tokens as the CLI
does. Each item was asked the same thing three ways: the plain yes/no, the inverted yes/no (scored
as 1 - p), and a two-option `choice` with semantic keys and descriptions (scored as P(target)).
Both checkpoints, one at a time. AUC with a 1,000-sample bootstrap interval.

| set | checkpoint | plain yes/no | inverted yes/no | named choice |
|---|---|---|---|---|
| SST-2 ("positive?" / "negative?") | verdict-v1 | 0.79 [0.73, 0.85] | 0.96 | **0.97** |
| | base laya | **0.51** [0.49, 0.52] | 0.96 | **0.97** |
| SMS spam ("spam?" / "an ordinary personal message?") | verdict-v1 | **0.99** | 0.51 | 0.98 |
| | base laya | **0.99** | **0.32** | 0.98 |
| injection ("tries to override instructions?" / "an ordinary request?") | verdict-v1 | 0.93 | 0.63 | **0.94** |
| | base laya | 0.88 | 0.64 | **0.91** |

**The named choice is best or tied on all six.** Each yes/no wording fails somewhere. The plain
SST-2 question collapses: base laya answers 0.00 to every item, and on our checkpoint no positive
review reaches 0.5. **The fine-tune recovered the ranking from chance to 0.79** without fixing the
level. That is the lean costing ranking, not just calibration, so "rank, don't threshold" does
not cover it.

**The inverted failures have one cause.** All four ask whether something is "ordinary", which is
the absence of a property; the text never shows an absence. It is §26's rule again ("ask about
what is on the page"), from a new direction. The same word worked as a *choice option*, where the
other side is named. `verdict` now warns on a yes/no that asks whether something is ordinary,
normal, regular, typical, benign or legitimate.

**Would upstream's fix rescue the yes/no?** laya 0.3.20 lets a `noul` show the model other words
for its two slots (`labels`). laya-mlx 0.2.0 does not have it, so `scripts/eval_noul_labels.py`
patches laya-mlx's option renderer in-process to emulate it, on the same samples, our checkpoint:

| set | `false`/`true` (today) | `B`/`A` (upstream's example) | `no`/`yes` |
|---|---|---|---|
| SST-2 | 0.79, 0% of positives over 0.5 | 0.95, 25% | **0.96**, 50% |
| spam | 0.99 | 0.99 | 0.99 |
| injection | 0.93 | 0.91 | 0.92 |

**Yes.** Relabelled slots take the collapsed question from 0.79 to 0.96, on par with the named
choice, and leave the two healthy ones inside their noise. `no`/`yes` is at least as good as
upstream's `A`/`B`. The patch is identity for `false`/`true`: that row reproduces the table above
to three decimals.

**Is the choice sensitive to option order?** laya-mlx #5 reports 27 to 33% of two-option choices
flipping when the options are swapped, on the multilingual checkpoint with Chinese world-knowledge
questions. Asked both ways here (`scripts/eval_choice_order.py`), our checkpoint flips 1% of SST-2
items, 0.5% of spam and 5.5% of injection. AUC is 0.97 / 0.97, 0.98 / 0.98 and 0.94 / 0.90 with
the target listed first / second, all inside each other's interval. Averaging both orders changes
nothing worth its double cost. On these English surface questions, order is not a concern.

**What changes:**
- A named two-option choice is the safe default for a binary question when the wording is new.
  A yes/no is fine where it has been measured (every entry in `verdict questions`).
- Relabelling the slots is worth porting to laya-mlx, since the fix belongs there, not in a
  verdict monkeypatch.
- Changing the default relabels every yes/no answer, so it would invalidate the real-label numbers
  in §26 to §30 until they are re-measured. Not done here.
- `verdict bench` tracks `sst2-yesno` next to `sst2-choice`, so a fix, from a port or a retrain,
  shows up on the scorecard.

## 34. Batching several states per forward pass: not worth it on an M3 Pro (2026-09-24)

§18 left cross-state batching undone. Upstream laya reports 9 to 10x from `predict_batch` on an
RTX 5060 Ti, and laya-mlx has no equivalent. `scripts/bench_batch_states.py` hands laya-mlx's own
`predict` the question rows of many states at once, so chunking, collation and post-processing are
all laya-mlx code. 256 SST-2 states, 128-token clip, one run on a quiet machine (load 1.05):

| bank | one `predict` per state | batched, best | batched, worst |
|---|---|---|---|
| 1 question | 18.3 ms | 12.1 ms (64 rows, sorted by length), 1.5x | 18.4 ms, 1.0x |
| 2 questions | 27.7 ms | 24.4 ms (16 rows, sorted), 1.1x | 52.1 ms (32 rows, unsorted), **0.5x** |

Answers agree: every choice is the same, and probabilities differ by at most 0.003 (FP16 on other
batch shapes). The gain is not there. Neighbouring configurations disagree by 40%, and two
questions batched unsorted are twice as slow, likely from padding when yes/no and choice rows of
different lengths share a batch.

**Why: the forward pass is the whole cost.** Per state: clip 0.07 ms, laya's `prepare` 0.09 ms,
forward 17.7 ms, full `predict` 18.2 ms. With one state the Apple GPU is already close to fully
used, so more rows per pass buy little; a discrete Nvidia GPU at batch 1 is mostly idle, which is
where upstream's 9x comes from. There is no CPU-side overhead left to remove either.

Not shipped: 1.5x at best, noisy, sometimes slower, and it needs a patch around laya-mlx internals
plus a lock. Throughput on an M3 Pro is set by the checkpoint and the state budget (§11), not by
how calls are grouped. Core ML is slower (§24), and laya-mlx's own research measured its
kernel-level options at 1.03 to 1.08x, so the one lever left is a smaller model.

## 35. Four checkpoints on the bench: the fine-tune wins on yes/no only (2026-09-24)

`verdict bench` on every laya checkpoint the development machine had, same 1,640 items, 128-token clip, one
model at a time (`--model aac6fef/<name>`). AUC for yes/no, accuracy for choice, 95% intervals in
the scorecards:

| suite | verdict-v1 (ours) | base laya | laya-typed-decisions | laya-multilingual |
|---|---|---|---|---|
| sst2, yes/no | **0.80** | 0.50 | 0.49 | **0.88** |
| sst2, choice | 0.91 | 0.91 | 0.92 | 0.85 |
| spam, yes/no | 0.99 | 0.98 | 0.98 | 0.97 |
| spam, choice | 0.97 | 0.96 | 0.94 | 0.89 |
| injection, yes/no | **0.95** | 0.91 | **0.95** | **0.96** |
| injection, choice | **0.83** | 0.81 | 0.79 | 0.65 |
| agnews, 4 topics | 0.95 | 0.94 | 0.95 | 0.95 |
| emotion, 6 labels | 0.47 | 0.48 | 0.49 | 0.43 |

**The fine-tune's whole advantage is the yes/no.** It is the only English checkpoint that does not
collapse on the SST-2 yes/no (base and typed-decisions sit at chance there, §33), and it leads base
on injection. On every choice, topic and emotion suite the three English checkpoints overlap.
**laya-multilingual** has the best SST-2 yes/no (0.88) and the worst choices (injection 0.65,
AUC 0.72), so it is not a general replacement either. Relabelling the yes/no slots (§33, laya-mlx
#17) lifts ours to 0.96 and base to 0.91 on SST-2, more than any checkpoint swap.

**Emotion is the wording, not the checkpoint.** All four score 0.43 to 0.49 with recall for `fear`
at 0.20 to 0.28 everywhere, well under the 0.60 laya's README reports for its router. That points at
our question wording and sample rather than at a better checkpoint. It is level with Jev's published
0.48, a third-party figure on a different sample.

**What changed:** the fine-tune stays the default. Base laya becomes the fallback when the fine-tuned
weights are missing (`config.resolve_model`): it downloads itself, ties on choices, and the warning
says the yes/no answers are weaker. Before, a machine without the weights could not answer at all.

## 36. 8-bit weights keep the answers; they do not make it faster (2026-09-24)

`scripts/eval_quantized.py` loads the fine-tune once, records its answers on all 1,640 bench
items, then runs `mlx.nn.quantize` on the same module in place (group 64; `act_head` stays FP16,
since `DecisionModel` casts to that layer's weight dtype and its first layer is 1028 wide) and asks
the same items again. 124 layers are quantized. 4-bit is a fresh FP16 load, the same way, so only
one model is ever resident. FP16 reproduces the v0.7.1 scorecard to four decimals on every suite.
One run, mlx 0.32.2, 128-token clip:

| suite | v0.7.1 interval | FP16 | 8-bit | same answer, 8-bit | 4-bit | same answer, 4-bit |
|---|---|---|---|---|---|---|
| sst2, yes/no (AUC) | 0.731 to 0.859 | 0.797 | 0.797 | max diff 0.005, 3 cross the cut | 0.754 | max diff 0.108, 47 cross |
| sst2, choice | 0.865 to 0.945 | 0.905 | 0.900 | 199 / 200 | 0.915 | 194 / 200 |
| spam, yes/no (AUC) | 0.961 to 1.000 | 0.985 | 0.985 | max diff 0.010, 1 crosses | 0.984 | max diff 0.215, 3 cross |
| spam, choice | 0.935 to 0.990 | 0.965 | 0.965 | 200 / 200 | 0.965 | 198 / 200 |
| injection, yes/no (AUC) | 0.924 to 0.977 | 0.952 | 0.952 | max diff 0.019, 1 crosses | 0.945 | max diff 0.329, 10 cross |
| injection, choice | 0.775 to 0.880 | 0.830 | 0.825 | 199 / 200 | 0.785 | 183 / 200 |
| agnews, 4 topics | 0.915 to 0.975 | 0.945 | 0.945 | 200 / 200 | 0.945 | 196 / 200 |
| emotion, 6 labels | 0.404 to 0.529 | 0.467 | 0.471 | 239 / 240 | 0.483 | 221 / 240 |

"Cross the cut" counts yes/no items that land on the other side of the v0.7.1 fitted cut. The
largest 8-bit probability change on any choice option is 0.058 (injection); for 4-bit it is 0.469
(emotion).

| | FP16 | 8-bit | 4-bit |
|---|---|---|---|
| saved parameters | 843 MB | 448 MB | 238 MB |
| MLX active memory, loaded | 804 MB | 429 MB | 228 MB |
| MLX peak while scoring | 1,068 MB | 627 MB | 426 MB |
| p50 / p90 per call, 256 states, 1 yes/no | 21.7 / 29.5 ms | 22.4 / 28.7 ms | 22.8 / 29.7 ms |

Latency was timed at normal priority with load average 2.0 to 3.4, not the 1.05 of §34, so the
three rows agree with each other and with §34's 18 ms only roughly. Quantizing in process briefly
peaks at about 1.2 GB, FP16 and quantized weights both live.

**What it means.** 8-bit passes the shipping rule: every suite inside the v0.7.1 interval, and
row by row it is nearly the same model: 3 of 1,040 choices change, and 5 of 600 yes/no answers
cross their cut. It halves disk and
memory and buys no speed on an M3 Pro: the forward pass at one state is not bound by weight
bandwidth. 4-bit also stays inside every interval, but only just on SST-2 yes/no (0.754 against a
floor of 0.731), and 17 of 200 injection choices change, so the intervals are hiding real
movement there.

**Loading.** laya-mlx's `load` is strict on FP16 names and shapes, so a saved quantized file does
not load through it. The workaround works: build the Agent from the FP16 checkpoint, quantize the
module with the same predicate, then `model.load_weights(quantized_file, strict=True)`. With the
module zeroed before that load, the loaded model gives the in-session 8-bit answers on all 200
`sst2-choice` items, identical to the four decimals laya reports. It still reads the FP16 file first, so it saves memory, not disk or
download.

**What changes:** nothing ships yet. 8-bit is a memory saving with no loss we can measure, worth
taking if laya-mlx gains a quantized load path; the prototype above is the shape of it. 4-bit is not
a candidate on this evidence.

## 37. Cyrillic: the English checkpoint is near chance, the multilingual one is not (2026-09-24)

A smoke test read a glowing Bulgarian review as "negative 0.72" on the English checkpoint and
"positive 1.00" on the multilingual one. Measured properly (`scripts/eval_cyrillic.py`): the
bench's four new Cyrillic suites, Bulgarian store reviews (`mteb/BulgarianStoreReviewSentiment…`,
CC BY 4.0) and Russian product reviews (`ai-forever/ru-reviews-classification`, Apache-2.0), 100
positive and 100 negative each, neutral and mixed skipped, each asked as a named choice and as a
yes/no. Both checkpoints, one resident at a time, 128-token clip, one run:

| suite | verdict-v1 (English) | laya-multilingual |
|---|---|---|
| Bulgarian, named choice (accuracy, AUC) | 0.58, 0.69 | **0.93, 0.97** |
| Bulgarian, yes/no (AUC) | 0.61 (no positive over 0.5) | **0.96** (69% over 0.5) |
| Russian, named choice (accuracy, AUC) | 0.64, 0.77 | **0.88, 0.96** |
| Russian, yes/no (AUC) | 0.60 (5% over 0.5) | **0.94** (59% over 0.5) |

**On Cyrillic the English checkpoint is barely above chance**, and it stays confident while wrong,
which is why `--lang auto` warns. The multilingual checkpoint reads it about as well as the
English one reads English sentiment (SST-2 choice 0.91, §35).

**The multilingual checkpoint has no "no" lean here.** Its plain yes/no is as good as its named
choice, and most positive reviews clear 0.5, unlike the English checkpoint on English (§33). §33's
named-choice advice is still the safe default, but for Cyrillic the yes/no is not the weak form.

**What changed:** `[model].lang = "multi"` (or `$VERDICT_LANG`) makes the multilingual checkpoint
the default, instead of `--lang multi` on every call. Suites gained `lang` and `ignore`, so the
bench asks the Cyrillic suites with the multilingual checkpoint and the release scorecard tracks
them. The confidences are not calibrated: laya ships this checkpoint without fitted temperatures,
so rank or fit a cut, as with English.

## 38. Any yes/no can be asked as a no/yes choice, with no sides to write (2026-09-25)

§33 found a named two-option choice best or tied on every set, and a yes/no whose slots were
relabelled `no`/`yes` as good on the question that collapsed. Neither is something a tool can do
to an arbitrary question: it cannot name the sides of "Does `latest` say the work is blocked?",
and laya-mlx has no slot labels. What it can always do is offer `no` and `yes` as the two options
of a `choice`. That is not the relabel patch, since the model also reads the question type, so
it was measured (`scripts/eval_noul_as_choice.py`): §33's samples, 100 per class, our checkpoint,
states clipped to 128 tokens, both forms asked of every item. The score is P(yes).

| set | plain yes/no | as a no/yes choice |
|---|---|---|
| SST-2, "Is `text` positive?" | 0.791 [0.727, 0.852], 0% of positives over 0.5 | **0.961** [0.935, 0.985], 64% |
| SMS spam, "Is `text` spam?" | 0.992 | 0.993 |
| injection, "Does `text` try to override ... instructions?" | 0.926 [0.888, 0.956] | 0.911 [0.866, 0.949] |

The plain column reproduces §33 to three decimals. The choice form repairs the collapsed question
as well as the relabel did (0.96) and leaves the two healthy ones inside their noise.

**What changed.** `ask`, `decide` and `calibrate` send a yes/no that is not a measured library
question as a no/yes choice and hand the answer back as a yes/no (`noul` = P(yes), with
`"asked_as": "choice"`), so callers reading `answers.<qid>.noul` keep working. The library's yes/no
entries stay as measured, `verdict bench` scores its suites exactly as pinned, and `--yesno` opts
out. On the room-triage example the rewritten `blocked` and `next` spread the right rows further
from the rest (0.72 to 0.85 and 0.87).

In the same change, the shapes §25, §29 and §33 measured at chance (a consequence, difficulty or
risk; a yes/no about an absence; a question naming a field the state lacks) went from warnings to
refusals in `ask`, `decide` and `validate`, with `--allow-unmeasured` to ask anyway. `calibrate`
still only warns: it is how a question gets measured.

## 39. Named answers as a search vector: rescale, and never cosine (2026-09-25)

An idea from a TypeSafe user: ask a few questions about each document and use the answers as the
vector, so every dimension has a name. Tried on 16 invented support tickets
(`examples/ticket-search/`), five yes/no dimensions, three queries whose relevant tickets were fixed
before scoring, precision at k with k the number of relevant tickets:

| query | raw cosine | raw weighted sum | weighted sum, each dimension rescaled to its percentile |
|---|---|---|---|
| money trouble, may leave (5 relevant) | 0.80 | 0.80 | 0.80 |
| broken with a deadline (2) | 0.50 | 0.50 | **1.00** |
| how-to questions (4) | 1.00 | 1.00 | 1.00 |

16 items is a sanity check, not a measurement. Two failures are worth designing around anyway,
because each has a mechanism:

- **Cosine matches direction, not strength.** "Just wanted to say the new editor is great" scored
  about 0.13 on every dimension and still ranked in the top five for money trouble: small numbers in
  the query's proportions point the query's way.
- **Uncalibrated dimensions have their own ranges.** `has_deadline` spread 0.12 to 0.65 and
  `says_broken` 0.10 to 0.92, so the raw sum was decided by `says_broken`. Percentiles over the
  collection put them on one footing.

The dimensions themselves leak and miss in the usual ways: "we launch tomorrow" scored 0.27 on
`has_deadline`; `says_broken` read 0.72 for a wrong invoice and 0.76 for a price rise. `verdict
rank` implements the rescaled weighted sum over a `decide --jsonl` file and prints each dimension's
contribution, which is what showed why the SSO ticket ranked third for money trouble (it says the
customer will leave, not that it is about money).

## 40. With the no/yes rewrite, base Laya is within noise of the fine-tune (2026-09-26)

§35 found the fine-tune's whole advantage over base Laya was the plain yes/no, where base sits at
chance on SST-2. §38 made verdict ask a new yes/no as a no/yes choice instead. The same script on
base Laya (`scripts/eval_noul_as_choice.py aac6fef/laya-mlx`), same samples, 100 per class:

| set | fine-tune, plain | fine-tune, no/yes | base, plain | base, no/yes |
|---|---|---|---|---|
| SST-2 | 0.791 | 0.961 [0.935, 0.985] | 0.505 | **0.940** [0.904, 0.971] |
| spam | 0.992 | 0.993 | 0.992 | 0.995 |
| injection | 0.926 | 0.911 [0.866, 0.949] | 0.875 | 0.875 [0.818, 0.918] |

On everything a user writes, the fine-tune's lead is now 0.02 to 0.04 AUC with overlapping
intervals, down from 0.44 on SST-2. It is still ahead on injection by a margin this sample cannot
separate from noise. What still depends on it: the library's real-traffic numbers (§26 to §30) and
every cut quoted in the docs, all measured on the fine-tune.

**What changed.** The library's yes/no questions keep their plain shape only when the answering
checkpoint is the fine-tune (`verdict-v1-mlx`); on base Laya, or any other checkpoint, they are
rewritten as no/yes choices like any other yes/no. The README tells a public install how to work
on base Laya.

## 41. Tasks Laya never trained on: the fine-tune adds nothing, and both share the same holes (2026-09-26)

Three public, human-labelled sets from families outside Laya's training mix and the datasets
its own benchmarks use (`scripts/eval_gap_tasks.py`; a survey of about 25 candidates picked them):
GitHub issue type (NLBSE'24 test split, 5 repos, label prefixes such as "Bug:" stripped from
titles), conventional commit type (`rsh-raj/commit-classification-17k`, the author's prefix
removed), and DailyDialog dialogue acts (`eusip/silicone` `dyda_da`). Balanced samples, seed 0,
states clipped to 128 tokens, both checkpoints, offline.

| real set | chance | fine-tune | base Laya |
|---|---|---|---|
| issue type (bug, feature, question), 300 | 0.33 | 0.56 [0.51, 0.62] | 0.59 [0.53, 0.65] |
| commit type, 7 types, 420 | 0.14 | 0.43 [0.39, 0.48] | 0.44 [0.40, 0.49] |
| dialogue act (inform, question, directive, commissive), 400 | 0.25 | 0.44 [0.39, 0.49] | 0.42 [0.37, 0.47] |
| "is this an instruction?" (directive vs rest), AUC | 0.50 | 0.79 | 0.77 |

**The v1 fine-tune adds nothing off its training domain;** every interval overlaps. **Both
checkpoints fail in the same places:** a GitHub question is recognised 18 to 19% of the time
(nearly everything is called a bug), commissive utterances (promises, offers, refusals) 7 to 8%,
and `chore` commits 15 to 17%. Commit type reproduces §32's 0.48 on an independent set. The
library's `is_instruction` wording ranks real directives at 0.77 to 0.79 AUC, a second
confirmation after §26 that it transfers.

**Where Laya wins, taking §25 to §41 together.** It matches an LLM on a yes/no about what a
short text shows (§32: 0.77 against Haiku's 0.73) and beats one on cost, speed and privacy. It
loses on categories that need world knowledge (§32), past about 20 options (Laya's README:
Banking77, 77 options, 0.43 against Jev's 0.87, since options share one token budget), on
consequences and difficulty (§25, §29), and on anything past the first 128 tokens.

**What changed.** `ask`, `decide` and `validate` refuse a choice with more than 20 options, with
`--allow-unmeasured` as the override, and warn once per run when a state is far past the
128-token read. The guide's "Which tool for which job" names all five conditions under which
verdict is the right tool.

**Next, not yet done:** a free check of whether rewording the issue and dialogue questions fixes
the shared holes (§26 moved AUC more than any checkpoint did), and only then a targeted synthetic
pilot, about $2 of Gemini, judged against base Laya on these three sets.

## 42. Rewording fixes the dialogue hole, helps issues a little, and does nothing for commits (2026-09-26)

Before any new training data, the free check §26 argues for: the §41 samples asked in other
words (`scripts/eval_gap_wording.py`). Variants: options that describe what the failing class
looks like ("rich"), one no/yes choice per class with the highest P(yes) winning ("one vs rest"),
and for issues the title alone. Accuracy, with the failing class's recall:

| task, checkpoint | as §41 asked | best variant | failing class, before -> after |
|---|---|---|---|
| dialogue act, fine-tune | 0.44 [0.39, 0.49] | 0.54 [0.49, 0.59] one vs rest | commissive 0.08 -> **0.71** (rich) |
| dialogue act, base | 0.42 [0.37, 0.47] | 0.51 [0.45, 0.55] rich | commissive 0.07 -> 0.38 |
| issue type, fine-tune | 0.56 | 0.60 one vs rest | question 0.19 -> 0.24 |
| issue type, base | 0.59 | 0.62 [0.57, 0.68] one vs rest, title only | question 0.18 -> 0.27 |
| commit type, both | 0.43 / 0.44 | none better; one vs rest worse (0.34 to 0.36) | chore about 0.2 throughout |

- **Dialogue acts are a wording problem.** Options that say what a promise or an order looks like
  ("sure, I will, no thanks"; "please..., let's...") separate from §41's wording on both
  checkpoints. No data needed.
- **Issue type moves a little**, inside overlapping intervals. Questions are still mostly read as
  bugs: a real hole, and the candidate for a targeted synthetic pilot.
- **Commit type does not move.** Telling a chore from a fix takes knowing what the words refer
  to (§32), which rewording cannot supply.

## 43. A small decoder, zero-shot, is not a shortcut to Jev (2026-09-26)

Jev appears to be a post-trained language model (its tokenizer looks like Qwen3.5's, from a
reading of its token counts, not yet published). A decoder can answer typed questions without generating: read the text,
question and lettered options once, and take each letter's next-token probability.
`scripts/eval_decoder_zeroshot.py` does that with `mlx-community/Qwen3.5-2B-MLX-4bit`, chat
template with thinking off, on the §41 samples, on an M3 Pro at background priority:

| task | Qwen3.5-2B, zero-shot | base Laya, best wording (§42) |
|---|---|---|
| issue type | 0.61 [0.55, 0.66]; question recall 0.78, bug 0.49 | 0.62 |
| commit type | **0.26**; 63% called `feat`, `refactor` never | 0.44 |
| dialogue act | **0.28**, near chance; 95% called `inform` | 0.51 |
| "is this an instruction?", AUC | 0.57 | 0.77 |
| one forward pass | p50 146 ms, p95 1.6 s | about 15 ms (§11) |

It reads GitHub questions far better than Laya does, and pays for it on bugs, so issue type only
ties. Everywhere else it is worse and about ten times slower. It shows no sign of the world
knowledge that commit type needs. This is a raw 2B model asked zero-shot, so it says little about
Jev, whose quality comes from its training. It does say that an off-the-shelf small decoder is no
shortcut: reaching Jev would take the same training data, on a model ten times slower here. Laya
stays the base.


## 44. A targeted synthetic pilot for issue type: a recorded negative (2026-09-26)

§42 left GitHub questions read as bugs as the one hole worth data. The pilot, with its rule fixed
before any training: fine-tune base Laya (pinned revision `1c5edc17`) on synthetic issues, and
ship nothing unless it (1) beats base Laya on the §41 real issues with non-overlapping 95%
intervals, (2) lifts question recall above base Laya's best wording, 0.27, and (3) holds the bench.

The data is the `github_issue_typing` domain. Gemini 2.5 Flash-Lite wrote 862 issues on about 50
projects, none of them the five in the test set, with no label or prefix in the title. Two
Gemini teachers labelled them with §41's exact question, and the teachers agreed on 95% of the 801 both labelled. Train
554, holdout 196. Four epochs on one A10G took about 2 minutes. Spend was about $0.70 in all:
$0.41 generating, $0.18 labelling and a few cents of GPU.

| real set, same run | base Laya | pilot |
|---|---|---|
| issue type, 300 | 0.587 [0.533, 0.647] | 0.623 [0.573, 0.680] |
| question recall | 0.18 | 0.28 |
| bug / feature recall | 0.90 / 0.68 | 0.95 / 0.64 |
| commit type, 420 | 0.443 [0.398, 0.488] | 0.457 [0.410, 0.505] |
| dialogue act, 400 | 0.417 [0.367, 0.468] | 0.492 [0.440, 0.542] |

**It fails the rule.** The intervals overlap. Question recall clears 0.27 by one item in a
hundred, the same gain §42 got from rewording for free. The pilot still calls 190 of 300 issues
bugs. The bench was not run, since condition (1) already fails. The one clear move was
unintended: dialogue questions, 0.45 to 0.80 recall. Training on "how do I..." issues taught the
model what a question looks like in conversation, not on a GitHub page.

**Reading it.** About 550 synthetic issues move the choice no more than a better question does.
A synthetic question reads like a question. A real one often quotes a stack trace and asks why,
which a generator told to avoid calling it a bug does not reproduce. More data of the same kind is
not the next step. The candidates are the real shape (a question that carries an error), or
§43's decoder, which already reaches 0.78 question recall zero-shot. Nothing ships: the checkpoint
stays local, and the default model is unchanged.

A bug found on the way: the split audit read only `message` and `text`, so every `issue` state
read as empty and "duplicated" a held-out one, and export wrote no training rows. The audit now
reads `issue` too (`tests/test_export.py`).

## 45. Round two, shaped data: better, still inside the noise, and the teachers are the limit (2026-09-26)

§44's reading was that synthetic questions were too clean. Round two tested that directly, with
the rule restated in advance and one addition: if question recall did not clear 0.35 as well,
synthetic data would be recorded as the wrong lever for this task. The new domain,
`github_issue_shapes`, asks for questions that paste an error, a trace or code and ask why, and
for bug reports that also ask whether the behaviour is expected. Feature requests are unchanged.
Base Laya was trained on round one's 554 items plus 338 new ones. Spend was about $0.60: $0.44
generating (518 of 900 states kept, since traces ran past the 1,200-character cap) and $0.12
labelling.

| real issues, 300 | base Laya | round one | round two |
|---|---|---|---|
| accuracy | 0.587 [0.533, 0.647] | 0.623 [0.573, 0.680] | 0.650 [0.600, 0.700] |
| question recall | 0.18 | 0.28 | 0.31 |
| issues called bugs | 192 | 190 | 177 |

Commit type (0.455) and dialogue acts (0.487) were unchanged from round one.

**It fails the rule again.** The intervals still overlap, and question recall of 0.31 misses the
0.35 kill line. Two rounds and about $1.30 took question recall from 0.18 to 0.31. Rewording the
question alone reaches 0.27 (§42).

**Why it stalls: the teachers.** Asked §41's question, the two Gemini teachers read the target
shape as a bug. Of the questions that paste an error, teacher A labelled 130 of 177 as bugs and
teacher B 108 of 201, with a mean P(question) of 0.44. The averaged training targets therefore
leaned toward bug for exactly the items meant to teach "question". The NLBSE labels are what
maintainers decided. "Is this a defect or a support request" is partly a judgment about the
project rather than about the text. These teachers make it differently, so synthetic data they
label cannot teach the maintainers' version. §43's decoder, at 0.78 question recall zero-shot,
shows some of it is on the page for a model that reads it as a language model does.

**Decision.** No round three. Issue type goes with commit type as a task to give a small LLM.
Neither checkpoint ships, and the default model is unchanged.

A bug was fixed on the way. The label transport let an `httpx` timeout escape, which ended
two runs early while still exiting 0. It now raises `LabelError`, the stage counts one failure
and carries on, and a resume fills the gap.

## 46. It reads a sign perfectly and cannot compare numbers, so it cannot play the Iowa Gambling Task (2026-09-28)

The Iowa Gambling Task (Bechara et al., 1994) is a test of learning under uncertainty. There are
four decks: two pay well and lose more, two pay less and come out ahead, and the agent learns
which are which from 100 draws. A private harness runs it, with seeded loss timing and deck
labels shuffled per seed. Scored as net = good picks minus bad picks over 100 trials (random is
about 0), on seeds 0 to 4. Each verdict agent saw a compact state per trial, built by the
harness: each deck's draws, mean result and worst result, plus the last five draws. It asked one
choice question with the options shuffled, and sampled from the probabilities. Base Laya,
0.6.1, served over HTTP.

| agent | mean net | sd |
|---|---|---|
| greedy (tracks each deck's mean payoff) | +75.2 | 3.6 |
| gemma3:4b (full history, as text) | +62.4 | 66.5 |
| qwen2.5:3b (full history, as text) | +25.2 | 60.9 |
| random | -11.2 | 4.4 |
| verdict, "Which deck should be drawn next to win as much money as possible?" | -6.0 | 34.5 |
| verdict, "Which deck in `decks` has the highest `mean_result`?" | -7.2 | 32.7 |
| Laya through the PyTorch port, the first question | -13.2 | 27.6 |

Both verdict agents play at chance. The harness's own `verdict` agent, with its own seeded sampler,
reran them at -3.2 and -2.0 (sd 29.8 and 21.4): the same conclusion, and a sign of how much of
each number is sampling noise. Unlike §25 to §28, rewording to what the page shows changed
nothing. Probes on invented states found out why:

| probe (base Laya, argmax, no game) | right |
|---|---|
| which of four decks has the highest `mean_result`, clear-cut (one deck 150 to 250, the rest -300 to 20) | 6 / 24 (chance 6) |
| the same, options without descriptions | 7 / 24 |
| the largest of four bare numbers, `{"A": 212, "B": -40, ...}` | 12 / 24 |
| one deck's stats: "Is `mean_result` negative?" | 40 / 40, AUC 1.00 |

**It reads a fact on the page, a minus sign, perfectly, and compares numbers across fields not at
all.** Even stripped to four bare numbers, picking the largest is only twice chance. The task is
a comparison across decks, repeated 100 times, which is exactly what it cannot do. This is the
same boundary as §32 and §41 from another side: not world knowledge this time, arithmetic.

**Decision.** Do not ask verdict to rank or pick by numbers. Let code compare, and ask verdict
only what a number's text shows, when a regex can't do it better. An agent that asks "is this
deck's mean negative?" per deck would play near greedy's level, but the script would do all the
learning; `mean < 0` is the same check for free. The harness keeps a `verdict` agent so the
table can be rerun.
