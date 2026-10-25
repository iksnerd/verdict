# Where Laya is rough

Every model has places where it fails, and the useful thing is to know them before you build on it.
This is the list for the Laya checkpoints verdict runs, in the spirit of the "model jaggedness"
page TypeSafe keeps for each Jev version. Each entry says what fails, the evidence, and what
verdict does about it. Numbers are base Laya (`aac6fef/laya-mlx`), what an install runs, unless
stated. Read `verdict docs findings N` for the measurement behind a section.

## A yes/no can follow its labels instead of the text

A plain yes/no ("Is `text` positive?") scored AUC **0.50** on SST-2: Laya's `noul` renders its sides
as `false:` and `true:`, and the label pair can dominate (laya #156, FINDINGS §33). The same words
asked as a choice between `no` and `yes` score 0.91 to 0.94. **verdict asks every new yes/no that
way** and hands back P(yes); `--yesno` sends the plain form. Laya's own advice, opaque `A`/`B` keys,
measured no better (§47).

## Past about 20 options an exclusive choice degrades

Options share one token budget. Banking77, 77 options: 0.38 here, 0.43 in Laya's README, against
Jev's 0.78 on the same items (§41, §50, §53). **verdict refuses a choice with more than 20
options.** `verdict classify` walks a two-level tree instead: 0.38 to 0.55 with `--beam 2`. For
topical labels an embedding model does better than either (below).

## Laya is a judge, not a retriever

Matching a text to one of many labels by topic is retrieval. A dedicated embedding model
(`embeddinggemma:300m`) picked the right Banking77 intent 0.89 of the time with no Laya at all, and
the right one was in its top 3 in 98% of cases. Laya choosing among those candidates scored 0.65,
and a blend of the two put all its weight on retrieval (§53). Laya's own encoder cannot do this job:
its embeddings found the gold label in the top 20 only 42% of the time, and trees or shortlists
built from them were worse than none. Use Laya for typed judgments about a text (is it spam, is it
an instruction, which of a few named options); use an embedding model to find the nearest of many.

## It reads the first 128 tokens

The default clip is 128 tokens, which is faster with no loss on short texts (§29); the English
checkpoint's context is 512 and the multilingual one's 1,024. **verdict warns once per run** when a
state is far past what it reads. Put the part that decides first. Raising the read budget changed
nothing on the bench suites, where texts are short (§51); it matters for long documents.

## The English checkpoint is confidently wrong off English

Bulgarian reviews: 0.58 accuracy on the English checkpoint, 0.93 on the multilingual one; Russian
0.64 against 0.88 (§37). It stays confident while wrong. **`--lang auto` warns** on non-English
text; `--lang multi` (or `[model].lang`) switches to the multilingual checkpoint, which is weaker on
English and, per Laya's README, rarely picks the first level of a `score` question (#131).

## A question about what is not on the page sits at chance

"How hard is this for a language model?" scored AUC 0.478 on 2,312 real turns, "could this command
cause harm that is hard to undo?" 0.529 (§25, §29). Consequences, difficulty and risk are not
visible in the text. **verdict refuses those wordings** (`--allow-unmeasured` asks anyway). What is
on the page works: "does `command` touch a secret?" ranked blocked commands at 0.74 (§29, measured
on the private fine-tune).

## A yes/no about an absence is near chance

"Is this an ordinary message?" asks for the absence of a property, which the text never shows (§33).
**Refused for a yes/no**; ask for the property itself, or make it a choice with both sides named.

## It cannot compare or rank numbers

Picking the deck with the highest mean was 6 of 24, chance; the largest of four bare numbers 12 of
24; but "is this mean negative?" was 40 of 40 (§46). **Let code compare.** Ask a question about one
thing, and combine answers in your own logic.

## Categories that need world knowledge are weak

Issue type 0.59, commit type 0.44, dialogue act 0.42 on base Laya; a GitHub question is recognised
18 to 19 percent of the time and a promise 7 to 8 percent (§41). Emotion is 0.475 on the bench (§48).
A small LLM beat it on a seven-way category (0.67 against 0.48, §32). Rewording helped some tasks
and not others (§42).

## Describing the two sides of a yes/no hurts

Adding `criteria` for `true` and `false` dropped `is_sensitive` from AUC 0.813 to 0.633 (§22). Leave
them out unless you have measured otherwise. Naming the sides with `labels` is the same lever and is
not measured either; a labelled yes/no is not rewritten as a no/yes choice, so it also loses that fix.

## Probabilities rank; they are not calibrated

The checkpoints ship over-confident, and the multilingual one with no fitted temperatures (§2).
Trust the order, not the number. **`verdict calibrate`** fits a cut from your own labelled examples
and reports it on a held-out split. It matters: Laya's yes/no scores sit away from 0.5 (fitted cuts
of 0.004 to 0.84 on five bench suites), and deciding at the fitted cut instead of 0.5 took accuracy
from 0.81 to 0.86 on held-out items, on every suite (§55).

## More questions cost more passes, and more is not better

Each question in a bank is a forward pass (Jev fans out in one call). Eight combined scored below
the best single one (§27), and batching states helps by at most 1.5 times (§34). Ask the one or two
questions you need.

## Where Jev is clearly better, send only the unsure

Prompt injection, Bulgarian and Russian yes/no, sentiment as a plain yes/no: Jev leads (§48). You
need not send everything: `verdict cascade` asks Laya first and re-asks Jev only about the answers
Laya is least sure of, which closed most of the gap at 20 to 50% sent (§52, §54). It is per task;
on AG News Laya is the better model and sending makes it worse.

## Ordinal `score` is the weakest primitive

SST-5 scored 0.372 in Laya's own README. Prefer a choice with named levels, or a few yes/no
questions combined in code (`verdict rank`).

## The numbers on real traffic depend on the model

`is_instruction` is AUC 0.65 on base Laya and 0.75 on the private fine-tune, measured on the same
kind of real turns (§49). Three library questions are still quoted from the fine-tune and say so in
`verdict questions`.
