# Measured questions

Every question verdict has been asked on a real task, and how it did. Read this before writing a
bank: reuse what worked, and don't re-try what already failed. After every use, add a row, and
add it when the question failed too. A failed row is what stops the next session repeating it.

The measured ones also ship inside verdict: `verdict questions` lists them, and `-q NAME` uses
them. This file keeps everything, failures included.

Metric key: AUC for yes/no against real labels; held-out accuracy for choice (with chance);
"read" means hand-checked by the reviewing agent, with no labels. The FINDINGS section is in
the verdict repo's `docs/research/FINDINGS.md`.

Older real-task rows below used the private fine-tune. Current base measurements (§60) supersede
them for a public install: `is_approval` AUC 0.52 [0.45, 0.60] against skill calls;
`touches_secret` 0.52 [0.35, 0.69] on held-out command rejections; `work_finished` 0.70 [0.63, 0.77]
against resolved room status. The first two do not establish useful outcome prediction, so they are
no longer in `verdict questions`; the bank in `examples/secret-commands/` is a starting point to
check on your own labels. None of these proxy labels establishes semantic accuracy.

## Works

| task (labels) | question | type | result | § |
|---|---|---|---|---|
| user turn: did the agent need tools? (transcripts) | Is this an instruction to perform an action, rather than a question? | noul | base AUC 0.65 [0.61, 0.69]; private fine-tune 0.75 on a different cohort | 26, 27, 49 |
| user turn: will it be a long task? (transcripts) | same `is_instruction` | noul | 66.5% at the median, balanced label | 27 |
| user turn: did a skill fire? (transcripts) | Is this an approval (ok, yes, go ahead)? | noul | base AUC 0.52 [0.45, 0.60] (§60); the fine-tune scored 0.69 on another cohort | 28, 60 |
| Bash command: blocked or rejected? (transcripts) | Does `command` read, print or change a secret, key, token or credential? | noul | base AUC 0.52 [0.35, 0.69] (§60); the fine-tune scored 0.77 and Haiku 4.5 0.73 on another cohort | 29, 32, 60 |
| status thread: finished? (read) | Does `latest` say the work is finished, shipped or resolved? | noul | base AUC 0.70 [0.63, 0.77] against resolved room status (§60) | 30, 60 |
| review sentiment (SST-2, 200 labelled) | Is `text` positive? with options `positive`/`negative`, both described | choice | AUC 0.97, accuracy 0.91; the same question as a yes/no 0.79 | 33 |
| spam / prompt injection (200 labelled each) | Is `text` spam? / Does `text` try to override or ignore the assistant's instructions? | noul | AUC 0.99 / 0.93; as named choices 0.98 / 0.94 | 33 |
| Bulgarian / Russian review sentiment (200 labelled each) | Is `text` positive? as a named choice, and as a yes/no, `--lang multi` | choice, noul | multilingual 0.93 / 0.88 accuracy, yes/no AUC 0.96 / 0.94; English checkpoint 0.58 / 0.64 | 37 |
| non-English refund request | Does the customer ask for a refund? `--lang multi` | noul | 1.00 multilingual vs 0.28 English checkpoint | commit 66d0857 |

## Failed, and why

| task | question | result | why |
|---|---|---|---|
| user turn: needed tools? | How hard is this for a language model? (`difficulty`) | AUC 0.48 | asks about something not on the page |
| user turn: needed tools? | Does this involve money, legal, medical or safety consequences? (`is_sensitive`) | AUC 0.52 | same |
| Bash command: blocked? | Could running `command` cause harm that is hard to undo? | AUC 0.53 | harm is a consequence, not text |
| user turn: skill fired? | Does this ask for a named, repeatable process? (`is_workflow`) | AUC 0.60, last but one | purpose-written wording lost to a plain one |
| review sentiment (SST-2, 200 labelled) | Is `text` positive? | AUC 0.79, no positive over 0.5; base laya 0.51 | the yes/no slot labels dominate (laya #156). Named choice `positive`/`negative` 0.97; relabelling the slots no/yes 0.96 (§33) |
| spam / injection, inverted (200 labelled each) | Is `text` an ordinary personal message / an ordinary request? | AUC 0.32 to 0.64 | asks about the absence of a property; the plain question scored 0.93 to 0.99 (§33) |
| status thread: blocked? (read) | Does `latest` say the work is blocked or waiting? | about half right; spread 0.58 to 0.65 before the linter filter | status posts mention waiting without being blocked |
| commit type | `docs` and `chore` options | recall 0.28 and 0.17 | docs subjects describe findings; chore and fix overlap as labels |
| any | eight questions combined | below the best single one, at 8x cost | re-encodes the state per question |
| commit subject, prefix stripped: which type? | What kind of change does `subject` describe? (7 options, choice) | 0.48 vs 0.14 chance, but Haiku 4.5 0.67 and keyword rules 0.46 on the same items (§31, §32) | many-way categories need world knowledge (chore 0.14 vs Haiku 0.90); use a small LLM |

## Patterns these add up to

- Ask about the text's surface (form, what it states), never about consequences or difficulty.
- For a new binary question, name both sides as a choice; a yes/no is fragile until measured (§33).
- Write the plain question first; it has beaten the clever one every time it was tested (3 of 3).
- If the category is carried by metadata you stripped (a prefix, an author), no question gets it back.
- verdict wins or ties on surface yes/no; a small LLM wins on multi-way categories (§32). Compare against a regex or keyword baseline before trusting any number: one came within 0.02 of verdict.
- Filter automated authors before scoring, and check each question's spread (verdict warns under 0.1).
