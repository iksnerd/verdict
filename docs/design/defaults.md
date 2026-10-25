# Decision defaults

These choices separate measurements from policy. Task-specific evidence can justify an explicit
override without changing every caller's behavior.

| Default | Decision and reason | Explicit alternative |
|---|---|---|
| At most 20 options in a flat Laya choice | Keep the conservative refusal. A tree helps Banking77 but loses on CLINC150 (§50, §58, §59); there is no measured accuracy cliff at 20 and no universal winning structure. | `--allow-unmeasured`, HTTP `?allow_unmeasured=true`, or Python `allow_unmeasured=True`. Compare flat and tree on paired labelled items. The override still emits warnings. |
| Two groups searched by a tree | Keep beam 2. It buys most of the measured Banking77 gain at three passes per item (§50). On CLINC150, a wider beam recovers groups but adds joint-ranking errors (§59). | Set `beam` after measuring group recall and end-to-end accuracy. |
| 128 tokens read | Keep the short-state budget. Increasing it did not improve the short public suites (§51). | Raise the budget for long inputs; put decisive fields first and inspect truncation. |
| Unlabelled yes/no asked as a no/yes choice | Keep the rewrite. It fixes the plain form's sentiment collapse (§38, §40). The small real-turn advantage for plain yes/no does not establish a general replacement (§49). | `--yesno`; a yes/no with explicit `labels` is passed as written. Measure each task. |
| Explicit multilingual checkpoint | Keep language selection explicit. The multilingual checkpoint helps Cyrillic and weakens some English tasks (§35, §37); translated questions did not improve the weak suites (§61). | `--lang multi` for Cyrillic. `auto` warns; it does not silently load another model. |
| Fit decision cuts per task | Keep calibration explicit. The measured cuts vary from 0.004 to 0.84 (§55), so a global 0.5 cut is not justified. | Fit a cut with `verdict calibrate` and assess it on held-out labels. |

The many-option override already exists. A bank's successful run is not evidence that its answers
are accurate, and a fitted exception is not transferred to other banks automatically. Missing
state fields and off-page questions remain guarded through the same preparation path. Jev keeps
its own option limits; these Laya defaults do not impose a 20-option cap on it.

Release benchmarks freeze language-filter eligibility and fingerprint sampled items (§61).
Detector updates can change warnings without silently changing the measurement cohort. Verification
rejects differing sample fingerprints when both scorecards have them; legacy cards lack that check.
