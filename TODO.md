# TODO

Second DRY audit, 2026-11-24, at HEAD 7eb2d7c (after the first DRY refactors, `017f3a9`, and the state
machine fixes). Four read-only agents, one per slice. Every item below is a candidate, not a decision:
read both copies before changing one. Line numbers are from that commit and drift.

Rules for all of it: a pure refactor changes no output, exit code, flag, wire format, on-disk format or
message text byte for byte. Run the covering tests first (green), refactor one item at a time, run the
suite once at the end. Commit by area.

Slices reported: backends and judges, core, CLI and hook, pipeline and training.

**Status 2026-11-24:** everything checked `[x]` below is done (full suite 1389 passed). Left open on
purpose: `write_jsonl` atomic (inverts a layer and changes the write), `json_digest`, the closed-backend
`__enter__`/`__exit__` mixin, and the `sys.stdin` fixture (one line already). The sections after
'Rejected' are still open.

## Source, low risk

- [x] `cli/update.py` 195-198, 207-209, 217-218: the reinstall hint
      `_tool_install_cmd(_tool_spec(extras, ...))` is built three times (two with the `X.Y.Z`
      placeholder and the suffix `  (a release tag)`, one with the real version). Add private
      `_reinstall_line(extras, version | None)`; each site keeps its own sentence. Tests:
      `test_update_audit2.py`, `test_cli_audit4.py` (second Ctrl-C), `test_cli_update.py`.
- [x] `support._warn(message)`: `print(f"verdict: {x}", file=sys.stderr)` is written at about 16 sites
      (catalog, decisions, inference, update, setup, evaluation). Look up `sys.stderr` at call time (not
      a default argument) so `capsys` still works. Replaces the two `lambda line: print(...)` in
      `decisions.py`. Convert only the simple one-line messages.
- [x] `inputs.read_jsonl(path)`: `[json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]`
      in `cli/decisions.py:347`, `cli/evaluation.py:210`, `evaluation/router_eval.py:52`, plus about
      six places in tests. Callers keep their own try/except and emptiness checks. Not
      `examples_cmd._rows` (a streaming generator).
- [x] `client.opt_flags(allow_unmeasured, yesno) -> dict[str, bool]`: the same pairs are built in
      `sdk.py:171`, `cli/inference.py:252` (`_flags`) and `client.py:188` (as a list). Keep the
      `/v1/decide?allow_unmeasured=true&yesno=true` URL byte for byte and dict order.
- [x] `cli/parser.py`: `pause_flag(p, help=...)` (calibrate, bench, examples share the help
      "between calls (default: 0.05)"; decide has its own) and `systemone_flags` for the identical
      `--systemone-model default="jev-latest"` in bench and examples. `--help` output must not change
      (`tests/project/test_help_text.py`).
- [x] `cli/inference.py` 147, 164, 187, 190: the "server or in-process" test on `result["source"]` is
      written four times. One `_kind(result)`; compute once in `__call__`, pass to `_note_source`.
      Leave line 180 (`== "in-process"`): it means exactly that, not "not server".
- [x] `judge/cascade.py`: the escalation-failure row is written in three branches (127-129, 131-139,
      143-146). A local `_fail(row, failed, i, message)` keeps `escalation_error` and `failed` in
      step; drop the `continue`/duplicate `rows.append` by using an `else`.
- [x] `judge/cascade.py` 115-116 and 202-203: `-1.0 if is_flat(a) else confidence(a, cut)` in `run` and
      `check`. One private `_urgency(answer, cut=None)`; do not fold it into the public `confidence()`.
- [x] `judge/calibrate.py` 87-88 and 126-127: the finite-and-in-[0,1] check and its message, in
      `fit_cut` and `fit_noul`. A `_check_unit_interval(scores)`. Leave `temper` (different message).
- [x] `backends/mlx.py` 42-44: `_max_len` is an alias of `engine.max_len`, read once (line 73). Read
      `self.engine.max_len` there. Do NOT read the resolved throwaway engine for a named checkpoint.
- [x] `cli/setup.py` and `cli/routing.py`: the flag-over-settings rule for the URL
      (`config.check_url(args.url, "--url") if args.url else settings.url`) and the budget
      (`args.budget if args.budget is not None else settings.prompt_token_budget`) is written twice
      each. One-line helpers next to `inference._resolved_model`. Optional.
- [x] `support._stdin_is_terminal()` for `getattr(sys.stdin, "isatty", lambda: False)()`
      (`decisions.py` 76, 385). Leave `setup.py`'s plain `sys.stdin.isatty()`: deliberate. Optional.

## Tests, low risk

- [x] `tests/fakes.py`: `FailsAt(exc=None, fail_at=2, name="flaky")` replaces `Flaky`
      (`tests/serving/test_api_audit4.py:13-35`) and `FailsOnThird`
      (`test_route_batch_audit3.py:11-33`); their `decide` loops are identical. Optionally a
      `post_batch` helper beside it. About 30 lines.
- [x] `tests/fakes.py`: `retry_run(first, monkeypatch, **client_kw)` replaces `run_post`
      (`test_systemone_retry_audit3.py:10-38`) and `run` (`test_systemone_audit4.py:9-31`); `OK` and `Q`
      are duplicated too. About 25 lines.
- [x] `tests/conftest.py` (or a fixture module): a `config_files(monkeypatch, tmp_path)` fixture that
      returns `use(*texts)` writing each file and setting `CONFIG_PATHS`. Copies: `test_config_audit2.py`
      10-15 and 41-44, `test_config_audit4.py` 11-16 and 56-62, `test_config_errors.py` 13-20,
      `test_client_audit2.py` 11-14, `test_serve_systemone.py` 64-67, `test_cli_more.py` (5).
      `_isolated` in `test_config.py` differs on purpose (relative path plus chdir): leave it.
- [x] Redundant tests. Delete or merge only after confirming no unique coverage is lost (a mutation
      check is advisable): `test_a_forced_path_that_is_not_a_file...` is identical in
      `test_config_audit2.py:35-37` and `test_config_audit4.py:73-75`; the two healthz tests in
      `test_api_audit2.py` 68-79 are weaker than the exact assertion in `test_api.py:16-21` (keep the
      `_closed` test in `test_api_audit4.py`, it adds coverage).
- [x] `tests/fakes.py`: `fake_transformers(monkeypatch, calls, fail=False)` (the audit4 copy in
      `test_backends_audit4.py:28-36` is the superset; `test_truncation_audit3.py:14-20` and the inline
      variant at `test_engine_audit2.py:67-74` duplicate it).
- [x] `tests/fakes.py`: `install_laya(monkeypatch, load)` for the fake `laya_mlx` module installed by
      hand 5-6 times (`test_bits.py:54-62`, `test_mlx_backend.py:88-95` and `107-109`,
      `test_engine_audit2.py:44` and `51`, `conftest.py:47-50`). Do not reuse the name `fake_laya`.
- [x] `tests/fakes.py::FakeEngine` gets an optional `name` (default `fake-local`); then the byte-identical
      `Eng` in `tests/cli/test_cli_audit4.py:26-35` and `test_pinning_audit3.py:17-25` (name `local`) and
      the variant in `tests/evaluation/test_bench.py:177` (name `toy`) go. `BANK` and `STATES` repeat in
      audit4, pinning3 and `test_decisions_audit2.py:13`.
- [x] `tests/cli/conftest.py`: a `local_engine` fixture returning `install(engine=None)` for the
      `inference.load` plus `_ENGINES = {}` pair (4 sites in `test_cli_audit4.py`/`test_pinning_audit3.py`,
      plus `test_bench.py:186-187`); an `init_env(free=...)` fixture for the init setup repeated in
      `test_cli_audit4._init`, `test_setup_audit2._config_target`, `test_cli_more.py` 218-220 and
      163-164, `test_cli_consistency.py` 191-193.
- [x] Hook tests: `_start_hook(repo, e, tmp_path, **popen_kw)` shared by
      `test_hooks_audit2.py::test_snapshot_removed_on_signal` and
      `test_hooks_audit4.py::test_a_signal_to_sh_alone_kills_the_pytest_child`; a `_wf()` and
      `_create_run()` pair for the workflow lookups; move `_git`, `_exe`, `_run`, `env`, `_steps`, `_idx`
      to a `tests/project/conftest.py` and drop audit4's `from test_hooks_audit2 import ... # noqa`.
- [x] `tests/fakes.py`: `project_version()` next to `REPO_ROOT` (the `pyproject.toml` read in
      `test_plugin.py:27`, `test_release_notes.py:29`, `test_examples.py:74`). `Q` in
      `test_backends_audit2.py:12` equals `fakes.NOUL_BANK`; `test_clip_state.py` and
      `test_truncated.py` inline the same bank.

## Pipeline and training slice

Source, low risk:

- [x] `pipeline/stages.py` 179 and 202-203: "key is still to ask" is written twice, negated
      (`key in labels.done and key not in redo` vs `label_key(...) not in labels.done or ... in redo`).
      One local `pending(key)` that reads `redo` live (it is mutated by `discard`).
- [x] `label/prompt_eval.py` 106-117: the failure branch and the success path each run
      `cost_of(...)`, `spent += cost`, `ledger.add(cost)`. Set a `failed` flag, bill once, then
      `continue` on failure. Keep ledger-before-cache and its comment.
- [x] Shared default constants: `["gemini-2.5-flash-lite", "gemini-3.1-flash-lite"]` and
      `"gemini-3.1-pro-preview"` in `config.py:19-20` and `label/__main__.py:16-17`; `"gemma3:4b"` in
      `config.py:12`, `gen/ollama.py:9`, `gen/__main__.py:16`; `"http://localhost:11434"` in
      `gen/ollama.py:9` and `gen/__main__.py:19`. Put them next to `PRICES` (`label/teacher.py`) and in
      `gen/ollama.py`. Defaults and digests must not change.
- [ ] Marginal: `json_digest(obj, n=None)` for `label_digest` (stages.py:49-51), `fingerprint`
      (router_eval.py:61-63) and `prompt_eval._key`. Do NOT touch `PipelineConfig.digest`
      (config.py:42): it omits `ensure_ascii=False`, so unifying would change every `config_digest`.

Source, not low risk (decide first):

- [ ] `gen/generate.py:192-197` `write_jsonl` repeats `jsonl_text` + `write_text_atomic`. Bytes are
      identical but the write becomes atomic and fsynced, and `gen` would import `pipeline.store`
      (inverted layer; `label` already does, no cycle seen).

Tests, low risk:

- [x] `laya_train.py` loader and Modal stub repeated in `test_training_audit2.py` 10-25, 56-61, 70-75,
      `test_training_audit3.py` 7-22 and `test_training_audit4.py` 13-24, 96-101: one
      `load_script(name, rel, stub_modal=False)` in `tests/fakes.py` plus `laya_train` and
      `train_modal` fixtures. The `lt` tests skip without torch, laya and transformers: run the Modal
      ones once on a bare setup.
- [x] `reply`/`Transport` verbatim in `test_stages_audit2.py` 39-55 and `test_stages_audit4.py` 38-52;
      `built`/`_built` in audit2, audit4, `test_export.py`; `T` (`test_prompt_eval.py`) and
      `FakeTransport` (`test_pipeline.py`) are the same class. Move `GoodTransport`,
      `AlternatingTransport`, `build_run(...)` to `tests/fakes.py`; audit4's four inline gen+label pairs
      become one `_labelled(tmp_path)`.
- [x] `score_preamble(...)` is called about 17 times with the same arguments
      (`test_prompt_eval.py`, `test_label_audit2.py`, `test_prompt_eval_audit4.py`): a `run_score(...)`
      helper in `test_prompt_eval.py`.
- [x] Tests build stores by hand instead of `states_store`/`labels_store`
      (`test_pipeline.py` 68, 100, 111, 129, 209; `test_stages_audit2.py` 202, 210; `test_export.py`
      47, 91). `test_stages_audit2.py:137` spells the label key format itself, so it would not notice if
      `label_key` changed. Also `jsonl_text(rows)` for the hand-written JSONL in audit4:120,
      audit2:157, `test_compile.py:9-10`, `test_label_audit2.py:74`.
- [x] About 15 sites write a scorecard by hand (`test_bench.py`, `test_eval_audit2.py`,
      `test_eval_audit4.py`): `write_cards(dir, {"0.5.0": card, ...})`. `card()` has two signatures
      (`test_bench.py:110`, `test_eval_audit2.py:9`): leave them.
- [x] Two fake router engines: `test_eval_audit2.py` 49-69 and `test_router_eval.py` ~128-153: one
      `CountingEngine(on_predict=None, die_after=None)`. Audit2 rebinds `holder["eng"]`: low to medium.

Rejected: the `run_lock` wrapper (5 two-line sites; a decorator would obscure the label stage's
pre-lock price check); `prompt_eval._cache_lock` vs `store.run_lock` (the cleanest subset if flocks are
ever reopened); `require_fresh` vs train_modal's `_require_fresh_export` (standalone by design);
`resume_mismatch`/`resume_identity`; `_run_committing`'s unguarded final commit; bench `scorecards()`
regex vs `_version_key`; `_default_transport` (a test seam); `router_eval._write_cache`; the shared
preamble text (hashed into stored `label_digest` values); `read_json(path, default)` (five one-liners).

## Needs a decision first

- [x] `--systemone` key resolution is written twice (`cli/evaluation.py:75-80`,
      `cli/examples_cmd.py:83-88`) and is stricter than `config.systemone_key` (`LOOPBACK` there also
      has `::1`). Unifying through `config.systemone_key_or_none(url)` newly accepts `[::1]`. Only
      do it if that widening is fine.
- [x] `library.served_checkpoint` (128-146) and `MlxBackend._resolve` (`mlx.py` 88-98) still write the
      same name precedence in full (`english` always served; `multilingual` beats an extra of the same
      name; `jev-latest` served only if no extra claims it). A shared `base.checkpoint_for(name,
      extras, multilingual_id)` is the drift guard. Risk: a bad merge changes which checkpoint decides
      whether a yes/no stays a noul. Best-covered item (`test_library_audit4.py`). Do it last.
- [ ] Closed-backend boilerplate in `backends/jev.py`, `backends/systemone.py`, `JevAsker`: about 12-13
      lines each. Only `__enter__`/`__exit__` (5 lines x 3 classes) extracts cleanly. Still marginal;
      revisit if more lifecycle logic arrives.

## Not DRY, found on the way (behaviour changes: treat as bug fixes)

- [x] `cascade.check` (190-194) does `fit["cut"]` and raises a raw `KeyError` on a malformed fit, and
      `_cuts` (49-54) checks only the type and returns `None` for a noul fit with no `cut`. `run` is
      protected only because `calibrate.apply` runs after `_cuts`. Extract
      `calibrate.validate_fit(qid, fit, expected_type)` from `apply` (196-203) and call it from `_cuts`
      and `check`; pass the noun ("answer" vs "question") because the mismatch wording differs.
- [x] `inputs.fields_named` (121) does `q.get("instructions", "")` and breaks on a structured or `None`
      instruction; `instruction_text` handles both. Reachability not checked.
- [x] `hierarchy.py` 103 vs 117: stage 1 tests "flat" inline (an empty distribution counts as flat),
      stage 2 uses `is_flat` (an empty one does not). Making them identical changes the empty case in one
      stage. Only for consistency.

## Rejected (do not re-propose without a new reason)

Merging the three flock implementations; the two AUC functions; `pipeline/audit._norm` vs
`gen/generate._words`; `label/scorecard._group` vs export's grouping loop; the `price_of` guard loops;
`FakeGen` and `cfg()` in several test files; `laya_train.py` sharing code with the package (standalone
Modal script); `sdk._local` vs `api._backend_errors` (different exception types); `config._url` vs
`_systemone_url`; `explain()` vs `load()`; sharing a base class between `DecideRequest` and
`SystemOneRequest` (changes wire field order); merging `client.capabilities`/`models` (tests pin the
urlopen call shape); a mixin for the `_closed` boilerplate; moving `GH_TOKEN` to job level in
`release.yml` (hands the write token to every step); `_restore_guarded` vs `_update_tool`'s
`except KeyboardInterrupt` (different on purpose); the hook's double `write-tree` (test-pinned);
the `time.sleep(args.pause)` guard; the `cascade --check` vs `calibrate` label loops; the
`NoServer` tuple; `client.decide` fake-server stubs in `tests/cli`.

## Left open on purpose from the state machine audits

- [ ] `route --batch` exits 0 whatever it routed (documented in the guide); `ask --json` exits 0 (the
      guide says so; the `--cut` help text is what is loose).
- [ ] `label_digest` does not cover the prompt template or the temperature: changing its inputs would
      invalidate every existing label row.
- [ ] A failed label call leaves no tombstone, so a poison state is paid for again on every rerun; fixing
      it changes an on-disk format.
- [ ] Hub-id router cache keys do not see an upstream revision change; gen has no cost cap; the
      `thoughtsTokenCount` difference between `label/teacher.py` and `gen/gemini.py` (check against
      current Gemini pricing before assuming it is intended).
- [ ] A training run resumed from a checkpoint made before the `T_max` fix keeps its old, too-short
      schedule.

## Never run, so unverified

- [ ] The release workflow's draft, upload, publish flow and its `gh release view TAG --json isDraft`
      lookup. The workflow was disabled on 2026-11-15 (`gh workflow disable release.yml`); re-enable it
      before the next tag and watch the first run.
- [ ] The Modal changes (`_run_committing`, `_require_fresh_export`).
- [ ] The pre-commit hook outside throwaway repos; `bench --out` refusing a server-answered run
      (check how you produce release scorecards before relying on it).
