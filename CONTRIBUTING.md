# Contributing

## Setup

```sh
uv sync --extra mlx --extra laya     # plain `uv sync` drops both extras and breaks the CLI
uv run pytest                        # no network and no model needed
git config core.hooksPath .githooks  # enables the pre-commit hook below
```

## Before you open a PR

CI (`.github/workflows/release.yml`) runs only on a `vX.Y.Z` tag push, by design: macOS is the
only runner with the mlx and laya extras, and downloading model weights plus the full suite is
real wall-clock time to spend on every push. It does not run on pushes or pull requests, so a PR
gets no CI signal from GitHub itself.

The pre-commit hook is the actual gate. It blocks the paths listed below, runs gitleaks on the
staged diff, and runs the test suite against the staged snapshot rather than your working tree, so
unrelated unstaged edits can't make it pass or fail for the wrong reason. Enable it once per clone
(above), and run `uv run pytest` yourself before opening a PR.

## What never gets committed

`.env`, `data/`, `models/`, `runs/`, `apol.db`, `experiments/`. The pre-commit hook blocks these;
see `CLAUDE.md`'s Constraints section for why each one is there (real data, large binaries, or
private-repo scripts).

## Tests

Tests first. `uv run pytest` needs no network and no model: the CLI and wire-shape tests run
against a `uniform` fake backend. `tests/test_examples.py` checks every bank under `examples/`
still validates. `tests/test_public_repo.py` fails on a home path or a private tool name anywhere
the package or plugin ships.

## Docs

When a command or a measured number changes, update `docs/guide.md` and `examples/README.md` in
the same change (`CLAUDE.md`'s Commands section says why).

## Commit messages

Small, atomic commits. A subject line describing why a change was made, not what it touched: the
diff already says what.

## Questions

Open an issue. For a vulnerability, see [SECURITY.md](SECURITY.md) instead of a public issue.
