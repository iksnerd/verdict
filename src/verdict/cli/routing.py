"""The big-or-small switch: `verdict route`, `--batch` and `verdict cases`, kept as the worked
example of routing rather than as a router to trust (FINDINGS §25).

`decide()`/`decide_many()` are unrelated to the `verdict decide` bank command despite the shared
name: library callers that import `decide`/`decide_many` directly get these, re-exported from
`verdict.cli`'s top level."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import inference, support
from .. import client
from ..engine import PROMPT_TOKEN_BUDGET
from ..router import BIG_SMALL, ROUTER_BANK, SMALL
from ..switch import Branch

#: Fallback for library callers that import `decide`/`decide_many` directly. The CLI resolves
#: `verdict.toml` first and passes the result in, so this is only what you get without one.
DEFAULT_MODEL = os.environ.get("VERDICT_MODEL", "models/verdict-v1-mlx")


def _say_local(exc: Exception) -> None:
    print(f"verdict: {exc}; loading the model in this process (a few seconds and about 800 MB "
          "of GPU memory; `verdict serve` keeps one loaded)", file=sys.stderr)


def decide(
    prompt: str,
    model: str = DEFAULT_MODEL,
    budget: int = PROMPT_TOKEN_BUDGET,
    *,
    url: str | None = None,
    use_server: bool = True,
):
    """Route one prompt. Returns (Branch, latency_ms, source).

    Asks a running server first. In-process costs about 2.1 s for a 26 ms decision, nearly all
    of it importing transformers and loading the checkpoint; a warm server has already paid that.
    Falls back to loading the model here, and says so on stderr: it works with no server
    running, just slowly. A server that answers too slowly is an error, not a fallback.
    """
    if use_server:
        try:
            t = time.perf_counter()
            payload = client.route(prompt, url)
            ms = (time.perf_counter() - t) * 1000
            branch = Branch(payload["branch"], payload["reason"], payload.get("scores", {}))
            return branch, ms, f"server {url or client.server_url()}"
        except client.NoServer as exc:
            _say_local(exc)

    eng = inference.load(model)
    state = eng.clip(prompt, budget)
    t = time.perf_counter()
    answers = eng.predict(state, ROUTER_BANK)
    ms = (time.perf_counter() - t) * 1000
    return BIG_SMALL.decide(answers), ms, "in-process"


def decide_many(
    prompts: list[str],
    model: str = DEFAULT_MODEL,
    budget: int = PROMPT_TOKEN_BUDGET,
    *,
    url: str | None = None,
    use_server: bool = True,
):
    """Route many prompts. Returns (list[Branch], total_ms, source).

    Worth having even with no server running: in-process this loads the checkpoint once for the
    whole set instead of once per prompt, which is the difference between 2.1 s x N and
    2.1 s + 26 ms x N. Against a warm server it is one connection instead of N.
    """
    if use_server:
        try:
            t = time.perf_counter()
            results = client.route_batch(prompts, url)
            ms = (time.perf_counter() - t) * 1000
            branches = [Branch(r["branch"], r["reason"], r.get("scores", {})) for r in results]
            return branches, ms, f"server {url or client.server_url()}"
        except client.NoServer as exc:
            _say_local(exc)

    eng = inference.load(model)
    # Loading is lazy, so without this the first prompt pays the checkpoint load inside the timed
    # region and ms_per_prompt reads ~600 ms on a batch of three. The server has already paid it,
    # so warming first is also what makes the two sources comparable.
    eng.agent, eng.tokenizer
    t = time.perf_counter()
    branches = [
        BIG_SMALL.decide(eng.predict(eng.clip(p, budget), ROUTER_BANK)) for p in prompts
    ]
    ms = (time.perf_counter() - t) * 1000
    return branches, ms, "in-process"


def _batch_cmd(args: argparse.Namespace) -> int:
    """Read prompts from stdin, one per line; write one JSON object per line, in order."""
    prompts = [line.strip() for line in sys.stdin if line.strip()]
    if not prompts:
        print("verdict: no prompts on stdin", file=sys.stderr)
        return 2

    branches, ms, source = decide_many(
        prompts, args.model, args.budget, url=args.url, use_server=not args.no_server
    )
    for i, (prompt, branch) in enumerate(zip(prompts, branches)):
        if args.quiet:
            print(branch.name)
        else:
            print(json.dumps({"index": i, "prompt": prompt, **branch.as_dict()}))
    if not args.quiet:
        print(json.dumps({"summary": {
            "n": len(prompts), "total_ms": round(ms, 1),
            "ms_per_prompt": round(ms / len(prompts), 1), "source": source,
            "small": sum(b.name == SMALL for b in branches),
            "big": sum(b.name != SMALL for b in branches),
        }}))
    return 0


def _route_cmd(args: argparse.Namespace) -> int:
    from .. import config

    settings = support._settings()
    args.model = inference._resolved_model(args.model, settings)
    args.url = config.check_url(args.url, "--url") if args.url else settings.url
    args.budget = args.budget if args.budget is not None else settings.prompt_token_budget

    if args.batch:
        return _batch_cmd(args)
    if not args.prompt:
        print("verdict: give a prompt, or --batch to read them from stdin", file=sys.stderr)
        return 2
    branch, ms, source = decide(
        args.prompt, args.model, args.budget, url=args.url, use_server=not args.no_server
    )

    if args.quiet:
        print(branch.name)
    elif args.json:
        print(json.dumps(
            {**branch.as_dict(), "latency_ms": round(ms, 1), "source": source},
            indent=2,
        ))
    else:
        print(f"branch     {branch.name.upper()}  ({ms:.0f} ms via {source})")
        print(f"reason     {branch.reason}")
        print(f"scores     " + "  ".join(f"{k}={v:.2f}" for k, v in branch.scores.items()))
    if args.json:
        # As `ask --json`: the answer is in the JSON, so a success exits 0 whichever branch.
        return 0
    return 0 if branch.name == SMALL else 1


def _cases_cmd(args: argparse.Namespace) -> int:
    settings = support._settings()
    print(f"switch {BIG_SMALL.name!r}  default={BIG_SMALL.default!r}")
    for name, description in BIG_SMALL.cases.items():
        marker = "*" if name == BIG_SMALL.default else " "
        print(f" {marker} {name:6s} {description}")
    print(f"\nquestions ({len(BIG_SMALL.questions)}):")
    for qid, q in BIG_SMALL.questions.items():
        print(f"  {qid:14s} {q['type']:6s} {q['instructions']}")
    print("\n* is the default branch, taken whenever the model is unsure or absent.")
    print(f"settings: {settings.source or 'built-in defaults (run `verdict init`)'}")
    return 0
