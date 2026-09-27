"""Getting a machine ready to answer: `verdict serve` and `verdict init`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import inference, support


def _serve_cmd(args: argparse.Namespace) -> int:
    """Serve the real model, which is the point of having a server.

    `uvicorn verdict.api:app` serves the `uniform` backend, because that is what `create_app()`
    defaults to and it is the right default for a module import. A server answering `uniform` is
    worse than no server for routing: the client treats it as absent and falls back, so you pay
    the round trip and then load the model anyway.

    The checkpoint is loaded before the first request rather than lazily, so the first caller is
    not the one who pays the ~2 s.
    """
    import uvicorn

    from ..api import create_app
    from ..backend_mlx import MlxBackend

    import socket

    settings = support._settings()
    args.model = inference._resolved_model(args.model, settings)
    args.port = args.port if args.port is not None else settings.port

    # Check the port before loading 400 MB of weights and before claiming to serve. This printed
    # "serving ... on ..." and then died on a bind error, which is a lie in the worst place: the
    # client reads that line to decide the server is up.
    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((args.host, args.port))
    except OSError as exc:
        print(f"verdict: cannot serve on {args.host}:{args.port}: {exc}", file=sys.stderr)
        print(f"  something else is listening. `lsof -nP -iTCP:{args.port} -sTCP:LISTEN` names it.",
              file=sys.stderr)
        print("  pick another with --port, and point clients at it with $VERDICT_URL.",
              file=sys.stderr)
        return 2
    finally:
        probe.close()

    budget = args.budget if args.budget is not None else settings.prompt_token_budget
    bits = args.bits or settings.bits
    backend = MlxBackend(args.model, multilingual_id=settings.multilingual_path,
                         extra_checkpoints=settings.extra_checkpoints, budget=budget or None,
                         bits=bits)
    print(f"loading {args.model} ...", flush=True)
    backend.engine.agent
    backend.engine.tokenizer
    print(f"serving {backend.name} on http://{args.host}:{args.port}, "
          f"reading {budget or 'all'} tokens of each state", flush=True)
    print("  it binds to localhost and has no auth; do not expose it beyond this machine.")
    uvicorn.run(create_app(backend), host=args.host, port=args.port,
                log_level=args.log_level)
    return 0


def _port_free(host: str, port: int) -> bool:
    import socket

    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _ask(prompt: str, default: str, assume_yes: bool) -> str:
    if assume_yes or not sys.stdin.isatty():
        print(f"  {prompt} [{default}]")
        return default
    got = input(f"  {prompt} [{default}]: ").strip()
    return got or default


def _config_cmd(args: argparse.Namespace) -> int:
    """The resolved settings, each with its source."""
    import json

    from .. import config

    rows = config.explain()
    if args.json:
        print(json.dumps({"settings": rows}, indent=2))
        return 0
    width = max(len(r["key"]) for r in rows)
    for r in rows:
        value = json.dumps(r["value"]) if isinstance(r["value"], (dict, str)) else r["value"]
        print(f"{r['key']:{width}s}  {value}  ({r['source']})")
    return 0


def _init_cmd(args: argparse.Namespace) -> int:
    """Write a config from what this machine actually has.

    Staged and confirmed, and idempotent: an existing config is offered back as the default at
    every prompt, so re-running after a partial setup is safe. It writes the user-level file by
    default, with absolute paths, because verdict is called from every directory.
    """
    from .. import config

    # The file being rewritten is the one whose settings to keep, wherever --out points.
    target = Path(args.out).expanduser()
    current = config.load(target) if target.is_file() else config.load()
    total = 3
    print(f"verdict init: writing {args.out}")
    if current.source:
        print(f"  (reading current values from {current.source})")
    print()

    # 1. the checkpoints
    print(f"[1/{total}] model checkpoints")
    found = sorted(str(p.resolve()) for p in Path("models").glob("*-mlx")) if Path("models").is_dir() else []
    if found:
        print(f"  found: {', '.join(found)}")
    else:
        print(f"  none found under ./models, which is fine: base Laya ({config.FALLBACK_MODEL}) "
              "is used, downloaded on first use")
    default_path = current.model_path
    if not Path(default_path).is_absolute() and Path(default_path).is_dir():
        default_path = str(Path(default_path).resolve())
    elif found and not Path(default_path).is_dir():
        default_path = next((f for f in found if "verdict" in Path(f).name), found[0])
    model_path = _ask("path", default_path, args.yes)
    hub = not config._is_local(model_path)
    ok_model = hub or Path(model_path).expanduser().is_dir()
    print(f"  {'hub, downloaded on first use' if hub else 'ok' if ok_model else 'MISSING'}: {model_path}")
    multilingual = _ask("multilingual (loaded only by --lang multi)", current.multilingual_path,
                        args.yes)
    print()

    # 2. the port
    print(f"[2/{total}] server")
    host, port = "127.0.0.1", current.port
    if not _port_free(host, port):
        free = next((p for p in range(port, port + 40) if _port_free(host, p)), port)
        print(f"  {port} is taken (`lsof -nP -iTCP:{port} -sTCP:LISTEN` names it); {free} is free")
        port = free
    url = _ask("url", f"http://{host}:{port}", args.yes)
    print()

    # 3. write it
    print(f"[3/{total}] write")
    settings = config.Settings(
        url=url.rstrip("/"), model_path=model_path,
        prompt_token_budget=current.prompt_token_budget, multilingual_path=multilingual,
        bits=current.bits, lang=current.lang, extra_checkpoints=current.extra_checkpoints,
    )
    out = target
    if out.exists() and not args.yes and sys.stdin.isatty():
        if input(f"  {out} exists. Overwrite? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("  left alone.")
            return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(config.to_toml(settings))
    print(f"  wrote {out}")
    print()
    print("Next:")
    print(f"  verdict serve              # binds {url}; stop it when you are done")
    print('  verdict ask "commit the fix and push it" "Is this an instruction to perform an action?"')
    if not ok_model:
        print(f"\n  NOTE: {model_path} is missing, so verdict will use base laya "
              f"({config.FALLBACK_MODEL}) instead.")
    return 0
