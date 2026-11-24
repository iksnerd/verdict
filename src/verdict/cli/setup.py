"""Getting a machine ready to answer: `verdict serve` and `verdict init`."""
from __future__ import annotations

import argparse
import os
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

    from .. import config
    from ..api import create_app

    settings = support._settings()
    remote = (config.check_systemone_url(args.systemone) if args.systemone
              else settings.systemone_url)
    if remote:
        # Fail on a missing key before binding anything or claiming to serve.
        key = config.systemone_key(remote)
    else:
        args.model = inference._resolved_model(args.model, settings)
    args.port = args.port if args.port is not None else settings.port

    # Check the port before loading 400 MB of weights and before claiming to serve. This printed
    # "serving ... on ..." and then died on a bind error, which is a lie in the worst place: the
    # client reads that line to decide the server is up.
    if not _port_free(args.host, args.port):
        return _port_taken(args)

    if remote:
        return _serve_remote(args, settings, remote, key, create_app, uvicorn)

    from ..backends.mlx import MlxBackend

    budget = inference._budget(args, settings)
    bits = args.bits or settings.bits
    backend = MlxBackend(args.model, multilingual_id=settings.multilingual_path,
                         extra_checkpoints=settings.extra_checkpoints, budget=budget or None,
                         bits=bits)
    print(f"loading {args.model} ...", flush=True)
    backend.engine.agent
    backend.engine.tokenizer
    # The load took seconds, so the port can have been taken since the first check. Look again
    # before claiming to serve.
    if not _port_free(args.host, args.port):
        return _port_taken(args)
    print(f"serving {backend.name} on http://{args.host}:{args.port}, "
          f"reading {budget or 'all'} tokens of each state", flush=True)
    if _is_loopback(args.host):
        print("  it binds to localhost and has no auth; do not expose it beyond this machine.")
    else:
        print(f"  WARNING: it binds to {args.host}, so it is reachable from other machines, "
              "and has no auth.")
    return _run(uvicorn, create_app(backend), args)


def _port_taken(args: argparse.Namespace) -> int:
    support._warn(f"cannot serve on {args.host}:{args.port}: the port is in use")
    print(f"  something else is listening. `lsof -nP -iTCP:{args.port} -sTCP:LISTEN` names it.",
          file=sys.stderr)
    print("  pick another with --port, and point clients at it with $VERDICT_URL.",
          file=sys.stderr)
    return 2


def _run(uvicorn, app, args: argparse.Namespace) -> int:
    """uvicorn.run, but a bind it loses is a failure (2), not uvicorn's `sys.exit(1)`, which
    here reads as an answer."""
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    except SystemExit as exc:
        if exc.code not in (0, None):
            # Uvicorn exits 1 for a lost bind and also for a failure in startup; name the port
            # only when it really is not bindable.
            why = (f"; is {args.host}:{args.port} taken?" if not _port_free(args.host, args.port)
                   else "; the port is free, so see the lines above for why")
            support._warn(f"the server stopped without serving (exit {exc.code}){why}")
            return 2
    return 0


def _is_loopback(host: str) -> bool:
    """Whether a `--host` keeps the server on this machine: `localhost`, `::1` or 127.0.0.0/8.
    Any other name is taken as reachable, since saying "localhost" wrongly is the worse error."""
    import ipaddress

    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _serve_remote(args, settings, url, key, create_app, uvicorn) -> int:
    """Serve a /v1/systemone server (Jev, Kev, another verdict) behind verdict's own API.

    No model loads here. The request goes upstream as given, so laya's clipping and rewrites do
    not apply; the refusals do (api.py), so a question gets the same answer to "may I ask this"
    whichever backend is behind it. Every state leaves this machine, which the first lines say.
    """
    from urllib.parse import urlsplit

    from ..backends.systemone import SystemOneBackend
    from ..config import LOOPBACK, ConfigError

    if (urlsplit(url).hostname or "") not in LOOPBACK and args.host not in LOOPBACK:
        raise ConfigError(f"--host {args.host} would expose an unauthenticated server that forwards "
                          f"to {url} with $TYPESAFE_API_KEY, and bills it. Bind to localhost")
    model = args.systemone_model or settings.systemone_model
    if (urlsplit(url).hostname or "") == "api.typesafe.ai":
        # Hosted Jev goes through TypeSafe's own SDK (retries, request ids); anything else
        # speaking the protocol (Kev, another verdict) keeps the dependency-free client.
        try:
            from ..backends.jev import JevBackend
        except ImportError as exc:
            from . import update

            version = support._version().split()[-1]
            spec = update._tool_spec([*update.TOOL_EXTRAS.split(","), "jev"], version)
            raise ConfigError(
                "hosted Jev uses TypeSafe's SDK, which is not installed here. Add it with: "
                f"{update._tool_install_cmd(spec)} "
                "(or `uv sync --extra jev` in a checkout). Not `pip install verdict`: "
                "that name on PyPI is a different project") from exc
        backend = JevBackend(model, api_key=key, base_url=url)
    else:
        backend = SystemOneBackend(url, model, key)
    billed = "" if (urlsplit(url).hostname or url) in LOOPBACK else " and billed per input token"
    print(f"serving {backend.name} on http://{args.host}:{args.port}", flush=True)
    print(f"  every state asked about is sent to {url}{billed}; nothing is read locally.",
          flush=True)
    print(f"  it binds to {args.host} and has no auth; do not expose it beyond this machine.",
          flush=True)
    try:
        return _run(uvicorn, create_app(backend), args)
    finally:
        backend.close()


def _verdict_listens(url: str) -> bool:
    """Whether what holds the port is a verdict server. Its own server is the usual reason the
    configured port is taken, and moving the config off it would point every client at nothing."""
    from .. import client

    try:
        return bool(client.capabilities(url).get("family"))
    except (client.NoServer, client.ServerError, client.ServerTimeout):
        return False


def _port_free(host: str, port: int) -> bool:
    """Whether nothing holds host:port. A bind alone is not enough: on macOS a SO_REUSEADDR bind
    to 127.0.0.1 succeeds while a 0.0.0.0 listener holds the port, so also try to connect."""
    import socket

    try:
        # The family follows the host: an AF_INET socket cannot bind `::1`.
        family, _, _, _, addr = socket.getaddrinfo(
            host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)[0]
    except (OSError, IndexError):
        family, addr = socket.AF_INET, (host, port)  # the bind below reports what is wrong
    probe = socket.socket(family, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(addr)
    except OSError:
        return False
    finally:
        probe.close()
    if port:
        target = {"": "127.0.0.1", "0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
        try:
            with socket.create_connection((target, port), timeout=0.5):
                return False
        except Exception:  # refused, timed out, unreachable: nothing is listening
            pass
    return True


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


def _load_files_only(path: Path | None):
    """`config.load` without the environment: a new file is written from the files on disk, so a
    `$VERDICT_URL` set for one shell is not frozen into every later one."""
    from .. import config

    saved = {v: os.environ.pop(v) for v in config.ENV_OVERRIDES.values() if v in os.environ}
    try:
        return config.load(path)
    finally:
        os.environ.update(saved)


def _init_cmd(args: argparse.Namespace) -> int:
    """Write a config from what this machine actually has.

    Staged and confirmed, and idempotent: an existing config is offered back as the default at
    every prompt, so re-running after a partial setup is safe. It writes the user-level file by
    default, with absolute paths, because verdict is called from every directory.
    """
    from .. import config

    # The file being rewritten is the one whose settings to keep, wherever --out points.
    target = Path(args.out).expanduser()
    current = _load_files_only(target if target.is_file() else None)
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
    if not _port_free(host, port) and _verdict_listens(f"http://{host}:{port}"):
        print(f"  {port} is held by a running verdict server; keeping it")
    elif not _port_free(host, port):
        free = next((p for p in range(port, port + 40) if _port_free(host, p)), None)
        taken = f"{port} is taken (`lsof -nP -iTCP:{port} -sTCP:LISTEN` names it)"
        if free is None:
            print(f"  {taken}, and so is every port up to {port + 39}; keeping {port}: "
                  "pass --port to `verdict serve`, or edit the url below")
        else:
            print(f"  {taken}; {free} is free")
            port = free
    url = _ask("url", f"http://{host}:{port}", args.yes)
    print()

    # 3. write it
    print(f"[3/{total}] write")
    settings = config.Settings(
        url=url.rstrip("/"), model_path=model_path,
        prompt_token_budget=current.prompt_token_budget, multilingual_path=multilingual,
        bits=current.bits, lang=current.lang, extra_checkpoints=current.extra_checkpoints,
        systemone_url=current.systemone_url, systemone_model=current.systemone_model,
    )
    out = target
    if out.exists() and not args.yes and not sys.stdin.isatty():
        return support._fail(f"{out} exists and stdin is not a terminal, so there is no one to "
                             "confirm the overwrite; pass --yes to overwrite it")
    if out.exists() and not args.yes:
        if input(f"  {out} exists. Overwrite? [y/N]: ").strip().lower() not in ("y", "yes"):
            print("  left alone.")
            return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    # Swapped in atomically, so Ctrl-C mid-write cannot leave a truncated config that every later
    # command would reject.
    inference.write_atomic(out, config.to_toml(settings))
    print(f"  wrote {out}")
    print()
    print("Next:")
    print(f"  verdict serve              # binds {url}; stop it when you are done")
    print('  verdict ask "commit the fix and push it" "Is this an instruction to perform an action?"')
    if not ok_model:
        print(f"\n  NOTE: {model_path} is missing, so verdict will use base laya "
              f"({config.FALLBACK_MODEL}) instead.")
    return 0
