"""Trivial cross-cutting helpers shared by every command group: resolved settings, the one place
that prints a usage failure and its exit code, the user config path, and the installed version
string. Nothing here loads a model; see `inference.py` for that."""
from __future__ import annotations

import sys


def _settings():
    """Resolved settings. Loaded inside a command rather than at import: `verdict.config` costs
    7.7 ms and `verdict --help` should not pay it."""
    from .. import config

    return config.load()


#: Set by `main` when the command was asked for --json or --jsonl: a failure then also prints
#: {"error": {"code", "message"}} on stdout, so a script parsing stdout gets JSON either way.
JSON_ERRORS = False

#: Stable codes an agent can branch on, by exception class name. Anything else is "usage".
_CODES = {"QuestionError": "refused", "NoServer": "no_server", "ServerTimeout": "server_timeout",
          "ServerError": "server_error", "ConfigError": "config", "FileNotFoundError": "not_found"}


def _fail(error: str | BaseException, code: str | None = None, **extra) -> int:
    """One `verdict:` line on stderr, exit 2; also a JSON error on stdout in JSON mode. Pass the
    exception itself where there is one, so its code is kept."""
    message = str(error)
    if code is None:
        code = "usage" if isinstance(error, str) else _CODES.get(type(error).__name__, "usage")
    print(f"verdict: {message}", file=sys.stderr)
    if JSON_ERRORS:
        import json

        print(json.dumps({"error": {"code": code, "message": message, **extra}}), flush=True)
    return 2


def _suggest(name: str, choices, end: str = ".") -> str:
    """"; did you mean X?" for a near miss, else `end`, so a sentence closes either way."""
    import difflib

    close = difflib.get_close_matches(name, list(choices), n=1, cutoff=0.6)
    return f"; did you mean {close[0]}?" if close else end


def _user_config():
    from ..config import USER_CONFIG

    return USER_CONFIG


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return f"verdict {version('verdict')}"
    except PackageNotFoundError:
        return "verdict (not installed)"
