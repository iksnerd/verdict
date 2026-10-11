"""Where the settings come from, and in what order.

Everything here had a hardcoded default somewhere in the code, and one was wrong in a way nothing
caught until the tool was used: the fix is not a better guess but letting the machine say what it
has (`verdict init`). There is no `[routes]` table any more: verdict answers and never dispatches,
so a file that still has one gets a warning rather than being silently half-read.

Precedence, highest first:

1. an explicit CLI flag
2. an environment variable
3. `verdict.toml` in the working directory, then `~/.config/verdict/config.toml`
4. the built-in default

Read with `tomllib`, stdlib from 3.11. It costs about 6.4 ms to import (it pulls `string` and
`re`), which is real against the CLI's 18.7 ms of fixed cost, so it is imported inside `load()`
only once a config file has actually been found. A machine with no config file never pays it.
Nothing here imports anything heavy.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

#: Searched in order; the first that exists wins. A project-local file beats the user-level one so
#: a checkout can pin its own model without touching the machine's settings.
USER_CONFIG = Path.home() / ".config" / "verdict" / "config.toml"
CONFIG_PATHS = (Path("verdict.toml"), USER_CONFIG)

DEFAULTS: dict[str, Any] = {
    "server": {"url": "http://127.0.0.1:8799"},
    # `multilingual` is laya's own mmBERT checkpoint for non-English state. It loads only when a
    # call asks for it (`--lang multi`), never at startup: a second model on the GPU is a choice.
    # `extra` names further laya checkpoints the same way: loaded on first request that names
    # them, never at startup, and stay resident once loaded (§ one model at a time still applies
    # to what you start deliberately, not to what a request asked for).
    "model": {"path": "aac6fef/laya-mlx", "prompt_token_budget": 128,
              "multilingual": "aac6fef/laya-multilingual-mlx", "bits": 16, "lang": "auto",
              "extra": {}},
    # A backend other than local Laya for `verdict serve`: any /v1/systemone server, Jev included.
    # Unset by default, so nothing leaves this machine unless it is named here (or by a flag).
    # The API key is deliberately not a setting: $TYPESAFE_API_KEY only.
    "systemone": {"url": "", "model": "jev-latest"},
}

#: Environment overrides, as (section, key) -> variable name.
ENV_OVERRIDES = {
    ("server", "url"): "VERDICT_URL",
    ("model", "path"): "VERDICT_MODEL",
    ("model", "multilingual"): "VERDICT_MULTILINGUAL",
    ("model", "bits"): "VERDICT_BITS",
    ("model", "lang"): "VERDICT_LANG",
    ("systemone", "url"): "VERDICT_SYSTEMONE_URL",
    ("systemone", "model"): "VERDICT_SYSTEMONE_MODEL",
}


class Settings:
    """Resolved settings. A plain class for the same reason `switch.Branch` is: the stdlib
    dataclass machinery pulls `inspect`, worth 3.7 ms, and this is eight fields with no generated
    behaviour worth it. Using it here would undo the saving §19 just made."""

    __slots__ = ("url", "model_path", "prompt_token_budget", "multilingual_path", "source",
                 "defaulted", "bits", "lang", "extra_checkpoints", "systemone_url",
                 "systemone_model")

    def __init__(self, url, model_path, prompt_token_budget,
                 multilingual_path=DEFAULTS["model"]["multilingual"], source=None, defaulted=(),
                 bits=16, lang="auto", extra_checkpoints=None, systemone_url=None,
                 systemone_model=DEFAULTS["systemone"]["model"]):
        self.url = url
        #: Which checkpoint reads the state: auto (English, warn on other languages), en, multi.
        self.lang = lang
        #: 16 is the checkpoint as shipped; 8 quantizes it at load (FINDINGS §36).
        self.bits = bits
        self.model_path = model_path
        self.prompt_token_budget = prompt_token_budget
        self.multilingual_path = multilingual_path
        #: Further named laya checkpoints, loaded on first request that names them (config.py's
        #: [model.extra]). A request's `model` selects one by name; unknown names are refused.
        self.extra_checkpoints = dict(extra_checkpoints) if extra_checkpoints else {}
        #: A /v1/systemone server to answer instead of local Laya (None: local). Every state is
        #: sent to it. The key is `systemone_key`, never stored here.
        self.systemone_url = systemone_url
        self.systemone_model = systemone_model
        #: The file it came from, or None when everything is a default.
        self.source = source
        #: Sections the loaded file did not set, so `verdict init` can say what it will add.
        self.defaulted = tuple(defaulted)

    @property
    def port(self) -> int:
        return int(self.url.rsplit(":", 1)[1])

    def __repr__(self) -> str:
        return (f"Settings(url={self.url!r}, model_path={self.model_path!r}, "
                f"source={str(self.source)!r})")


def find_config(start: Path | None = None) -> Path | None:
    for candidate in CONFIG_PATHS:
        path = (start / candidate) if (start and not candidate.is_absolute()) else candidate
        if path.is_file():
            return path
    return None


def _merge(base: dict, over: dict) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


#: What a config file may set; anything else is a typo that would otherwise be ignored in silence.
KNOWN = {"server": {"url"},
         "model": {"path", "prompt_token_budget", "multilingual", "bits", "lang", "extra"},
         "systemone": {"url", "model"}}


def _read(found: Path) -> dict[str, Any]:
    import tomllib  # 6.4 ms, and only when there is a file to read

    try:
        loaded = tomllib.loads(found.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{found} is not valid TOML: {exc}") from exc
    if "routes" in loaded:
        import sys

        print(f"verdict: {found} has a [routes] table, which is no longer read: verdict "
              "answers and never dispatches. Delete the table to silence this.",
              file=sys.stderr)
        del loaded["routes"]
    for section, value in loaded.items():
        if section not in KNOWN:
            raise ConfigError(f"{found}: unknown section [{section}]; there are "
                              f"{', '.join(f'[{k}]' for k in KNOWN)}")
        if not isinstance(value, dict):
            raise ConfigError(f"{found}: {section} must be a [{section}] table")
        unknown = sorted(set(value) - KNOWN[section])
        if unknown:
            raise ConfigError(f"{found}: unknown key {', '.join(unknown)} in [{section}]; it takes "
                              f"{', '.join(sorted(KNOWN[section]))}")
    return loaded


def config_files(path: Path | None = None) -> list[Path]:
    """The files that apply, least specific first: `path` alone when given, else every existing
    file on the search path, so ./verdict.toml layers over the user config key by key."""
    if path is not None:
        return [path] if path.is_file() else []
    return [p for p in reversed(CONFIG_PATHS) if p.is_file()]


def load(path: Path | None = None) -> Settings:
    """Resolve settings. `path` forces one file; otherwise every file on the search path applies,
    the more specific overriding the less, key by key."""
    files = config_files(path)
    found = files[-1] if files else (path if path is not None else None)
    loaded: dict[str, Any] = {}
    for f in files:
        loaded = _merge(loaded, _read(f))

    data = _merge(DEFAULTS, loaded)
    defaulted = tuple(k for k in DEFAULTS if k not in loaded)

    for (section, key), var in ENV_OVERRIDES.items():
        value = os.environ.get(var)
        if value:
            data[section][key] = value

    return Settings(
        url=_url(data["server"]["url"]),
        model_path=str(data["model"]["path"]),
        prompt_token_budget=_budget(data["model"]["prompt_token_budget"]),
        multilingual_path=str(data["model"]["multilingual"]),
        source=found,
        defaulted=defaulted,
        bits=_bits(data["model"]["bits"]),
        lang=_lang(data["model"]["lang"]),
        extra_checkpoints=_extra(data["model"]["extra"]),
        systemone_url=_systemone_url(data["systemone"]["url"]),
        systemone_model=_systemone_model(data["systemone"]["model"]),
    )


def explain(path: Path | None = None) -> list[dict[str, Any]]:
    """Every setting as `load` resolves it, with where it came from: `$VAR`, the file's path, or
    "default". Flags are per command, so they are not in this list."""
    settings = load(path)
    #: Most specific first, so the first file that sets a key is the one that won.
    files = [(f, _read(f)) for f in reversed(config_files(path))]
    values = {("server", "url"): settings.url, ("model", "path"): settings.model_path,
              ("model", "prompt_token_budget"): settings.prompt_token_budget,
              ("model", "multilingual"): settings.multilingual_path,
              ("model", "bits"): settings.bits, ("model", "lang"): settings.lang,
              ("model", "extra"): settings.extra_checkpoints,
              ("systemone", "url"): settings.systemone_url,
              ("systemone", "model"): settings.systemone_model}
    rows = []
    for (section, key), value in values.items():
        var = ENV_OVERRIDES.get((section, key))
        if var and os.environ.get(var):
            source = f"${var}"
        else:
            source = next((str(f) for f, data in files if key in data.get(section, {})),
                          "default")
        rows.append({"key": f"{section}.{key}", "value": value, "source": source})
    return rows


class ConfigError(ValueError):
    """A setting the CLI reports and stops on, rather than a bug to trace."""


def check_url(value, where: str) -> str:
    """The same check for a `--url` flag, which never passes through `load`."""
    return _url(value, where)


def _url(value, where: str | None = None) -> str:
    """`http://host:port`, checked here so a bad one is named once instead of surfacing as an
    `int()` or `unknown url type` traceback wherever the port or the request is first built."""
    from urllib.parse import urlsplit

    where = where or ("$VERDICT_URL" if os.environ.get("VERDICT_URL") == value
                      else "[server].url")
    url = str(value).rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError(f"{where} is {value!r}; it takes http://host:port, like "
                          f"{DEFAULTS['server']['url']}")
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        raise ConfigError(f"{where} is {value!r} with no port; `verdict serve` binds one, e.g. "
                          f"{DEFAULTS['server']['url']}")
    return url


def _budget(value) -> int:
    try:
        budget = int(value)
    except (TypeError, ValueError):
        budget = -1
    if budget < 0:
        raise ConfigError(f"[model].prompt_token_budget is {value!r}; it takes a whole number of "
                          "tokens, 0 for the whole state (128 by default, FINDINGS §11)")
    return budget


LANGS = ("auto", "en", "multi")


def _lang(value) -> str:
    if value not in LANGS:
        raise ConfigError(f"[model].lang (or $VERDICT_LANG) is {value!r}; it takes auto, en or multi. "
                          "multi reads the state with laya's multilingual checkpoint, which Cyrillic "
                          "and other non-English text needs")
    return value


def _systemone_url(value, where: str | None = None) -> str | None:
    """None when unset; otherwise http(s)://host[:port]. A port is optional: api.typesafe.ai has
    none. A remote server must be https: the API key travels in a header and every state in the
    body, so cleartext is only acceptable on loopback."""
    from urllib.parse import urlsplit

    if not value:
        return None
    where = where or ("$VERDICT_SYSTEMONE_URL" if os.environ.get("VERDICT_SYSTEMONE_URL") == value
                      else "[systemone].url")
    url = str(value).rstrip("/")
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ConfigError(f"{where} is {value!r}; it takes http(s)://host[:port], like "
                          "https://api.typesafe.ai")
    if parts.scheme == "http" and parts.hostname not in LOOPBACK:
        raise ConfigError(f"{where} is {value!r}; a remote server needs https, because "
                          "$TYPESAFE_API_KEY is sent in a header and every state in the body. "
                          "Plain http is only for loopback")
    return url


def check_systemone_url(value, where: str = "--systemone") -> str:
    """The same check for a `--systemone` flag, which never passes through `load`."""
    return _systemone_url(value, where)


def _systemone_model(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"[systemone].model is {value!r}; it takes the model name the server "
                          "expects, like jev-latest")
    return value.strip()


LOOPBACK = ("127.0.0.1", "localhost", "::1")


def systemone_key(url: str) -> str:
    """The key for a /v1/systemone server: $TYPESAFE_API_KEY, or a placeholder for loopback,
    where a local server (verdict itself, Kev) ignores it. Never read from a file: `verdict init`
    writes the config and people commit it."""
    from urllib.parse import urlsplit

    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    if (urlsplit(url).hostname or url) in LOOPBACK:
        return "local"
    raise ConfigError(f"{url} needs $TYPESAFE_API_KEY. Every state asked about is sent to that "
                      "service and billed per input token")


def _extra(value) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ConfigError(f"[model.extra] is {value!r}; it takes a table of name = \"hf-id-or-path\" "
                          "pairs, one per additional checkpoint a request can name")
    return {str(k): str(v) for k, v in value.items()}


#: Widths with a measurement behind them. 4-bit stayed inside the bench intervals but moved 17
#: of 200 injection choices (FINDINGS §36), so it is not offered.
BITS = (16, 8)


def _bits(value) -> int:
    try:
        bits = int(value)
    except (TypeError, ValueError):
        bits = None
    if bits not in BITS:
        raise ConfigError(f"[model].bits (or $VERDICT_BITS) is {value!r}; it takes 16 or 8. 8 keeps "
                         "every bench suite and halves memory; 4 moved too many answers "
                         "(FINDINGS §36)")
    return bits


def _toml_str(value: str) -> str:
    """A TOML basic string, usable as a value or a quoted key (`support.v2` unquoted is a nested
    table). JSON's escapes are TOML's, except that TOML also forbids a raw DEL."""
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def _extra_toml(extra: dict[str, str]) -> str:
    """`[model.extra]`, only when there is one: an empty table that always reads back empty is
    still a key doing nothing until a request actually names one."""
    if not extra:
        return ('# [model.extra] names further checkpoints a request can select by name; none '
                'are configured.\n# support = "org/support-mlx"')
    lines = "\n".join(f"{_toml_str(name)} = {_toml_str(path)}" for name, path in extra.items())
    return f"[model.extra]\n{lines}"


def _systemone_toml(s: Settings) -> str:
    """`[systemone]`, only when one is chosen. Unset, it is a comment: a table that is present
    reads as a decision to send every state to a remote service."""
    if not s.systemone_url:
        return ('# [systemone] answers with a /v1/systemone server (TypeSafe Jev, Kev, another verdict)\n'
                '# instead of local laya, and sends it every state. The key is $TYPESAFE_API_KEY only.\n'
                '# url = "https://api.typesafe.ai"\n# model = "jev-latest"')
    return (f"[systemone]\n# Every state is sent to this server. The key is $TYPESAFE_API_KEY only.\n"
            f"url = {_toml_str(s.systemone_url)}\nmodel = {_toml_str(s.systemone_model)}")


def to_toml(s: Settings) -> str:
    """Render settings as the file `verdict init` writes. Hand-rolled because `tomllib` reads only,
    and the shape is two small tables rather than a reason to take a dependency."""
    return f"""# verdict settings. Written by `verdict init`; safe to hand-edit.
# Precedence: CLI flag > environment variable > this file > built-in default.

[server]
# `verdict serve` binds here and every command asks here. $VERDICT_URL overrides it.
url = {_toml_str(s.url)}

[model]
path = {_toml_str(s.model_path)}
# Tokens of the prompt the model sees. 128 costs about 36 ms a question against 87 ms at 512,
# and routing intent is nearly always stated up front (FINDINGS section 11).
prompt_token_budget = {s.prompt_token_budget}
# laya's multilingual checkpoint, for state that is not English. Loaded only by `--lang multi`.
multilingual = {_toml_str(s.multilingual_path)}
# 16 loads the checkpoint as shipped; 8 quantizes it at load, about 430 MB of GPU memory instead of
# 800, with the same answers on every bench suite (FINDINGS §36). $VERDICT_BITS overrides it.
bits = {s.bits}
# Which checkpoint reads the state. auto uses the English one and warns on other languages; multi
# uses the multilingual one for everything, which Cyrillic and other non-English text needs (a
# Bulgarian positive review: "negative 0.72" English, "positive 1.00" multilingual). $VERDICT_LANG
# and --lang override it.
lang = {_toml_str(s.lang)}

{_extra_toml(s.extra_checkpoints)}

{_systemone_toml(s)}

# There is deliberately no [thresholds] table. The cuts are applied by the switch inside the
# server process, so a value here would be read and ignored on the server path, and a config key
# that silently does nothing is worse than no key. They are a measured artifact anyway: refit with
# `python -m verdict.router_eval --refit`, which prints the two lines to paste into router.py.
"""


#: Loaded when the configured fine-tuned checkpoint is missing. laya-mlx downloads it on first use.
FALLBACK_MODEL = "aac6fef/laya-mlx"


#: The old built-in default: a local checkpoint that has the `owner/name` shape of a Hub id, so it
#: is named here explicitly. Configs written before base Laya became the default may still hold it.
LEGACY_LOCAL = "models/verdict-v1-mlx"


def _is_local(path: str) -> bool:
    """A filesystem path rather than a Hugging Face id."""
    return path.startswith(("/", ".", "~")) or path == LEGACY_LOCAL or path.count("/") != 1


def resolve_model(path: str, explicit: bool = False) -> tuple[str, str | None]:
    """The checkpoint to load, and a warning when it is not the one configured.

    A configured local checkpoint that is missing falls back to base laya rather than failing:
    with every new yes/no asked as a no/yes choice, base laya is within noise of the fine-tune
    (FINDINGS §40), which beats no answer, provided the caller is told. A `--model` the caller
    typed is never replaced: a typo there should fail, not quietly load something else.
    """
    if explicit or Path(path).expanduser().exists() or not _is_local(path):
        return path, None
    return FALLBACK_MODEL, (
        f"{path} not found, so using base laya ({FALLBACK_MODEL}), downloaded on first use. "
        "It is within noise of the fine-tune on the bench's yes/no and choice sets "
        "(FINDINGS §40); set model.path to use another checkpoint.")
