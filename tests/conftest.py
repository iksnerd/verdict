"""No machine configuration reaches a test.

Settings too: with `lang = "multi"` in ~/.config/verdict/config.toml, four language tests failed
in the pre-commit export, and passed in the checkout only because its gitignored verdict.toml
shadowed the user file. No config file and no VERDICT_* variable reaches a test unless it sets one."""

from __future__ import annotations

import os

import pytest

from verdict import config


@pytest.fixture(autouse=True)
def _no_machine_config(monkeypatch):
    monkeypatch.setattr(config, "CONFIG_PATHS", ())
    for name in [n for n in os.environ if n.startswith("VERDICT_")]:
        monkeypatch.delenv(name)


def typed(questions, noul_answer):
    """A fake server's answers, typed as a real server types them. `noul_answer(qid)` is the
    yes/no answer the fake would give; a question the CLI rewrote as a no/yes choice gets the
    same number back as a choice, so fakes written for yes/no keep their meaning."""
    out = {}
    for qid, q in questions.items():
        a = noul_answer(qid)
        if q.get("type") == "choice" and list(q.get("criteria") or {}) == ["no", "yes"]:
            p = a["noul"]
            a = {"type": "choice", "choice": "yes" if p >= 0.5 else "no", "confidence": a["confidence"],
                 "probabilities": {"no": 1 - p, "yes": p}}
        out[qid] = a
    return out


@pytest.fixture(autouse=True)
def _no_real_server(monkeypatch):
    """A test never reaches a `verdict serve` that happens to be running on the developer's
    machine (the CLI asks a server what it fronts, at /v1/capabilities, before it answers). The
    network layer refuses unless the test installs its own transport."""
    import urllib.error

    from verdict import client

    def refuse(request, timeout=None):
        raise urllib.error.URLError("no server in tests")

    monkeypatch.setattr(client.urllib.request, "urlopen", refuse)
