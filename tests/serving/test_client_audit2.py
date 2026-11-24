"""Audit 2: one resolver for the server URL, and no server for a uniform one. No network."""
from __future__ import annotations

import json
import urllib.error

import pytest

from verdict import client, config


def _file(monkeypatch, tmp_path, url):
    f = tmp_path / "verdict.toml"
    f.write_text(f'[server]\nurl = "{url}"\n')
    monkeypatch.setattr(config, "CONFIG_PATHS", (f,))


def test_server_url_reads_the_config_file(monkeypatch, tmp_path):
    _file(monkeypatch, tmp_path, "http://127.0.0.1:9123")
    assert client.server_url() == "http://127.0.0.1:9123"


def test_the_environment_beats_the_file_and_empty_is_ignored(monkeypatch, tmp_path):
    _file(monkeypatch, tmp_path, "http://127.0.0.1:9123")
    monkeypatch.setenv("VERDICT_URL", "http://127.0.0.1:9456/")
    assert client.server_url() == "http://127.0.0.1:9456"
    monkeypatch.setenv("VERDICT_URL", "")
    assert client.server_url() == "http://127.0.0.1:9123"


def test_an_empty_variable_without_a_file_is_the_default(monkeypatch):
    monkeypatch.setenv("VERDICT_URL", "")
    assert client.server_url() == client.DEFAULT_URL


def test_a_malformed_variable_is_a_named_config_error(monkeypatch):
    monkeypatch.setenv("VERDICT_URL", "elsewhere")
    with pytest.raises(config.ConfigError, match="VERDICT_URL"):
        client.server_url()


def test_a_base_with_no_scheme_is_no_server_not_a_valueerror():
    with pytest.raises(client.NoServer):
        client.route("x", url="no-scheme-here")


def _serve(monkeypatch, payload, headers=None):
    class R:
        def __init__(self):
            self.headers = headers or {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(payload).encode()

    monkeypatch.setattr(client.urllib.request, "urlopen", lambda *a, **k: R())


def test_capabilities_of_a_uniform_server_is_no_server(monkeypatch):
    _serve(monkeypatch, {"family": "uniform", "model": "uniform"})
    with pytest.raises(client.NoServer, match="uniform"):
        client.capabilities("http://x:1")


def test_models_of_a_uniform_server_is_no_server(monkeypatch):
    _serve(monkeypatch, {"models": [{"name": "english"}]}, {"X-Verdict-Backend": "uniform"})
    with pytest.raises(client.NoServer, match="uniform"):
        client.models("http://x:1")


def test_models_of_a_real_server_still_lists(monkeypatch):
    _serve(monkeypatch, {"models": [{"name": "english"}]})
    assert client.models("http://x:1") == [{"name": "english"}]
