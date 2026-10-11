"""`verdict serve --systemone URL`: answer with a /v1/systemone server instead of local laya.

The seams are the command's outputs: which backend the served app reports, what stderr says about
where states go, and what it refuses. No model loads and no network is touched.
"""
from __future__ import annotations

import pytest
import uvicorn
from fastapi.testclient import TestClient

import verdict.backend_mlx as backend_mlx
from verdict import cli, config
from verdict.cli import support


@pytest.fixture
def served(monkeypatch, tmp_path):
    """Run `serve` with uvicorn stubbed; returns the app it would have served."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CONFIG_PATHS", ())
    for var in ("VERDICT_SYSTEMONE_URL", "VERDICT_SYSTEMONE_MODEL", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    box = {}

    def no_mlx(*a, **k):
        raise AssertionError("a systemone backend must not load a local model")

    monkeypatch.setattr(backend_mlx, "MlxBackend", no_mlx)
    monkeypatch.setattr(uvicorn, "run", lambda app, **k: box.update(app=app))
    return box


def backend_name(box) -> str:
    return TestClient(box["app"]).get("/healthz").json()["backend"]


def test_serve_answers_with_the_named_server_and_loads_no_model(served, monkeypatch, capsys):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-live")
    assert cli.main(["serve", "--port", "0", "--systemone", "https://api.typesafe.ai"]) == 0
    assert backend_name(served) == "jev:jev-latest@https://api.typesafe.ai"


def test_serve_says_where_every_state_goes(served, monkeypatch, capsys):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-live")
    cli.main(["serve", "--port", "0", "--systemone", "https://api.typesafe.ai"])
    out = capsys.readouterr()
    text = out.out + out.err
    assert "https://api.typesafe.ai" in text and "sent to" in text and "billed" in text


def test_a_hosted_server_without_a_key_stops_before_serving(served, capsys):
    assert cli.main(["serve", "--port", "0", "--systemone", "https://api.typesafe.ai"]) == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err
    assert "app" not in served


def test_a_loopback_server_needs_no_key(served):
    assert cli.main(["serve", "--port", "0", "--systemone", "http://127.0.0.1:8009",
                     "--systemone-model", "kev-latest"]) == 0
    assert backend_name(served) == "systemone:kev-latest@http://127.0.0.1:8009"


def test_the_file_can_choose_the_backend_and_a_flag_beats_it(served, monkeypatch, tmp_path):
    (tmp_path / "verdict.toml").write_text(
        '[systemone]\nurl = "http://127.0.0.1:8009"\nmodel = "kev-latest"\n')
    monkeypatch.setattr(config, "CONFIG_PATHS", (config.Path("verdict.toml"),))
    assert cli.main(["serve", "--port", "0"]) == 0
    assert backend_name(served) == "systemone:kev-latest@http://127.0.0.1:8009"
    assert cli.main(["serve", "--port", "0", "--systemone", "http://127.0.0.1:8010"]) == 0
    assert backend_name(served) == "systemone:kev-latest@http://127.0.0.1:8010"


def test_a_malformed_url_flag_is_named(served, capsys):
    assert cli.main(["serve", "--port", "0", "--systemone", "api.typesafe.ai"]) == 2
    assert "http" in capsys.readouterr().err


def test_without_the_flag_or_a_setting_serve_is_still_local_laya(served, monkeypatch):
    built = {}

    class FakeMlx:
        def __init__(self, model_id, **kw):
            built["model"] = model_id
            self.name = f"fake:{model_id}"
            self.engine = type("E", (), {"agent": None, "tokenizer": None})()

    monkeypatch.setattr(backend_mlx, "MlxBackend", FakeMlx)
    assert cli.main(["serve", "--port", "0"]) == 0
    assert "model" in built and backend_name(served).startswith("fake:")


def test_a_hosted_backend_will_not_serve_beyond_this_machine(served, monkeypatch, capsys):
    """An unauthenticated proxy that forwards, with the owner's key, to a paid service."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-live")
    assert cli.main(["serve", "--port", "0", "--host", "0.0.0.0",
                     "--systemone", "https://api.typesafe.ai"]) == 2
    assert "localhost" in capsys.readouterr().err and "app" not in served


def test_a_local_upstream_may_be_served_on_any_host(served):
    assert cli.main(["serve", "--port", "0", "--host", "0.0.0.0",
                     "--systemone", "http://127.0.0.1:8009"]) == 0


def test_the_startup_lines_say_the_host_it_really_binds(served, monkeypatch, capsys):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-live")
    cli.main(["serve", "--port", "0", "--systemone", "https://api.typesafe.ai"])
    assert "127.0.0.1" in capsys.readouterr().out
