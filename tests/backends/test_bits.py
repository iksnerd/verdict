"""`[model].bits = 8`: quantize the checkpoint at load, the way FINDINGS §36 measured it.

8-bit kept every bench suite inside the v0.7.1 intervals and changed 3 of 1,040 choices, at half
the memory (804 -> 429 MB active). 4-bit moved too many answers, so it is refused. The weights
file stays FP16: laya-mlx loads nothing else, so this saves memory, not disk."""
from __future__ import annotations

import types

import pytest

from verdict import cli, config
from verdict.backends import engine
from verdict.cli import inference, support
from fakes import FakeEngine, fake_mlx_serve, install_laya


def test_bits_default_to_full_precision():
    assert config.load().bits == 16


def test_bits_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("VERDICT_BITS", "8")
    assert config.load().bits == 8


def test_bits_come_from_the_file(config_files):
    config_files('[model]\nbits = 8\n')
    assert config.load().bits == 8


@pytest.mark.parametrize("bad", ["4", "7", "32"])
def test_only_measured_widths_are_accepted(monkeypatch, bad):
    monkeypatch.setenv("VERDICT_BITS", bad)
    with pytest.raises(ValueError, match="§36"):
        config.load()


def test_init_writes_the_bits_line():
    text = config.to_toml(config.load())
    assert "bits = 16" in text and "§36" in text


# ---- the engine ----------------------------------------------------------------------------------

def fake_laya(monkeypatch, loaded):
    agent = types.SimpleNamespace(model=object())
    agent._inference = agent.model
    install_laya(monkeypatch, lambda model_id, **kw: loaded.append(kw) or agent)
    return agent


def test_an_8_bit_engine_quantizes_once_at_load_and_says_so(monkeypatch):
    loaded, quantized = [], []
    agent = fake_laya(monkeypatch, loaded)
    monkeypatch.setattr(engine, "quantize", lambda a, bits: quantized.append((a, bits)))
    e = engine.load("m", bits=8)
    assert e.agent is agent and e.agent is agent
    assert quantized == [(agent, 8)]
    assert loaded == [{}], "bits is ours, never passed to laya_mlx.load"
    assert e.name.endswith("q8")


def test_a_full_precision_engine_is_untouched(monkeypatch):
    fake_laya(monkeypatch, [])
    monkeypatch.setattr(engine, "quantize", lambda *a: pytest.fail("quantized at 16 bits"))
    e = engine.load("m")
    e.agent
    assert not e.name.endswith("q8")


def test_an_injected_agent_is_not_quantized_so_its_name_does_not_say_q8(monkeypatch):
    monkeypatch.setattr(engine, "quantize", lambda *a: pytest.fail("quantized an injected agent"))
    agent = object()
    e = engine.Engine("m", agent=agent, bits=8)
    assert e.agent is agent
    assert e.name == "laya-mlx:m"
    assert e.bits == 8, "bits is what the caller asked for; the name is what is served"


def test_a_lazy_8_bit_engine_is_named_q8_before_it_loads(monkeypatch):
    """`MlxBackend` reads the name at construction, before any weights load."""
    fake_laya(monkeypatch, [])
    monkeypatch.setattr(engine, "quantize", lambda a, bits: None)
    assert engine.load("m", bits=8).name == "laya-mlx:m q8"


def test_a_failed_quantize_never_serves_the_fp16_agent_under_a_q8_name(monkeypatch):
    loaded, attempts = [], []
    agent = fake_laya(monkeypatch, loaded)

    def boom(a, bits):
        attempts.append(bits)
        if len(attempts) < 3:
            raise RuntimeError("out of memory")

    monkeypatch.setattr(engine, "quantize", boom)
    e = engine.load("m", bits=8)
    for _ in range(2):  # every access retries; none hands back the unquantized agent
        with pytest.raises(RuntimeError, match="out of memory"):
            e.agent
    assert e.agent is agent and e.name.endswith("q8")
    assert attempts == [8, 8, 8] and len(loaded) == 3


def test_compile_and_quantize_are_refused_together():
    """A compiled forward pass would keep the FP16 graph, so the quantization would do nothing."""
    with pytest.raises(ValueError, match="compile"):
        engine.load("m", bits=8, compile=True)


def test_quantize_is_what_section_36_measured():
    """On a real (tiny) laya model: every quantizable layer but the action head, group 64."""
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    model_mod = pytest.importorskip("laya_mlx.model")
    cfg = {"model_type": "modernbert", "vocab_size": 128, "hidden_size": 64,
           "intermediate_size": 128, "num_hidden_layers": 1, "num_attention_heads": 1,
           "local_attention": 16, "max_position_embeddings": 256}
    agent_cfg = {"encoder": "t", "head_layers": 1, "max_len": 128, "head_max_len": 32,
                 "act_costs": {"escalate": 0.5}, "temperature": [1, 1, 1]}
    model = model_mod.DecisionModel(model_mod.EncoderConfig.from_dict(cfg), agent_cfg)
    agent = types.SimpleNamespace(model=model)
    agent._inference = model
    engine.quantize(agent, 8)
    mods = dict(model.named_modules())
    assert type(model.encoder.layers[0].attn.Wqkv).__name__ == "QuantizedLinear"
    assert all(not type(m).__name__.startswith("Quantized")
               for p, m in mods.items() if p.startswith("act_head"))
    assert model.encoder.layers[0].attn.Wqkv.weight.dtype == mx.uint32


# ---- wiring ---------------------------------------------------------------------------------------

def test_serve_passes_bits_to_the_backend(monkeypatch):
    s = config.load()
    s.bits = 8
    monkeypatch.setattr(support, "_settings", lambda: s)
    built = fake_mlx_serve(monkeypatch)
    assert cli.main(["serve", "--port", "0"]) == 0
    assert built[-1][1]["bits"] == 8


def test_the_in_process_fallback_loads_with_bits(monkeypatch):
    from verdict import client

    monkeypatch.setenv("VERDICT_BITS", "8")
    monkeypatch.setattr(client, "decide", lambda *a, **k: (_ for _ in ()).throw(client.NoServer("x")))
    monkeypatch.setattr(inference, "_ENGINES", {})
    seen = {}

    monkeypatch.setattr(inference, "load", lambda path, **kw: seen.update(kw) or FakeEngine())
    assert cli.main(["ask", "x", "q?", "--lang", "en", "--model", "/tmp/any"]) == 0
    assert seen.get("bits") == 8


def test_a_bad_bits_value_is_a_clean_error_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setenv("VERDICT_BITS", "4")
    assert cli.main(["ask", "x", "q?", "--lang", "en"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("verdict: ") and "§36" in err
