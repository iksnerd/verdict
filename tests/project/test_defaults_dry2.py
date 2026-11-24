"""The shared default constants change no default: a config's digest is a cache key for every stage."""
from verdict.gen.ollama import DEFAULT_MODEL, DEFAULT_URL, OllamaClient
from verdict.label.teacher import DEFAULT_CHEAP_MODELS, DEFAULT_REFERENCE_MODEL
from verdict.pipeline.config import LabelCfg, PipelineConfig


def test_the_digest_of_a_default_config_is_what_it_was_before_the_constants():
    # Computed at HEAD 7eb2d7c, before the defaults moved into constants.
    assert PipelineConfig(name="x").digest() == "a4541416c575d493"


def test_the_constants_hold_the_old_literals():
    assert list(DEFAULT_CHEAP_MODELS) == ["gemini-2.5-flash-lite", "gemini-3.1-flash-lite"]
    assert DEFAULT_REFERENCE_MODEL == "gemini-3.1-pro-preview"
    assert (DEFAULT_MODEL, DEFAULT_URL) == ("gemma3:4b", "http://localhost:11434")
    client = OllamaClient()
    assert (client.model, client.url) == ("gemma3:4b", "http://localhost:11434")


def test_a_default_models_list_is_a_fresh_copy_per_config():
    a, b = LabelCfg(), LabelCfg()
    a.models.append("other")
    assert b.models == list(DEFAULT_CHEAP_MODELS)
