import random

import pytest

from verdict.domains import DOMAINS, get_domain
from verdict.gen.generate import build_prompt, generate_cases, plan_seeds
from verdict.pipeline.audit import split_stats


class FakeGen:
    model = "fake"

    def __init__(self):
        self.n = 0

    def generate(self, prompt, seed):
        self.n += 1
        return f"i feel really {['sad', 'happy', 'scared'][seed % 3]} about thing number {self.n} today, honestly {seed}"


def test_registry_has_both_domains_with_their_own_bank_and_preamble():
    assert {"hub_message_typing", "emotion"} <= set(DOMAINS)
    d = get_domain("emotion")
    assert set(d.domain.types) == {"sadness", "joy", "love", "anger", "fear", "surprise"}
    assert list(d.bank) == ["emotion"] and d.bank["emotion"]["type"] == "choice"
    assert d.preamble and "social media" in d.preamble
    assert get_domain("hub_message_typing").preamble is None
    with pytest.raises(KeyError):
        get_domain("nope")


def test_emotion_states_are_text_only_so_they_match_real_inputs():
    d = get_domain("emotion").domain
    cases = generate_cases(d, FakeGen(), n=12, rng=random.Random(0))
    assert cases and all(set(c.state) == {"text"} for c in cases)
    assert all(c.topic and c.author for c in cases)  # split metadata lives beside the state, not in it


def test_emotion_prompt_is_about_a_post_not_an_engineering_ledger():
    d = get_domain("emotion").domain
    seed = plan_seeds(d, 1, random.Random(0))[0]
    p = build_prompt(d, seed)
    assert "engineering" not in p and "ledger" not in p and "first person" in p.lower()
    assert d.types[seed.intended_type] in p


def test_ledger_domain_is_unchanged():
    d = get_domain("hub_message_typing").domain
    cases = generate_cases(d, FakeGen(), n=6, rng=random.Random(0))
    assert all(set(c.state) == {"room_topic", "author", "message"} for c in cases)


def test_split_stats_reads_topic_from_the_case_not_the_state():
    d = get_domain("emotion").domain
    rows = [c.__dict__ for c in generate_cases(d, FakeGen(), n=8, rng=random.Random(0))]
    split = {"train": [r["id"] for r in rows[:6]], "holdout": [rows[6]["id"]], "test": [rows[7]["id"]]}
    st = split_stats(rows, split)
    assert st["room_topic"] and st["author"]


def test_parallel_generation_matches_sequential_content_count_and_is_deterministic():
    d = get_domain("emotion").domain
    a = generate_cases(d, FakeGen(), n=20, rng=random.Random(3), workers=1)
    b = generate_cases(d, FakeGen(), n=20, rng=random.Random(3), workers=8)
    assert len(a) == len(b) == 20
    assert [c.intended_type for c in a] == [c.intended_type for c in b]
