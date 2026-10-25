import json
import random

from verdict.gen.domains import HUB_TYPES, MESSAGE_TYPING
from verdict.gen.generate import (
    Case,
    build_prompt,
    clean_output,
    generate_cases,
    is_near_duplicate,
    plan_seeds,
    write_jsonl,
)


class FakeClient:
    """Returns a distinct, plausible message per call so dedup does not eat the run."""

    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str, seed: int) -> str:
        self.calls += 1
        return f"Finished item number {self.calls} for the {seed % 7} module and verified with tests {seed}."


def test_plan_is_balanced_across_types():
    seeds = plan_seeds(MESSAGE_TYPING, n=100, rng=random.Random(0))
    counts = {}
    for s in seeds:
        counts[s.intended_type] = counts.get(s.intended_type, 0) + 1
    assert set(counts) == set(HUB_TYPES)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_plan_is_deterministic():
    a = plan_seeds(MESSAGE_TYPING, n=30, rng=random.Random(7))
    b = plan_seeds(MESSAGE_TYPING, n=30, rng=random.Random(7))
    assert a == b


def test_some_seeds_are_marked_ambiguous():
    seeds = plan_seeds(MESSAGE_TYPING, n=200, rng=random.Random(1))
    frac = sum(s.ambiguous for s in seeds) / len(seeds)
    assert 0.1 < frac < 0.4


def test_prompt_does_not_leak_the_type_word_instruction_only_the_meaning():
    seed = plan_seeds(MESSAGE_TYPING, n=1, rng=random.Random(0))[0]
    prompt = build_prompt(MESSAGE_TYPING, seed)
    assert MESSAGE_TYPING.types[seed.intended_type] in prompt
    assert "Reply with the message text only" in prompt


def test_clean_output_strips_fences_quotes_and_labels():
    assert clean_output('```\nDone with it.\n```') == "Done with it."
    assert clean_output('"Done with it."') == "Done with it."
    assert clean_output("Message: Done with it.") == "Done with it."


def test_clean_output_strips_chatty_preamble_and_curly_quotes():
    assert clean_output("Okay, here's the message:\n\nShipped the fix.") == "Shipped the fix."
    assert clean_output("Okay, here\u2019s the Claude-Code message:\n\n\u201cShipped the fix.\u201d") == "Shipped the fix."
    assert clean_output("Here is the message for the engineering room:\nShipped it.") == "Shipped it."
    # a real message that merely contains a colon is left alone
    assert clean_output("Plan: ship Friday, then review.") == "Plan: ship Friday, then review."


def test_code_prompt_demands_a_snippet():
    seed = plan_seeds(MESSAGE_TYPING, n=10, rng=random.Random(0))
    code_seed = next(s for s in seed if s.intended_type == "code")
    assert "fenced" in build_prompt(MESSAGE_TYPING, code_seed).lower()


def test_near_duplicate_detection():
    a = "Shipped the retry logic for the sync worker and added a regression test"
    assert is_near_duplicate(a, [a.upper()])
    assert is_near_duplicate(a, ["Shipped the retry logic for the sync worker and added a regression test today"])
    assert not is_near_duplicate(a, ["Should we move the queue to Postgres or keep Redis?"])


def test_generate_cases_returns_requested_count_with_state_shape():
    cases = generate_cases(MESSAGE_TYPING, FakeClient(), n=12, rng=random.Random(0))
    assert len(cases) == 12
    c = cases[0]
    assert set(c.state) == {"room_topic", "author", "message"}
    assert c.intended_type in HUB_TYPES
    assert len({x.id for x in cases}) == 12


def test_generate_skips_bad_outputs_and_retries():
    class Flaky(FakeClient):
        def generate(self, prompt, seed):
            self.calls += 1
            if self.calls % 2:
                return ""  # empty -> rejected
            return f"A real message number {self.calls} about module {seed} finishing cleanly."

    cases = generate_cases(MESSAGE_TYPING, Flaky(), n=5, rng=random.Random(0))
    assert len(cases) == 5


def test_generate_gives_up_rather_than_loop_forever():
    class Empty:
        def generate(self, prompt, seed):
            return ""

    cases = generate_cases(MESSAGE_TYPING, Empty(), n=5, rng=random.Random(0), max_attempts_factor=3)
    assert cases == []


def test_write_jsonl_round_trips(tmp_path):
    cases = generate_cases(MESSAGE_TYPING, FakeClient(), n=3, rng=random.Random(0))
    p = tmp_path / "s.jsonl"
    write_jsonl(cases, p)
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert [Case(**r).id for r in rows] == [c.id for c in cases]
