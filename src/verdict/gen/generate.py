"""Turn a Domain into synthetic states by prompting a local model.

The intended type is recorded as metadata, never as a label: a 4B model does not reliably
write the type it was asked for, and the teacher labels the text it actually got."""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .domains import Domain

AMBIGUOUS_RATE = 0.25
MIN_CHARS, MAX_CHARS = 15, 1200  # ~300 tokens: the state budget of the 512-token base model


class Client(Protocol):
    def generate(self, prompt: str, seed: int) -> str: ...


@dataclass(frozen=True)
class Seed:
    intended_type: str
    room_topic: str
    author: str
    length: str
    ambiguous: bool
    confusable: str | None


@dataclass(frozen=True)
class Case:
    id: str
    domain: str
    intended_type: str
    ambiguous: bool
    confusable: str | None
    generator: str
    state: dict
    topic: str = ""
    author: str = ""


def plan_seeds(domain: Domain, n: int, rng: random.Random) -> list[Seed]:
    types = list(domain.types)
    rng.shuffle(types)
    seeds = []
    for i in range(n):
        t = types[i % len(types)]
        ambiguous = rng.random() < AMBIGUOUS_RATE
        seeds.append(
            Seed(
                intended_type=t,
                room_topic=rng.choice(domain.room_topics),
                author=rng.choice(domain.authors),
                length=rng.choice(domain.lengths),
                ambiguous=ambiguous,
                confusable=domain.confusable.get(t) if ambiguous else None,
            )
        )
    return seeds


def build_prompt(domain: Domain, seed: Seed) -> str:
    lines = [
        domain.intro.format(author=seed.author, topic=seed.room_topic),
        f"{domain.lead} {domain.types[seed.intended_type]}.",
        f"Length: {seed.length}.",
        domain.style,
    ]
    if seed.intended_type == "code":
        lines.append("Include a short fenced code or config snippet (2 to 8 lines) as the main content.")
    lines.append("Start directly with the message text, with no preface such as 'Here is the message'.")
    if seed.ambiguous and seed.confusable:
        lines.append(
            f"Make its purpose slightly ambiguous, so a reader could also take it for {domain.ambiguity_lead} "
            f"{domain.types[seed.confusable]}."
        )
    lines.append("Reply with the message text only.")
    return "\n".join(lines)


_FENCE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")
_LABEL = re.compile(r"^(message|reply|response)\s*:\s*", re.I)
_PREAMBLE = re.compile(r"^(okay|ok|sure|here)\b[^\n]{0,80}:\s*\n+", re.I)
_QUOTES = {"\u201c": "\u201d", '"': '"', "'": "'", "\u2018": "\u2019"}


def clean_output(text: str) -> str:
    text = text.strip()
    text = _FENCE.sub("", text).strip()
    text = _PREAMBLE.sub("", text).strip()
    if len(text) >= 2 and _QUOTES.get(text[0]) == text[-1]:
        text = text[1:-1].strip()
    return _LABEL.sub("", text).strip()


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_near_duplicate(text: str, existing: list[str], threshold: float = 0.8) -> bool:
    a = _words(text)
    for other in existing:
        b = _words(other)
        if a and b and len(a & b) / len(a | b) >= threshold:
            return True
    return False


def _acceptable(text: str, accepted: list[str]) -> bool:
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    if re.match(r"(i (cannot|can't|am unable)|sorry|as an ai)", text.lower()):
        return False
    return not is_near_duplicate(text, accepted)


def _state(domain: Domain, seed: Seed, text: str) -> dict:
    if domain.state_style in ("text", "issue"):
        return {domain.state_style: text}
    return {"room_topic": seed.room_topic, "author": seed.author, "message": text}


def generate_cases(
    domain: Domain,
    client: Client,
    n: int,
    rng: random.Random,
    max_attempts_factor: int = 4,
    seen: list[str] | None = None,
    on_case=None,
    workers: int = 1,
) -> list[Case]:
    """Rounds: every seed still without an accepted text gets one call (in parallel when workers > 1),
    then acceptance runs in seed order, so the result does not depend on which call finished first."""
    from concurrent.futures import ThreadPoolExecutor

    generator = getattr(client, "model", "unknown")
    accepted: list[str] = list(seen or [])
    seeds = plan_seeds(domain, n, rng)
    prompts = [build_prompt(domain, s) for s in seeds]
    done: dict[int, Case] = {}
    for _ in range(max_attempts_factor):
        todo = [i for i in range(len(seeds)) if i not in done]
        if not todo:
            break
        call_seeds = {i: rng.randrange(2**31) for i in todo}

        def one(i):
            return clean_output(client.generate(prompts[i], call_seeds[i]))

        # Texts are taken in seed order as they arrive, and each accepted one is handed over at
        # once: a kill mid-round keeps what was already paid for. `map` yields in input order,
        # so the result still does not depend on which call finished first.
        ex = ThreadPoolExecutor(workers) if workers > 1 else None
        texts = ex.map(one, todo) if ex else map(one, todo)
        try:
            for i, text in zip(todo, texts):
                if not _acceptable(text, accepted):
                    continue
                accepted.append(text)
                sd = seeds[i]
                case = Case(
                    id=hashlib.sha1(f"{domain.name}|{text}".encode()).hexdigest()[:12],
                    domain=domain.name, intended_type=sd.intended_type, ambiguous=sd.ambiguous,
                    confusable=sd.confusable, generator=generator, state=_state(domain, sd, text),
                    topic=sd.room_topic, author=sd.author,
                )
                done[i] = case
                if on_case:
                    on_case(case)
        finally:
            if ex:
                ex.shutdown(wait=False, cancel_futures=True)
    return [done[i] for i in sorted(done)]


def write_jsonl(cases: list[Case], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for c in cases:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
