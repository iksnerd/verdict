import argparse
import os
import random
from collections import Counter
from pathlib import Path

from .domains import MESSAGE_TYPING
from .generate import generate_cases, write_jsonl
from .ollama import DEFAULT_MODEL, DEFAULT_URL, OllamaClient


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic states with a local Ollama model.")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("data/states.jsonl"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    client = OllamaClient(args.model, url=os.environ.get("OLLAMA_URL", DEFAULT_URL))
    cases = generate_cases(MESSAGE_TYPING, client, n=args.n, rng=random.Random(args.seed))
    write_jsonl(cases, args.out)
    print(f"wrote {len(cases)}/{args.n} cases to {args.out}")
    print("by intended type:", dict(sorted(Counter(c.intended_type for c in cases).items())))


main()
