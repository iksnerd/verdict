"""The project's documents, readable from an installed tool.

The wheel carries them under `verdict/_docs/` (`force-include` in pyproject.toml); a checkout reads
them in place, so `uv run verdict docs` shows what is being edited.
"""
from __future__ import annotations

import re
from pathlib import Path

#: Topic -> path in the repository. The wheel packs each at `verdict/_docs/<file name>`.
TOPICS = {
    "readme": "README.md",
    "guide": "docs/guide.md",
    "api": "docs/api.md",
    "findings": "docs/FINDINGS.md",
    "pipeline": "docs/pipeline.md",
    "routing": "docs/routing.md",
}

_PACKED = Path(__file__).with_name("_docs")
_CHECKOUT = Path(__file__).resolve().parents[2]


def read(topic: str) -> str:
    source = TOPICS[topic]
    for path in (_PACKED / Path(source).name, _CHECKOUT / source):
        if path.is_file():
            return path.read_text()
    raise FileNotFoundError(f"{source} is not in this install; see "
                            "https://github.com/iksnerd/verdict")


def section(text: str, number: int) -> str:
    """One `## N.` section of FINDINGS, heading included."""
    starts = [(int(m.group(1)), m.start()) for m in re.finditer(r"^## (\d+)\.", text, re.M)]
    for i, (n, start) in enumerate(starts):
        if n == number:
            end = starts[i + 1][1] if i + 1 < len(starts) else len(text)
            return text[start:end].rstrip("\n") + "\n"
    have = [n for n, _ in starts]
    raise ValueError(f"no §{number} in FINDINGS; it has §{min(have)} to §{max(have)}")
