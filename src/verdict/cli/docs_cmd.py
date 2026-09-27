"""`verdict docs`: print the README, guide, API notes or one FINDINGS section.

Named `docs_cmd`, not `docs`, so it never shadows the sibling `verdict.docs` domain module this
command's handler reads from."""
from __future__ import annotations

import argparse
import sys

from . import support

#: `verdict docs` topics; kept equal to `docs.TOPICS` by tests/test_docs_cmd.py.
TOPIC_NAMES = ("readme", "guide", "api", "findings", "pipeline", "routing")


def _docs_cmd(args: argparse.Namespace) -> int:
    """Print a project document, or one FINDINGS section."""
    from .. import docs

    try:
        text = docs.read(args.topic)
        if args.section is not None:
            if args.topic != "findings":
                return support._fail("a section number reads FINDINGS: verdict docs findings N")
            text = docs.section(text, args.section)
    except ValueError as exc:
        return support._fail(exc)
    sys.stdout.write(text)
    return 0
