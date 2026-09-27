"""The `verdict` command: typed decisions from the shell.

    verdict ask "<text>" "<question>" [-o A -o B | -l LOW -l HIGH] [--cut N|FILE]
    verdict decide '<json state>' --questions PRESET|JSON|FILE [--jsonl] [--calibration FILE]
    verdict presets [NAME]
    verdict calibrate labelled.jsonl --questions ... --out fit.json
    verdict route "<prompt>"        # the big-or-small switch, kept as the worked example
    verdict serve                   # hold the model so each call costs milliseconds

`verdict --help` and `verdict COMMAND --help` carry the examples, and live in `parser.py` next to
the flags they describe. Nothing here runs anything: every command prints an answer and exits,
and acting on it is the caller's job. Exit status is 2 for a usage error everywhere; `ask --cut`
exits 0 for yes and 1 for no, and `route` 0 for small and 1 for big.
"""
from __future__ import annotations

import argparse
import time

from . import (catalog, decisions, docs_cmd, evaluation, inference, parser, routing, setup,
               support, update)

main = parser.main

#: Re-exported for library callers that import `decide`/`decide_many` directly.
decide = routing.decide
decide_many = routing.decide_many
DEFAULT_MODEL = routing.DEFAULT_MODEL
TOPIC_NAMES = docs_cmd.TOPIC_NAMES

#: Re-exported because tests/test_help_text.py scans `dir(cli)` for every *_HELP/*_EPILOG/
#: DESCRIPTION constant, and other code may still reach for `cli.DECIDE_EPILOG` etc.
DESCRIPTION = parser.DESCRIPTION
EPILOG = parser.EPILOG
ASK_HELP = parser.ASK_HELP
ASK_EPILOG = parser.ASK_EPILOG
DECIDE_HELP = parser.DECIDE_HELP
DECIDE_EPILOG = parser.DECIDE_EPILOG
CALIBRATE_HELP = parser.CALIBRATE_HELP
CALIBRATE_EPILOG = parser.CALIBRATE_EPILOG
PRESETS_EPILOG = parser.PRESETS_EPILOG
ROUTE_EPILOG = parser.ROUTE_EPILOG
SERVE_EPILOG = parser.SERVE_EPILOG
BENCH_EPILOG = parser.BENCH_EPILOG
BENCH_HELP = parser.BENCH_HELP
CASES_EPILOG = parser.CASES_EPILOG
CASES_HELP = parser.CASES_HELP
CONFIG_EPILOG = parser.CONFIG_EPILOG
CONFIG_HELP = parser.CONFIG_HELP
DOCS_EPILOG = parser.DOCS_EPILOG
DOCS_HELP = parser.DOCS_HELP
EXAMPLES_EPILOG = parser.EXAMPLES_EPILOG
EXAMPLES_HELP = parser.EXAMPLES_HELP
INIT_EPILOG = parser.INIT_EPILOG
INIT_HELP = parser.INIT_HELP
PRESETS_HELP = parser.PRESETS_HELP
QUESTIONS_EPILOG = parser.QUESTIONS_EPILOG
QUESTIONS_HELP = parser.QUESTIONS_HELP
RANK_EPILOG = parser.RANK_EPILOG
RANK_HELP = parser.RANK_HELP
UPDATE_EPILOG = parser.UPDATE_EPILOG
UPDATE_HELP = parser.UPDATE_HELP
VALIDATE_EPILOG = parser.VALIDATE_EPILOG
VALIDATE_HELP = parser.VALIDATE_HELP
