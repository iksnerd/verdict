"""Shared loaders and doubles for tests/pipeline. Invented data only."""
import importlib.util
import json
import sys
from unittest import mock

import pytest

from verdict.gen.domains import MESSAGE_TYPING
from verdict.label.bank import BANK
from verdict.pipeline.stages import gen_stage, label_stage, split_stage

from fakes import GOOD_LABELS, REPO_ROOT


def load_script(name, rel, stub_modal=False):
    """Import a script that is not a package module, under its own module name.

    `stub_modal` swaps `modal` for a MagicMock whose decorators return the function unchanged, so
    training/cloud/train_modal.py loads on a machine without Modal.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    if not stub_modal:
        spec.loader.exec_module(mod)
        return mod
    fake = mock.MagicMock()
    fake.App.return_value.function.side_effect = lambda **kw: (lambda f: f)
    fake.App.return_value.local_entrypoint.side_effect = lambda *a, **kw: (lambda f: f)
    with mock.patch.dict(sys.modules, {"modal": fake}):
        spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def laya_train(request):
    pytest.importorskip("torch")
    pytest.importorskip("laya")
    pytest.importorskip("transformers")
    return load_script(f"laya_train_under_test_{request.module.__name__}", "training/cloud/laya_train.py")


@pytest.fixture
def train_modal(request):
    return load_script(f"train_modal_under_test_{request.module.__name__}_{request.node.name}",
                       "training/cloud/train_modal.py", stub_modal=True)


# --- label transports and a built run ---------------------------------------------------------------


class GoodTransport:
    """Counts calls and answers every one with GOOD_LABELS at 1000 in, 500 out."""

    def __init__(self):
        self.calls = 0

    def __call__(self, model, prompt):
        self.calls += 1
        return json.dumps(GOOD_LABELS), {"promptTokenCount": 1000, "candidatesTokenCount": 500}


def reply(kind_action):
    rest = (1 - kind_action) / 6
    return {"kind": {"action": kind_action, "synthesis": rest, "decision": rest, "thought": rest,
                     "draft": rest, "note": rest, "other": rest},
            "records_decision": {"true": 0.1, "false": 0.9}, "reports_shipped_work": {"true": 0.9, "false": 0.1},
            "leaves_open_question": {"true": 0.1, "false": 0.9}, "is_proposal": {"true": 0.1, "false": 0.9}}


class AlternatingTransport:
    """Per model, repeat 0 answers action 0.8 and repeat 1 answers 0.2 (a state's repeats are consecutive)."""

    def __init__(self):
        self.calls = {}

    def __call__(self, model, prompt):
        n = self.calls[model] = self.calls.get(model, 0) + 1
        return json.dumps(reply(0.8 if (n - 1) % 2 == 0 else 0.2)), {"promptTokenCount": 1000, "candidatesTokenCount": 500}


def build_run(cfg, run_dir, gen, transport, bank=BANK, split=True):
    """gen, label and (unless split=False) split a run in `run_dir`; returns `cfg`."""
    gen_stage(cfg, run_dir, gen, MESSAGE_TYPING)
    label_stage(cfg, run_dir, transport, bank)
    if split:
        split_stage(cfg, run_dir)
    return cfg
