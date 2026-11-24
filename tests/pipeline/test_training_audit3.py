"""The cosine schedule's length equals the optimizer steps the loop takes (audit 3). No GPU, no Modal."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lt():
    pytest.importorskip("torch")
    pytest.importorskip("laya")
    pytest.importorskip("transformers")
    return load("laya_train_under_test3", "training/cloud/laya_train.py")


def loop_steps(n, micro, accum, epochs):
    """Replays the training loop's step condition."""
    steps = 0
    for _ in range(epochs):
        a = 0
        for b in range(0, n, micro):
            a += 1
            if a % accum == 0 or (b + micro) >= n:
                steps += 1
    return steps


@pytest.mark.parametrize("n,micro,accum,epochs", [
    (64, 4, 4, 3),    # divides evenly
    (10, 4, 2, 3),    # 3 micro-batches, last accumulation partial
    (10, 4, 4, 2),    # one partial accumulation per epoch
    (7, 8, 4, 2),     # fewer items than one micro-batch
    (100, 3, 5, 4),
])
def test_total_updates_counts_the_steps_the_loop_takes(lt, n, micro, accum, epochs):
    assert lt.total_updates(n, micro, accum, epochs) == loop_steps(n, micro, accum, epochs)


def test_an_uneven_count_rounds_up_so_the_cosine_never_rises_again(lt):
    assert lt.total_updates(10, 4, 4, 2) == 2 * 1  # 3 micro-batches -> 1 step per epoch
    assert lt.total_updates(10, 4, 2, 3) == 3 * 2  # floor formula said 3 * 1
