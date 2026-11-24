"""The cosine schedule's length equals the optimizer steps the loop takes (audit 3). No GPU, no Modal."""
import pytest

from pfakes import laya_train  # noqa: F401  (fixture)


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
def test_total_updates_counts_the_steps_the_loop_takes(laya_train, n, micro, accum, epochs):
    assert laya_train.total_updates(n, micro, accum, epochs) == loop_steps(n, micro, accum, epochs)


def test_an_uneven_count_rounds_up_so_the_cosine_never_rises_again(laya_train):
    assert laya_train.total_updates(10, 4, 4, 2) == 2 * 1  # 3 micro-batches -> 1 step per epoch
    assert laya_train.total_updates(10, 4, 2, 3) == 3 * 2  # floor formula said 3 * 1
