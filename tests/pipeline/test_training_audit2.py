"""Training resume identity, shuffle order, calibration wording, Modal commits (audit 2). No GPU, no Modal."""
import subprocess
import time
from pathlib import Path

import pytest

from pfakes import laya_train, train_modal  # noqa: F401  (fixtures)


def test_an_epochs_order_depends_only_on_seed_and_epoch(laya_train):
    assert laya_train.epoch_order(50, 2) == laya_train.epoch_order(50, 2)  # a resumed run at epoch 2 sees what a whole run saw
    assert laya_train.epoch_order(50, 2) != laya_train.epoch_order(50, 3)
    assert sorted(laya_train.epoch_order(50, 1)) == list(range(50))


def test_resume_mismatch_names_each_key_that_differs_or_is_missing(laya_train):
    was = {"items_sha256": "a", "precision": "fp16"}  # a checkpoint from before the new keys
    now = {**was, "epochs": 4, "lr_head": 1e-4}
    lines = laya_train.resume_mismatch(was, now)
    assert any("epochs" in l and "<not recorded>" in l for l in lines)
    assert laya_train.resume_mismatch(now, now) == []
    assert laya_train.resume_mismatch({**now, "epochs": 5}, now) == ["epochs: checkpoint 5, this run 4"]


def test_the_calibration_line_does_not_claim_held_out_fitting_for_an_empty_holdout(laya_train):
    note = laya_train.calibration_note(0, {})
    assert "HELD-OUT" not in note and "0 items" in note and "fallback" in note
    assert "fitted on 40 HELD-OUT" in laya_train.calibration_note(40, {0: 20, 1: 10, 2: 10})
    assert "score (3)" in laya_train.calibration_note(23, {0: 10, 1: 3, 2: 10})


def test_json_is_written_atomically(laya_train, tmp_path):
    p = str(tmp_path / "rl_agent_config.json")
    laya_train.write_json_atomic(p, {"a": 1})
    assert Path(p).read_text().count("\n") >= 1 and not list(tmp_path.glob("*.tmp"))


def test_modal_train_commits_the_volume_when_a_checkpoint_lands(train_modal, tmp_path):
    tm = train_modal
    ckpt = tmp_path / "ckpt.pt"
    script = tmp_path / "fake_train.py"
    script.write_text(f"import time\nopen({str(ckpt)!r}, 'w').write('1')\ntime.sleep(0.6)\n")
    commits = []
    tm._run_committing([str(script)], str(ckpt), commit=lambda: commits.append(time.time()), poll=0.2)
    assert len(commits) >= 1  # committed while the script was still running, not only after it returned


def test_modal_train_still_fails_when_the_script_fails(train_modal, tmp_path):
    tm = train_modal
    script = tmp_path / "bad.py"
    script.write_text("raise SystemExit(3)\n")
    with pytest.raises(subprocess.CalledProcessError):
        tm._run_committing([str(script)], str(tmp_path / "ckpt.pt"), commit=lambda: None, poll=0.1)
