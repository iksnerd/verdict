"""Modal commit resilience and the upload freshness check (audit 4). No GPU, no Modal."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, rel="training/cloud/train_modal.py"):
    fake = mock.MagicMock()
    fake.App.return_value.function.side_effect = lambda **kw: (lambda f: f)
    fake.App.return_value.local_entrypoint.side_effect = lambda *a, **kw: (lambda f: f)
    with mock.patch.dict(sys.modules, {"modal": fake}):
        spec = importlib.util.spec_from_file_location(name, ROOT / rel)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def test_a_failed_commit_does_not_kill_the_training_run(tmp_path):
    tm = load("tm4a")
    ckpt, done = tmp_path / "ckpt.pt", tmp_path / "done"
    script = tmp_path / "t.py"
    script.write_text(f"import time\nopen({str(ckpt)!r},'w').write('1')\ntime.sleep(0.8)\nopen({str(done)!r},'w').write('x')\n")
    calls = []

    def flaky():
        calls.append(time.time())
        if len(calls) == 1:
            raise RuntimeError("transient volume error")

    tm._run_committing([str(script)], str(ckpt), commit=flaky, poll=0.2)
    assert done.exists()  # the script ran to the end
    assert len(calls) >= 2  # and the commit was retried


def test_a_failing_script_still_gets_a_final_commit(tmp_path):
    tm = load("tm4b")
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_text("old")
    script = tmp_path / "bad.py"
    script.write_text("raise SystemExit(3)\n")
    calls = []
    with pytest.raises(subprocess.CalledProcessError):
        tm._run_committing([str(script)], str(ckpt), commit=lambda: calls.append(1), poll=0.1)
    assert calls


def test_a_failing_final_commit_does_not_hide_the_scripts_failure(tmp_path):
    tm = load("tm4c")
    script = tmp_path / "bad.py"
    script.write_text("raise SystemExit(3)\n")

    def boom():
        raise RuntimeError("volume down")

    with pytest.raises(subprocess.CalledProcessError):
        tm._run_committing([str(script)], str(tmp_path / "ckpt.pt"), commit=boom, poll=0.1)


def _export(tmp_path, tamper=False):
    run = tmp_path / "runs" / "r"
    (run / "export").mkdir(parents=True)
    outs = {}
    for n in ("train", "holdout", "test"):
        p = run / "export" / f"{n}.jsonl"
        p.write_text(f"{n}\n")
        outs[f"export/{n}.jsonl"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (run / "export.manifest.json").write_text(json.dumps({"outputs": outs}))
    if tamper:
        (run / "export" / "train.jsonl").write_text("changed\n")
    return run


def test_upload_refuses_an_export_that_no_longer_matches_its_manifest(tmp_path):
    tm = load("tm4d")
    with pytest.raises(SystemExit, match=r"train\.jsonl.*(stale|no longer)"):
        tm._require_fresh_export(_export(tmp_path, tamper=True))


def test_upload_accepts_a_matching_export_and_one_without_a_manifest(tmp_path):
    tm = load("tm4e")
    run = _export(tmp_path)
    tm._require_fresh_export(run)
    (run / "export.manifest.json").unlink()
    tm._require_fresh_export(run)


@pytest.fixture(scope="module")
def lt():
    pytest.importorskip("torch")
    pytest.importorskip("laya")
    pytest.importorskip("transformers")
    return load("laya_train_audit4", "training/cloud/laya_train.py")


def test_a_checkpoint_from_before_the_new_keys_still_resumes(lt):
    now = {"items_sha256": "a", "epochs": 4, "group_size": 4, "sigma_start": 0.4, "sigma_end": 0.1, "model_name": "m"}
    old = {"items_sha256": "a", "epochs": 4}
    assert lt.resume_identity(old, now) == old


def test_changing_a_recorded_new_key_refuses_a_resume(lt):
    now = {"items_sha256": "a", "group_size": 4, "sigma_start": 0.4, "model_name": "m"}
    was = {**now, "group_size": 8}
    assert lt.resume_identity(was, now) != was
    assert lt.resume_mismatch(was, lt.resume_identity(was, now)) == ["group_size: checkpoint 8, this run 4"]


def test_a_new_run_records_the_new_keys(lt):
    assert {"group_size", "sigma_start", "sigma_end", "model_name"} <= set(lt.NEW_MADE_FROM_KEYS)
