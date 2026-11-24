"""Fine-tune Laya on a Modal A10G.

Patterns carried over from an earlier Modal project: pip layers before `add_local_*` (the reverse order goes
stale), data and checkpoints on a persistent Volume, a hard function timeout as a cost ceiling,
and `modal app stop <id> --yes` for a run that crash-loops. A run that fails does not stop itself.

    uv run --extra cloud modal run training/cloud/train_modal.py::probe_main           # the probe (~minutes, pennies)
    uv run --extra cloud --extra laya modal run training/cloud/train_modal.py::upload --run v1   # build items, put on the volume
    uv run --extra cloud modal run training/cloud/train_modal.py::train --run v1       # the real run (~hours, dollars)

`probe` times 30 micro-batches per precision arm and reports ms/step and peak memory, so the real
run is budgeted from a measurement instead of a guess (an earlier project measured TF32 up and
bf16 down on one loop and both up on another; nothing transfers without its own probe).
"""
import json
import os
import subprocess
import sys

import modal

app = modal.App("verdict-train")
vol = modal.Volume.from_name("verdict-vol", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.12.1", "transformers==5.17.0", "laya==0.3.4", "safetensors", "huggingface_hub", "numpy")
    .env({"HF_HOME": "/data/hf"})
    .add_local_file("training/cloud/laya_train.py", "/root/laya_train.py")
)

BASE = "convaiinnovations/laya"
#: The base revision every fine-tune and comparison so far used. Unpinned, a download would train
#: on whatever upstream published last, unmeasured.
BASE_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"


def _model_dir() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(BASE, revision=BASE_REVISION)


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-u", *args], check=True)


def _run_committing(args: list[str], ckpt: str, commit=None, poll: float = 60.0) -> None:
    """`_run`, but the volume is committed whenever `ckpt` has a new version while the script runs.
    `train` committed only after the script returned, so a timeout kill lost every epoch's
    checkpoint with it. The script replaces `ckpt` atomically, so a commit sees a whole file."""
    commit = commit or vol.commit
    print("+", " ".join(args), flush=True)
    proc = subprocess.Popen([sys.executable, "-u", *args])
    seen = os.path.getmtime(ckpt) if os.path.exists(ckpt) else None
    def try_commit() -> bool:
        # A transient volume error must never kill a multi-hour run: log it, keep going.
        try:
            commit()
            return True
        except Exception as e:
            print(f"commit failed ({type(e).__name__}: {e}); retrying at the next poll", flush=True)
            return False

    try:
        while True:
            try:
                proc.wait(timeout=poll)
                break
            except subprocess.TimeoutExpired:
                pass
            now = os.path.getmtime(ckpt) if os.path.exists(ckpt) else None
            if now != seen and try_commit():  # `seen` stays put on failure, so the next poll retries
                seen = now
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    if proc.returncode:
        try_commit()  # keep the partial outputs and metrics of a failed run
        raise subprocess.CalledProcessError(proc.returncode, args)
    if os.path.exists(ckpt) and os.path.getmtime(ckpt) != seen:
        commit()  # the last epoch's checkpoint can land between the last poll and the exit


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=30 * 60)
def probe(precisions: str = "fp16,bf16,tf32", steps: int = 30) -> list[dict]:
    md = _model_dir()
    vol.commit()  # the model download is the slow part; keep it for the next call
    out = []
    for p in precisions.split(","):
        _run(["/root/laya_train.py", "--model-dir", md, "--items", "/data/verdict/probe_items.pt",
              "--out", f"/data/verdict/probe_{p}", "--precision", p, "--max-steps", str(steps)])
        out.append(json.load(open(f"/data/verdict/probe_{p}/probe_{p}.json")))
    vol.commit()
    return out


@app.function(image=image, gpu="A10G", volumes={"/data": vol}, timeout=60 * 60)
def train(run: str = "v1", precision: str = "fp16", epochs: int = 4, resume: bool = True) -> str:
    md = _model_dir()
    d = f"/data/verdict/{run}"
    args = ["/root/laya_train.py", "--model-dir", md, "--items", f"{d}/train_items.pt", "--calib-items", f"{d}/holdout_items.pt",
            "--out", f"{d}/model", "--precision", precision, "--epochs", str(epochs), "--model-name", f"verdict-{run}"]
    if resume:
        args.append("--resume")
    _run_committing(args, f"{d}/model/ckpt.pt")
    vol.commit()
    return f"{d}/model"


@app.local_entrypoint()
def probe_main(items: str = "runs/dev/probe_items.pt", precisions: str = "fp16,bf16,tf32", steps: int = 30):
    with vol.batch_upload(force=True) as b:
        b.put_file(items, "/verdict/probe_items.pt")
    for r in probe.remote(precisions, steps):
        print(json.dumps(r))


def _require_fresh_export(run_dir) -> None:
    """Refuse train/holdout files that no longer match what the export stage recorded. A run with
    no export manifest (made by hand) is not checked. Plain json and hashlib: no verdict import."""
    import hashlib
    from pathlib import Path

    run_dir = Path(run_dir)
    mpath = run_dir / "export.manifest.json"
    if not mpath.exists():
        return
    outputs = json.loads(mpath.read_text()).get("outputs", {})
    for name in ("train", "holdout"):
        rel = f"export/{name}.jsonl"
        path, recorded = run_dir / rel, outputs.get(rel)
        if recorded is None or not path.exists():
            continue
        now = hashlib.sha256(path.read_bytes()).hexdigest()
        if now != recorded:
            raise SystemExit(f"{path} no longer matches the copy the export stage made (recorded "
                             f"{recorded[:12]}, now {now[:12]}): rerun `python -m verdict.pipeline "
                             f"--run-dir {run_dir} --stage export` before uploading")


@app.local_entrypoint()
def upload(run: str = "v1", model_dir: str = ""):
    """Build training items from runs/<run>/export and put them on the volume for `train`.
    The test split is never uploaded: train and holdout only."""
    from pathlib import Path

    import torch
    from huggingface_hub import snapshot_download

    from verdict.train.items import build_items

    md = Path(model_dir or snapshot_download(BASE, revision=BASE_REVISION,
                                                       allow_patterns=["tokenizer/*", "rl_agent_config.json", "encoder/*"]))
    _require_fresh_export(Path(f"runs/{run}"))
    out = Path(f"runs/{run}/items")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "holdout"):
        items, dropped = build_items(Path(f"runs/{run}/export/{name}.jsonl"), md)
        torch.save(items, out / f"{name}_items.pt")
        print(f"{name}: {len(items)} items ({dropped} dropped for not fitting the option budget)")
    with vol.batch_upload(force=True) as b:
        for name in ("train", "holdout"):
            b.put_file(str(out / f"{name}_items.pt"), f"/verdict/{run}/{name}_items.pt")
    print(f"uploaded to verdict-vol:/verdict/{run}/")
