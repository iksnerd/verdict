"""Single-GPU RLCD fine-tune of Laya. Adapted from the fine-tune notebook in NandhaKishorM/laya
(Apache-2.0); changes from the upstream script:

  * single GPU, no DDP (an A10G holds 421M parameters with gradient checkpointing);
  * `--precision {fp16,bf16,tf32}` so the speed/numerics tradeoff is measured, not assumed;
  * `--max-steps` probe mode that times micro-batches and reports peak memory, then exits;
  * a checkpoint each epoch and `--resume`, so a preempted run loses at most one epoch;
  * calibration temperatures are fitted on HELD-OUT items (`--calib-items`), not on the training
    items as upstream does, which would make the model look better calibrated than it is;
  * max_len / head_max_len are left as the base checkpoint's, matching how the items were built.
"""
import argparse
import contextlib
import json
import os
import random
import time

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.common import build_model, proper_reward


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
            "qtype": torch.tensor([it["qtype"] for it in items]), "label": torch.tensor([it["label"] for it in items])}


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--calib-items", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-name", default="verdict-laya")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--precision", choices=["fp16", "bf16", "tf32"], default="fp16")
    ap.add_argument("--max-steps", type=int, default=0, help="probe: time this many micro-batches, report, exit")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device: refusing to train on CPU"
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  precision {a.precision}", flush=True)
    if a.precision == "tf32":
        torch.set_float32_matmul_precision("high")

    cfg = json.load(open(os.path.join(a.model_dir, "rl_agent_config.json")))
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096
    tok = AutoTokenizer.from_pretrained(os.path.join(a.model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(a.model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(a.model_dir, "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device).train()

    items = torch.load(a.items, weights_only=False)
    sigma_start, sigma_end = 0.4, 0.1
    enc = [p for n, p in model.named_parameters() if "encoder." in n]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW([{"params": enc, "lr": a.lr_encoder}, {"params": head, "lr": a.lr_head}], weight_decay=0.01)
    total_updates = max(1, (len(items) // (a.micro_batch * a.grad_accum)) * a.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=a.precision == "fp16")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(a.precision)

    def amp():
        return torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()

    os.makedirs(a.out, exist_ok=True)
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    start_epoch = 0
    if a.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"]
        print(f"resumed from epoch {start_epoch}", flush=True)

    metrics = open(os.path.join(a.out, "metrics.jsonl"), "a")
    print(f"{len(items)} items | micro-batch {a.micro_batch} x accum {a.grad_accum} | {a.epochs} epochs", flush=True)
    t0, step, timed, t_timed = time.time(), 0, 0, 0.0
    for epoch in range(start_epoch, a.epochs):
        random.seed(42 + epoch)
        random.shuffle(items)
        sigma = sigma_start + (sigma_end - sigma_start) * epoch / max(1, a.epochs - 1)
        optimizer.zero_grad(set_to_none=True)
        accum, ep_loss, nb = 0, 0.0, 0
        for b in range(0, len(items), a.micro_batch):
            chunk = items[b : b + a.micro_batch]
            batch = collate_train_batch(chunk, tok.pad_token_id)
            torch.cuda.synchronize()
            ts = time.perf_counter()
            with amp():
                logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                    batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device))
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            eps = torch.randn((a.group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / a.grad_accum + 0.0 * act.sum()
            scaler.scale(loss).backward()
            accum += 1
            if accum % a.grad_accum == 0 or (b + a.micro_batch) >= len(items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.perf_counter() - ts
            step += 1
            if step > 3:  # first micro-batches carry allocator and kernel warmup
                timed += 1
                t_timed += dt
            ep_loss += loss.item() * a.grad_accum
            nb += 1
            if step % 10 == 0:
                rec = {"step": step, "epoch": epoch + 1, "loss": round(loss.item() * a.grad_accum, 4), "reward": round(r.mean().item(), 4),
                       "lr": scheduler.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
                metrics.write(json.dumps(rec) + "\n")
                metrics.flush()
                print(rec, flush=True)
            if a.max_steps and step >= a.max_steps:
                ms = 1000 * t_timed / max(1, timed)
                mbs_per_epoch = -(-len(items) // a.micro_batch)
                res = {"precision": a.precision, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                       "micro_batches_timed": timed, "ms_per_micro_batch": round(ms, 1),
                       "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "items": len(items),
                       "mean_tokens_per_item": round(sum(len(i["ids"]) for i in items) / len(items), 1),
                       "est_minutes_per_epoch": round(ms * mbs_per_epoch / 60000, 2), "final_loss": round(loss.item() * a.grad_accum, 4)}
                json.dump(res, open(os.path.join(a.out, f"probe_{a.precision}.json"), "w"), indent=1)
                print("PROBE " + json.dumps(res), flush=True)
                return
        print(f"=== epoch {epoch + 1}/{a.epochs} done in {time.time() - t0:.0f}s | avg loss {ep_loss / max(1, nb):.4f} ===", flush=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch + 1}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    temps = [1.2, 1.2, 1.2]
    if a.calib_items:
        calib = torch.load(a.calib_items, weights_only=False)
        model.eval()
        preds = []
        with torch.no_grad():
            for c in range(0, len(calib), 16):
                chunk = calib[c : c + 16]
                cb = collate_train_batch(chunk, tok.pad_token_id)
                with amp():
                    ls, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device), cb["marker_pos"].to(device),
                                  cb["marker_mask"].to(device), cb["qtype"].to(device))
                arr = ls.float().cpu().numpy()
                for i, it in enumerate(chunk):
                    preds.append((it["qtype"], arr[i, : len(it["markers"])], it["target"]))
        for qt in range(3):
            sel = [(z, t) for qtp, z, t in preds if qtp == qt]
            if sel:
                temps[qt] = fit_one_temp(sel)
        print("temperatures fitted on HELD-OUT items (choice, score, noul):", [round(t, 3) for t in temps], flush=True)
    else:
        print("WARNING: no --calib-items, temperatures left at the 1.2 fallback; do not trust the probabilities", flush=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}, os.path.join(a.out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(a.out, "encoder"))
    tok.save_pretrained(os.path.join(a.out, "tokenizer"))
    cfg.update({"fine_tuned": True, "model_name": a.model_name, "temperature": temps})
    json.dump(cfg, open(os.path.join(a.out, "rl_agent_config.json"), "w"), indent=2)
    print(f"saved to {a.out}", flush=True)


if __name__ == "__main__":
    main()
