"""    uv run python -m verdict.pipeline --config pipeline.json --run-dir runs/pilot --stage all

Stages: gen (local Ollama), label (Gemini teachers, cost-capped), split. Rerun any stage to resume."""
import argparse
import sys
from pathlib import Path

from ..domains import get_domain
from ..gen.gemini import GeminiGenClient
from ..gen.ollama import OllamaClient
from ..label.teacher import gemini_transport, load_key
from .config import PipelineConfig
from .export import export_stage
from .stages import gen_stage, label_stage, split_stage


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--stage", choices=["gen", "label", "split", "export", "all"], default="all")
    ap.add_argument("--key-file", default=None, help="env file to read GOOGLE_API_KEY from (overrides the config)")
    a = ap.parse_args()
    cfg = PipelineConfig.load(a.config)
    spec = get_domain(cfg.domain)
    stages = ["gen", "label", "split", "export"] if a.stage == "all" else [a.stage]
    print(f"run {cfg.name} (config {cfg.digest()}) in {a.run_dir}: {stages}")
    if "gen" in stages:
        if cfg.gen.provider == "gemini":
            client = GeminiGenClient(cfg.gen.model, load_key(a.key_file or cfg.label.key_file))
        else:
            client = OllamaClient(cfg.gen.model)
        n = gen_stage(cfg, a.run_dir, client, spec.domain)
        print(f"gen: {n}/{cfg.gen.n} states" + (f"  (generator spend ${client.cost_usd():.4f})" if hasattr(client, "cost_usd") else ""))
    failed = 0
    if "label" in stages:
        counts = label_stage(cfg, a.run_dir, gemini_transport(load_key(a.key_file or cfg.label.key_file)), spec.bank, spec.preamble)
        print("label:", counts)
        failed = counts.get("failures_this_run", 0)
    if "split" in stages:
        s = split_stage(cfg, a.run_dir)
        print(f"split: train {len(s['train'])}  holdout {len(s['holdout'])}  test {len(s['test'])}  (seed used {s['seed_used']})")
    if "export" in stages:
        print("export:", export_stage(cfg, a.run_dir, spec.bank))
    if failed:  # the later stages ran on what was labelled; the exit code says the run is incomplete
        print(f"label: {failed} labels failed in this run; rerun to fill them", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
