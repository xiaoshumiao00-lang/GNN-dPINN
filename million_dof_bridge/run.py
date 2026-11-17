#!/usr/bin/env python3
"""One-command entry point for the million-DOF GNN-dPINN bridge case."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

from bridge_problem import BridgeConfig, BridgeProblem
from optimize import optimize_case
from plotting import generate_all_outputs

ROOT = Path(__file__).resolve().parent


class Tee:
    def __init__(self, path: Path):
        self.handle = path.open("w", encoding="utf-8", buffering=1)
        self.streams = [sys.stdout, sys.stderr]

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        self.handle.write(text)
        self.handle.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()
        self.handle.flush()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config_million.json"))
    parser.add_argument("--smoke-test", action="store_true", help="Run the small pipeline test; diagnostic output only")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint_latest.pt and start from scratch")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None)
    parser.add_argument("--check-only", action="store_true", help="Build mesh/operators and report memory-safe configuration only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = ROOT / "config_smoke.json" if args.smoke_test else Path(args.config)
    config = BridgeConfig.load(config_path)
    if args.no_resume:
        config.resume = False
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.device:
        config.device = args.device
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(output_dir / "run.log")
    sys.stdout, sys.stderr = tee, tee
    started = time.perf_counter()
    try:
        print("GNN-dPINN million-DOF bridge runner", flush=True)
        print("Configuration:", config_path, flush=True)
        print("PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available(), flush=True)
        config.validate()
        if config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(config.device)
        if device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            print(f"GPU: {torch.cuda.get_device_name(device)}; free={free_bytes/1e9:.2f} GB; total={total_bytes/1e9:.2f} GB", flush=True)
            if config.nelx * config.nely > 500_000 and free_bytes < 12e9:
                print("WARNING: less than 12 GB GPU memory is currently free; close other GPU jobs before production.", flush=True)
        os.environ.setdefault("PYTHONHASHSEED", str(config.seed))
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)
        torch.set_float32_matmul_precision("high" if config.allow_tf32 else "highest")

        print("Building geometry, mesh, and sparse operators...", flush=True)
        problem = BridgeProblem(config, device)
        summary = problem.summary()
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        (output_dir / "mesh_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if not args.smoke_test and problem.num_dofs < 1_000_000:
            raise ValueError(f"Production case has only {problem.num_dofs:,} DOFs; at least 1,000,000 is required.")
        if args.check_only:
            print("Preflight check passed. No optimization was run.", flush=True)
            return 0
        metrics, history = optimize_case(config, problem, output_dir)
        metrics["end_to_end_time_s"] = time.perf_counter() - started
        (output_dir / "results_summary.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        generate_all_outputs(problem, metrics, history, output_dir)
        print("All outputs and validation checks completed.", flush=True)
        print("Results directory:", output_dir, flush=True)
        if args.smoke_test:
            print("SMOKE TEST ONLY: reduced mesh; do not report these values as production results.", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        print("Run failed. See run.log; checkpoint_error.pt is saved when failure occurs during training.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
