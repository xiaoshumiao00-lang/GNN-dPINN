#!/usr/bin/env python3
"""Single-epoch forward/backward probe on the production mesh, for memory and timing only."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch

from bridge_problem import BridgeConfig, BridgeProblem
from model import GNNdPINN
from physics import evaluate_physics

ROOT = Path(__file__).resolve().parent


def main():
    config = BridgeConfig.load(ROOT / "config_million.json")
    config.validate()
    device = torch.device("cuda")
    problem = BridgeProblem(config, device)
    model = GNNdPINN(problem.a_hat, problem.bc_fixed, problem.bc_symmetry,
                     problem.lx, problem.ly, config.gradient_checkpointing).to(device)
    rho = problem.initial_density()
    nodal = problem.nodal_density(rho)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = evaluate_physics(model, problem, rho, nodal, compute_stress=False)
    result.potential_energy.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    report = {
        "probe": "one forward and backward epoch only; not an optimization result",
        "mesh": problem.summary(),
        "loss": float(result.potential_energy.detach()),
        "elapsed_s": elapsed,
        "peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "gpu": torch.cuda.get_device_name(device),
        "gradient_checkpointing": config.gradient_checkpointing,
        "allow_tf32": config.allow_tf32,
    }
    output = ROOT / "results" / "memory_probe"
    output.mkdir(parents=True, exist_ok=True)
    (output / "memory_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
