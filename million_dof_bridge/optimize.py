"""Topology optimization loop, OC update, checkpoints, and machine-readable outputs."""
from __future__ import annotations

import csv
import copy
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import Any

import numpy as np
import torch

from bridge_problem import BridgeConfig, BridgeProblem
from model import GNNdPINN
from physics import evaluate_physics, simp_sensitivity


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / 1e9)


def oc_update(problem: BridgeProblem, rho: torch.Tensor, dc: torch.Tensor) -> torch.Tensor:
    c = problem.config
    lower, upper = 0.0, 1e9
    rho_new = rho
    for _ in range(100):
        if upper - lower <= 1e-4:
            break
        midpoint = 0.5 * (lower + upper)
        factor = torch.sqrt(torch.abs(-dc / midpoint))
        candidate = rho * factor
        rho_new = torch.clamp(candidate, min=rho - c.move, max=rho + c.move)
        rho_new = torch.clamp(rho_new, min=0.001, max=1.0)
        rho_new[problem.fixed_elements] = 1.0
        rho_new[problem.void_elements] = 0.0
        if float(torch.sum(rho_new)) - problem.target_volume > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    return rho_new


def _case_signature(config: BridgeConfig) -> dict[str, Any]:
    keys = [
        "nelx", "nely", "lx", "ly", "volfrac", "penalty",
        "filter_radius_physical", "E0", "Emin", "nu", "total_load", "seed",
    ]
    return {key: getattr(config, key) for key in keys}


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    if not history:
        return
    keys = list(history[0])
    with (output_dir / "metrics_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    for key in ["compliance", "volume_fraction", "density_change", "loss", "iteration_time_s"]:
        np.save(output_dir / f"history_{key}.npy", np.asarray([row[key] for row in history]))


def _hardware(device: torch.device) -> dict[str, Any]:
    info = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "gpu": "CPU",
        "cuda": None,
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda"] = torch.version.cuda
    return info


def _capture_rng(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _train_epochs(model, optimizer, problem: BridgeProblem, rho: torch.Tensor,
                  nodal_rho: torch.Tensor, epochs: int, config: BridgeConfig) -> tuple[float, float]:
    """Train for a fixed budget, restoring the state with the smallest equilibrium residual."""
    best_score = float("inf")
    best_model = None
    best_optimizer = None
    free_ux = ~(problem.bc_fixed | problem.bc_symmetry)
    free_uy = ~problem.bc_fixed
    if config.restore_best_training_state and config.training_selection == "residual":
        model.eval()
        with torch.enable_grad():
            initial = evaluate_physics(model, problem, rho, nodal_rho, compute_stress=False)
            initial_residual = torch.autograd.grad(
                initial.potential_energy, initial.displacements, create_graph=False)[0]
        initial_free = torch.cat((initial_residual[:, 0][free_ux],
                                  initial_residual[:, 1][free_uy]))
        best_score = float(torch.linalg.vector_norm(initial_free)
                           / max(abs(problem.config.total_load), 1e-12))
        best_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
        best_optimizer = copy.deepcopy(optimizer.state_dict())
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        result = evaluate_physics(model, problem, rho, nodal_rho, compute_stress=False)
        consistency = (result.strain_energy - 0.5 * result.external_work) ** 2
        loss = result.potential_energy + config.energy_consistency_weight * consistency
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite potential energy: {loss}")
        loss_value = float(loss.detach())
        if (config.restore_best_training_state
                and (epoch % config.training_eval_interval == 0 or epoch == epochs)):
            if config.training_selection == "residual":
                residual = torch.autograd.grad(
                    result.potential_energy, result.displacements,
                    retain_graph=True, create_graph=False)[0]
                free_residual = torch.cat((
                    residual[:, 0][free_ux], residual[:, 1][free_uy]))
                score = float(torch.linalg.vector_norm(free_residual)
                              / max(abs(problem.config.total_load), 1e-12))
            else:
                score = loss_value
            if score < best_score:
                best_score = score
                best_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
                best_optimizer = copy.deepcopy(optimizer.state_dict())
        loss.backward()
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()
    if best_model is not None:
        model.load_state_dict(best_model)
        optimizer.load_state_dict(best_optimizer)
    model.eval()
    with torch.no_grad():
        result = evaluate_physics(model, problem, rho, nodal_rho, compute_stress=False)
        restored_loss = float(result.potential_energy)
    return restored_loss, (best_score if best_model is not None else restored_loss)


def optimize_case(config: BridgeConfig, problem: BridgeProblem,
                  output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = problem.device
    c = config
    model = GNNdPINN(
        problem.a_hat, problem.bc_fixed, problem.bc_symmetry,
        problem.lx, problem.ly, c.gradient_checkpointing).to(device)
    if c.initial_model_file:
        model_path = Path(c.initial_model_file)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parent / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Initial model file not found: {model_path}")
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        model.load_state_dict(state)
        print(f"Loaded warm-start network: {model_path}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=c.lr_step_size, gamma=c.lr_gamma)
    rho = problem.initial_density()
    history: list[dict[str, Any]] = []
    start_iteration = 0
    elapsed_prior = 0.0
    already_completed = False
    convergence_reason = "maximum iterations reached"
    checkpoint_path = output_dir / "checkpoint_latest.pt"

    if c.resume and checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if payload.get("case_signature") != _case_signature(c):
            raise ValueError("Checkpoint physics/mesh signature differs from the current configuration.")
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        if "rng_state" in payload:
            _restore_rng(payload["rng_state"], device)
        rho = payload["rho"].to(device=device, dtype=problem.dtype)
        history = payload["history"]
        start_iteration = int(payload["next_iteration"])
        elapsed_prior = float(payload.get("elapsed_prior", 0.0))
        if payload.get("completed", False):
            already_completed = True
            start_iteration = c.max_iterations
            convergence_reason = payload.get("convergence_reason", convergence_reason)
            print("Checkpoint is already complete; no further training will be performed.", flush=True)
        print(f"Resumed checkpoint at iteration {start_iteration}; next={start_iteration + 1}", flush=True)

    print("Optimization configuration:", json.dumps(config.to_dict(), ensure_ascii=False), flush=True)
    print("Mesh summary:", json.dumps(problem.summary(), ensure_ascii=False), flush=True)
    print(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    _sync(device)
    optimization_start = time.perf_counter()

    for opt_iter in range(start_iteration, c.max_iterations):
        iteration_start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        nodal_rho = problem.nodal_density(rho)
        if opt_iter == 0:
            epochs = c.epochs_init
        elif opt_iter < c.early_until:
            epochs = c.epochs_early
        elif opt_iter < c.middle_until:
            epochs = c.epochs_middle
        else:
            epochs = c.epochs_late

        try:
            last_loss, _ = _train_epochs(
                model, optimizer, problem, rho, nodal_rho, epochs, c)
        except Exception:
            _atomic_checkpoint(output_dir / "checkpoint_error.pt", {
                "case_signature": _case_signature(c), "model_state": model.state_dict(),
                "rng_state": _capture_rng(device),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "rho": rho.detach().cpu(), "history": history, "next_iteration": opt_iter,
                "elapsed_prior": elapsed_prior + time.perf_counter() - optimization_start,
            })
            raise

        model.eval()
        if c.sensitivity_mode == "autodiff":
            model.zero_grad(set_to_none=True)
            rho_ad = rho.detach().requires_grad_(True)
            nodal_ad = problem.nodal_density(rho_ad)
            result = evaluate_physics(model, problem, rho_ad, nodal_ad, compute_stress=False)
            compliance = float(result.external_work.detach())
            sensitivity = torch.autograd.grad(result.external_work, rho_ad)[0]
        else:
            with torch.no_grad():
                result = evaluate_physics(model, problem, rho, nodal_rho, compute_stress=False)
                compliance = float(result.external_work)
                sensitivity = simp_sensitivity(problem, rho, result.base_element_energy)
        if not torch.isfinite(sensitivity).all():
            raise FloatingPointError("Non-finite density sensitivity")
        if c.filter_mode == "density":
            filtered_density = problem.density_filter(rho)
            filtered_sensitivity = (
                problem.density_filter(rho * sensitivity)
                / torch.clamp(filtered_density, min=1e-3))
        else:
            filtered_sensitivity = problem.density_filter(sensitivity)
        rho_new = oc_update(problem, rho, filtered_sensitivity)
        rho_new = problem.apply_constraints_and_symmetry(rho_new)
        density_change = float(torch.max(torch.abs(rho_new - rho)))
        rho = rho_new
        scheduler.step()
        _sync(device)
        iteration_time = time.perf_counter() - iteration_start
        elapsed = elapsed_prior + time.perf_counter() - optimization_start
        row = {
            "iteration": opt_iter + 1,
            "compliance": compliance,
            "volume_fraction": float(rho.mean()),
            "design_volume_fraction": float(rho[problem.design_elements].mean()),
            "density_change": density_change,
            "loss": last_loss,
            "epochs": epochs,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "iteration_time_s": iteration_time,
            "cumulative_time_s": elapsed,
            "peak_gpu_memory_gb": _peak_gb(),
        }
        history.append(row)
        print(
            f"Iter {opt_iter + 1:03d}/{c.max_iterations} | C={compliance:.6e} | "
            f"change={density_change:.4e} | Vd={row['design_volume_fraction']:.5f} | "
            f"epochs={epochs} | loss={last_loss:.6e} | dt={iteration_time:.2f}s | "
            f"GPU peak={row['peak_gpu_memory_gb']:.2f}GB", flush=True)
        _save_history(history, output_dir)
        np.save(output_dir / "rho_latest.npy", rho.detach().cpu().numpy())

        if (opt_iter + 1) % c.checkpoint_interval == 0 or opt_iter + 1 == c.max_iterations:
            _atomic_checkpoint(checkpoint_path, {
                "case_signature": _case_signature(c), "model_state": model.state_dict(),
                "rng_state": _capture_rng(device),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "rho": rho.detach().cpu(), "history": history, "next_iteration": opt_iter + 1,
                "elapsed_prior": elapsed,
            })

        if opt_iter + 1 >= c.min_iterations_for_change_stop and density_change < c.change_tolerance:
            convergence_reason = f"density change below {c.change_tolerance}"
            break
        if c.enable_compliance_std_stop and opt_iter + 1 >= c.min_iterations_for_compliance_stop and len(history) >= c.compliance_window:
            recent = np.asarray([r["compliance"] for r in history[-c.compliance_window:]])
            relative_std = float(np.std(recent) / max(abs(np.mean(recent)), 1e-30))
            if relative_std < c.compliance_relative_std:
                convergence_reason = f"compliance relative std {relative_std:.6f} below {c.compliance_relative_std}"
                break

    final_equilibration_loss = None
    final_equilibration_start = time.perf_counter()
    if not already_completed and c.final_equilibration_epochs > 0:
        final_nodal_rho = problem.nodal_density(rho)
        final_equilibration_loss, _ = _train_epochs(
            model, optimizer, problem, rho, final_nodal_rho,
            c.final_equilibration_epochs, c)
        print(f"Final-density equilibration completed for {c.final_equilibration_epochs} epochs; loss={final_equilibration_loss:.6e}", flush=True)
    final_equilibration_time = 0.0 if already_completed else time.perf_counter() - final_equilibration_start
    model.eval()
    with torch.no_grad():
        final_nodal_rho = problem.nodal_density(rho)
        final = evaluate_physics(model, problem, rho, final_nodal_rho, compute_stress=True)
    _sync(device)
    optimization_time = elapsed_prior if already_completed else elapsed_prior + time.perf_counter() - optimization_start
    final_peak_gpu = _peak_gb()
    rho_np = rho.detach().cpu().numpy()
    displacement_np = final.displacements.detach().cpu().numpy()
    vm_np = final.von_mises.detach().cpu().numpy()
    design_values = rho_np[problem.design_mask_np.ravel()]
    metrics: dict[str, Any] = {
        "case_name": c.case_name,
        "completed": True,
        "convergence_reason": convergence_reason,
        "iterations": len(history),
        "final_compliance": float(final.external_work),
        "final_strain_energy": float(final.strain_energy),
        "final_potential_energy": float(final.potential_energy),
        "final_loss": float(final.potential_energy),
        "final_equilibration_epochs": 0 if already_completed else c.final_equilibration_epochs,
        "final_equilibration_loss": final_equilibration_loss,
        "final_equilibration_time_s": final_equilibration_time,
        "final_fields_match_final_density": True,
        "final_density_change": history[-1]["density_change"] if history else None,
        "final_volume_fraction": float(rho.mean()),
        "final_design_volume_fraction": float(design_values.mean()),
        "gray_fraction_design_0.1_0.9": float(np.mean((design_values > 0.1) & (design_values < 0.9))),
        "max_displacement_magnitude": float(np.linalg.norm(displacement_np, axis=1).max()),
        "max_von_mises": float(vm_np.max()),
        "optimization_time_s": optimization_time,
        "mean_iteration_time_s": float(np.mean([r["iteration_time_s"] for r in history])),
        "peak_gpu_memory_gb": max(max((r["peak_gpu_memory_gb"] for r in history), default=0.0), final_peak_gpu),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "config": c.to_dict(),
        "mesh": problem.summary(),
        "hardware": _hardware(device),
    }
    np.save(output_dir / "rho_half.npy", rho_np.reshape(c.nelx, c.nely))
    np.save(output_dir / "rho_full.npy", problem.mirror_density(rho_np))
    np.save(output_dir / "displacement_half.npy", displacement_np)
    np.save(output_dir / "von_mises_half.npy", vm_np)
    _save_history(history, output_dir)
    (output_dir / "results_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _atomic_checkpoint(checkpoint_path, {
        "case_signature": _case_signature(c), "model_state": model.state_dict(),
                "rng_state": _capture_rng(device),
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
        "rho": rho.detach().cpu(), "history": history, "next_iteration": len(history),
        "elapsed_prior": optimization_time, "completed": True,
        "convergence_reason": convergence_reason,
    })
    return metrics, history
