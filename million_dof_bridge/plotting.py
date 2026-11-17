"""Diagnostic figures and metric tables generated from a completed run."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from bridge_problem import BridgeProblem

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 16,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _save_both(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_validation(problem: BridgeProblem, metrics: dict[str, Any],
                    history: list[dict[str, Any]], output_dir: Path) -> None:
    c = problem.config
    rho = np.load(output_dir / "rho_half.npy")
    rho_full = problem.mirror_density(rho)
    displacement = np.load(output_dir / "displacement_half.npy")
    vm = np.load(output_dir / "von_mises_half.npy")
    u_mag = np.linalg.norm(displacement, axis=1).reshape(problem.nnx, problem.nny).T

    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.05, 1.0, 1.0], hspace=0.34, wspace=0.26)
    ax = fig.add_subplot(gs[0, :])
    image = ax.imshow(1.0 - rho_full.T, cmap="gray", origin="lower",
                      extent=[0.0, 2.0 * problem.lx, 0.0, problem.ly],
                      vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    ax.set_title("(a) Optimized full-bridge topology", fontweight="bold")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    bar.set_label(r"Void-to-solid indicator $1-\rho$")

    iterations = [row["iteration"] for row in history]
    compliance = [row["compliance"] for row in history]
    change = [max(row["density_change"], 1e-16) for row in history]
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(iterations, compliance, color="#184E77", linewidth=2.2)
    ax.set_title("(b) Compliance history", fontweight="bold")
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel(r"Compliance $F^{T}U$")
    ax.grid(True, linestyle="--", linewidth=0.6, color="#C8C8C8")

    ax = fig.add_subplot(gs[1, 1])
    ax.semilogy(iterations, change, color="#A4133C", linewidth=2.2)
    ax.set_title("(c) Density-update history", fontweight="bold")
    ax.set_xlabel("Optimization iteration")
    ax.set_ylabel(r"$\max|\Delta\rho|$")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, color="#C8C8C8")

    ax = fig.add_subplot(gs[2, 0])
    image = ax.imshow(u_mag, cmap="cividis", origin="lower",
                      extent=[0.0, problem.lx, 0.0, problem.ly], aspect="equal")
    ax.set_title("(d) Half-bridge displacement magnitude", fontweight="bold")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label(r"$|U|$")

    ax = fig.add_subplot(gs[2, 1])
    image = ax.imshow(vm.reshape(c.nelx, c.nely).T, cmap="cividis", origin="lower",
                      extent=[0.0, problem.lx, 0.0, problem.ly], aspect="equal")
    ax.set_title("(e) Half-bridge von Mises stress", fontweight="bold")
    ax.set_xlabel(r"$x$ (m)")
    ax.set_ylabel(r"$y$ (m)")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.set_label(r"$\sigma_{vm}$")

    fig.suptitle(f"{metrics['mesh']['displacement_dofs']:,} displacement DOFs", fontsize=20, fontweight="bold")
    caption_prefix = "Million-DOF" if problem.num_dofs >= 1_000_000 else "Smoke-test"
    fig.text(0.5, 0.012,
             f"{caption_prefix} GNN-dPINN bridge topology, convergence, and physical fields",
             ha="center", va="bottom", fontsize=18)
    fig.subplots_adjust(top=0.94, bottom=0.06)
    _save_both(fig, output_dir, "bridge_validation")


def plot_performance(metrics: dict[str, Any], history: list[dict[str, Any]], output_dir: Path) -> None:
    iterations = [row["iteration"] for row in history]
    times = [row["iteration_time_s"] for row in history]
    memory = [row["peak_gpu_memory_gb"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(iterations, times, color="#184E77", linewidth=2.0)
    axes[0].set_xlabel("Optimization iteration")
    axes[0].set_ylabel("Iteration wall-clock time (s)")
    axes[0].set_title("(a) Computational time", fontweight="bold")
    axes[0].grid(True, linestyle="--", linewidth=0.6, color="#C8C8C8")
    axes[1].plot(iterations, memory, color="#A4133C", linewidth=2.0)
    axes[1].set_xlabel("Optimization iteration")
    axes[1].set_ylabel("Peak allocated GPU memory (GB)")
    axes[1].set_title("(b) Memory footprint", fontweight="bold")
    axes[1].grid(True, linestyle="--", linewidth=0.6, color="#C8C8C8")
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    caption_prefix = "million-DOF bridge analysis" if metrics["mesh"]["displacement_dofs"] >= 1_000_000 else "small smoke-test analysis"
    fig.text(0.5, 0.02, f"Per-iteration computational cost of the {caption_prefix}",
             ha="center", fontsize=17)
    _save_both(fig, output_dir, "bridge_performance")


def write_result_tables(metrics: dict[str, Any], output_dir: Path) -> None:
    rows = [
        ("Elements", f"{metrics['mesh']['elements']:,}"),
        ("Nodes", f"{metrics['mesh']['nodes']:,}"),
        ("Displacement DOFs", f"{metrics['mesh']['displacement_dofs']:,}"),
        ("Free DOFs", f"{metrics['mesh']['free_dofs']:,}"),
        ("Iterations", str(metrics["iterations"])),
        ("Final compliance", f"{metrics['final_compliance']:.6e}"),
        ("Final design volume fraction", f"{metrics['final_design_volume_fraction']:.6f}"),
        ("Final maximum density change", f"{metrics['final_density_change']:.6e}"),
        ("Gray fraction in design domain", f"{100.0 * metrics['gray_fraction_design_0.1_0.9']:.2f}%"),
        ("Optimization time (s)", f"{metrics['optimization_time_s']:.2f}"),
        ("Mean iteration time (s)", f"{metrics['mean_iteration_time_s']:.3f}"),
        ("Peak allocated GPU memory (GB)", f"{metrics['peak_gpu_memory_gb']:.3f}"),
        ("Maximum displacement magnitude", f"{metrics['max_displacement_magnitude']:.6e}"),
        ("Maximum von Mises stress", f"{metrics['max_von_mises']:.6e}"),
        ("GPU", metrics["hardware"]["gpu"]),
    ]
    with (output_dir / "table_million_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Value"])
        writer.writerows(rows)
    lines = ["| Metric | Value |", "|---|---:|"] + [f"| {key} | {value} |" for key, value in rows]
    (output_dir / "table_million_metrics.md").write_text("\n".join(lines), encoding="utf-8")


def generate_all_outputs(problem: BridgeProblem, metrics: dict[str, Any],
                         history: list[dict[str, Any]], output_dir: Path) -> None:
    plot_validation(problem, metrics, history, output_dir)
    plot_performance(metrics, history, output_dir)
    write_result_tables(metrics, output_dir)
    manifest = {
        "required_files": [
            "results_summary.json", "metrics_history.csv",
            "bridge_validation.png", "bridge_validation.pdf",
            "bridge_performance.png", "bridge_performance.pdf",
            "table_million_metrics.csv", "rho_half.npy", "rho_full.npy",
        ],
        "result_status": "diagnostic run (reduced mesh)" if problem.num_dofs < 1_000_000 else "production run",
    }
    missing = [name for name in manifest["required_files"] if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing expected outputs: {missing}")
    if not np.isfinite(metrics["final_compliance"]):
        raise RuntimeError("Final compliance is not finite")
    if problem.num_dofs >= 1_000_000 and metrics["final_compliance"] <= 0.0:
        raise RuntimeError("Production run has non-positive final compliance; inspect the physics and training before using the result.")
    if problem.num_dofs >= 1_000_000 and metrics["iterations"] < 1:
        raise RuntimeError("Production run has no optimization history")
    (output_dir / "result_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
