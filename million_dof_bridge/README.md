# Million-DOF GNN-dPINN bridge case

Reference implementation of the engineering-scale study: a 740 m three-span long-span
bridge discretized into a 1480 x 504 quadrilateral mesh, which gives 745,920 elements,
747,905 nodes and 1,495,810 displacement degrees of freedom. A half-bridge model is
optimized and mirrored to obtain the full-bridge layout.

## Requirements

- Python 3.10 or newer
- PyTorch 2.2 or newer with CUDA (the production case requires a GPU)
- numpy, matplotlib
- About 3 GB of free GPU memory for the production mesh

## Running

Linux or macOS:

```bash
bash run.sh
```

Windows: run `run_windows.bat`, either by double-clicking it or from a terminal.

| Option | Effect |
|---|---|
| none | Production case, resuming from `checkpoint_latest.pt` when present |
| `--smoke-test` | 74 x 63 mesh with two outer iterations, to check the pipeline end to end |
| `--check-only` | Builds the mesh and the sparse operators, reports the DOF count, runs no optimization |
| `--no-resume` | Ignores the checkpoint and restarts from the initial density |
| `--output-dir DIR` | Writes results elsewhere (default `results/million_1480x504_final`) |
| `--device {auto,cuda,cpu}` | Overrides the device in the configuration |

The production run is long: 150 outer iterations with 300 training epochs each. A
checkpoint is written every five iterations, so an interrupted run continues from the
last one.

## Configuration

`config_million.json` holds the production settings; `config_smoke.json` holds the
reduced settings used by `--smoke-test`. The main entries are the mesh (`nelx`, `nely`)
and physical size (`lx`, `ly`) in metres, the volume fraction and SIMP parameters, the
filter radius in metres, the load, the training budget per iteration, the OC move limit,
the convergence tolerances, and the checkpoint interval. `strain_model` selects the
strain measure. `assets/bridge_coarse_370x126.npy` provides the initial density field of
the original coarse model and `assets/bridge_warm_model.pt` the warm-start weights.

## Outputs

| File | Content |
|---|---|
| `results_summary.json` | Mesh, hardware, configuration and all reported metrics |
| `mesh_summary.json` | Elements, nodes, DOFs, boundary-condition counts |
| `metrics_history.csv`, `history_*.npy` | Per-iteration compliance, volume fraction, density change, loss and wall-clock time |
| `bridge_validation.png/.pdf` | Full-bridge topology, compliance history, density-update history, displacement magnitude, von Mises stress |
| `bridge_performance.png/.pdf` | Per-iteration wall-clock time and peak GPU memory |
| `table_million_metrics.csv/.md` | Metrics table |
| `rho_half.npy`, `rho_full.npy` | Half-bridge density field and its mirrored full-bridge version |
| `displacement_half.npy`, `von_mises_half.npy` | Nodal displacement and element von Mises stress |
| `checkpoint_latest.pt` | Model, optimizer, RNG state and history for resume |
| `run.log` | Console output of the run |

## Verification

```bash
python -m unittest discover -s tests -v
python run.py --check-only
python memory_probe.py
```

The production mesh gives 745,920 elements, 747,905 nodes, 1,495,810 displacement DOFs,
1,479,525 free DOFs and a block-sparse adjacency with 6,719,233 nonzeros at
dx = dy = 0.25 m. On one 16 GB card, `memory_probe.py` needs about 9.5 s for a single
forward and backward pass, with 2.39 GB peak allocated and 4.30 GB peak reserved.

The nine unit tests check the mesh and mask counts against the original 370 x 126 model,
the matrix-free cone filter against an explicit `H x / Hs` reference, the analytic
eight-neighbour adjacency against the element connectivity, the Green-Lagrange energy,
autograd gradients, the SIMP sensitivities, the OC volume update, and checkpoint resume
on CUDA.

## Numerical notes

- **Strain measure.** The bridge case runs at service load with `strain_model = "small"`
  (linear elasticity). Setting `strain_model = "green_lagrange"` switches to the
  Green-Lagrange strain and St. Venant-Kirchhoff stress used in the benchmark studies.
- **Filtering.** The density filter is a matrix-free convolution that reproduces
  `H x / Hs`. The cone radius is fixed in physical length (3 m), so it does not shrink
  when the mesh is refined.
- **Graph operators.** The GCN adjacency is generated analytically from the structured
  grid, so no dense neighbour table is stored.
- **Memory.** float32 throughout, with optional gradient checkpointing and a TF32 switch
  (`allow_tf32`).
- **Final fields.** After the last OC, symmetry and volume update, the density is frozen
  and training continues for `final_equilibration_epochs`; compliance, displacement and
  stress are evaluated afterwards, so the reported fields belong to the reported
  topology.

## Files

| File | Role |
|---|---|
| `run.py` | Command-line entry point; seeding, logging, preflight, output |
| `bridge_problem.py` | Geometry, mesh, boundary conditions, sparse operators, config |
| `model.py` | GNN-dPINN network with sparse message passing and optional checkpointing |
| `physics.py` | Total potential energy, strain measure, von Mises stress, SIMP sensitivity |
| `optimize.py` | Outer optimization loop, training, OC update, checkpoints |
| `plotting.py` | Figures, metric tables and the result manifest |
| `memory_probe.py` | Single forward/backward probe on the production mesh |
| `tests/test_invariants.py` | Unit tests |
| `run.sh`, `run_windows.bat` | Launch scripts |
| `config_million.json`, `config_smoke.json` | Production and reduced configurations |
| `assets/` | Initial density field and warm-start weights |

## Citation

P. Chen, Y. Yang, W. Tian, S. Weng, GNN-enhanced discrete physics-informed neural
networks for efficient geometrically nonlinear topology optimization. Please cite the
journal version.
