# GNN-dPINN

Code for *GNN-enhanced discrete physics-informed neural networks for efficient
geometrically nonlinear topology optimization* (P. Chen, Y. Yang, W. Tian, S. Weng).

## Contents

| Path | Description |
|---|---|
| `million_dof_bridge/` | Engineering-scale case: 740 m three-span bridge, 1480 x 504 half-bridge mesh, 1,495,810 displacement DOFs. Self-contained, with runners, configurations and unit tests. |
| `v1.0PiNN.py` | Discrete physics-informed neural network for geometrically nonlinear topology optimization. |
| `v1.1PiNN_MultiHole.py` | Multi-hole design domain variant of v1.0. |
| `v2.0GNNPINN.py` | Hybrid GNN-dPINN with latent-feature visualization. |
| `v2.1GNNPINN_MultiHole.py` | Multi-hole variant of v2.0. |
| `v3.0bridge_gnn_dpinn_v3Visual.py` | First bridge implementation (truss bridge, parabola-arch scheme) with visualization. |
| `bridge_results/` | Outputs of v3.0: density fields, convergence histories, topology animations. |

The scripts in the repository root are the development versions of the method and are
kept for reference. The million-DOF case is the implementation used for the
engineering-scale results; see `million_dof_bridge/README.md` for requirements, options
and outputs.

## Requirements

Python 3.10 or newer with PyTorch, numpy and matplotlib. The million-DOF case needs a
CUDA device; the reduced test run works on CPU.

## Citation

P. Chen, Y. Yang, W. Tian, S. Weng, GNN-enhanced discrete physics-informed neural
networks for efficient geometrically nonlinear topology optimization. Please cite the
journal version.
