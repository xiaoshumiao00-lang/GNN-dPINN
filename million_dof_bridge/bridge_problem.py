"""Geometry, mesh, boundary conditions, and sparse operators for the bridge case."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class BridgeConfig:
    case_name: str = "bridge_million_1480x504"
    nelx: int = 1480
    nely: int = 504
    lx: float = 370.0
    ly: float = 126.0
    volfrac: float = 0.5
    penalty: float = 3.0
    filter_radius_physical: float = 3.0
    E0: float = 1.0
    Emin: float = 1e-4
    nu: float = 0.3
    total_load: float = -0.5
    epochs_init: int = 500
    epochs_early: int = 300
    epochs_middle: int = 300
    epochs_late: int = 200
    final_equilibration_epochs: int = 200
    initial_density_file: str | None = None
    initial_model_file: str | None = None
    enable_compliance_std_stop: bool = False
    restore_best_training_state: bool = True
    training_eval_interval: int = 10
    training_selection: str = "residual"
    energy_consistency_weight: float = 0.01
    gradient_clip_norm: float = 1.0
    sensitivity_mode: str = "energy"
    filter_mode: str = "density"
    strain_model: str = "small"
    early_until: int = 20
    middle_until: int = 60
    learning_rate: float = 5e-4
    lr_step_size: int = 50
    lr_gamma: float = 0.5
    max_iterations: int = 150
    move: float = 0.02
    min_iterations_for_change_stop: int = 81
    change_tolerance: float = 0.003
    min_iterations_for_compliance_stop: int = 101
    compliance_window: int = 30
    compliance_relative_std: float = 0.02
    checkpoint_interval: int = 5
    seed: int = 42
    device: str = "cuda"
    gradient_checkpointing: bool = True
    allow_tf32: bool = True
    output_dir: str = "results/million_1480x504"
    resume: bool = True

    @classmethod
    def load(cls, path: str | Path) -> "BridgeConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.nelx <= 0 or self.nely <= 0:
            raise ValueError("nelx and nely must be positive")
        if not 0.0 < self.volfrac <= 1.0:
            raise ValueError("volfrac must be in (0, 1]")
        if self.filter_radius_physical <= 0.0:
            raise ValueError("filter radius must be positive")
        if self.max_iterations <= 0 or self.epochs_init <= 0:
            raise ValueError("iteration and epoch counts must be positive")
        if self.final_equilibration_epochs < 0:
            raise ValueError("final_equilibration_epochs must be nonnegative")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Use a CUDA-enabled PyTorch build or set device=auto/cpu for a smoke test."
            )


class MatrixFreeDensityFilter:
    """Cone filter H x / Hs evaluated by convolution, with no dense H matrix."""

    def __init__(self, nelx: int, nely: int, dx: float, dy: float,
                 radius: float, device: torch.device, dtype: torch.dtype):
        self.nelx, self.nely = nelx, nely
        self.dx, self.dy, self.radius = dx, dy, radius
        rx = int(math.ceil(radius / dx))
        ry = int(math.ceil(radius / dy))
        ox = np.arange(-rx, rx + 1, dtype=np.float64) * dx
        oy = np.arange(-ry, ry + 1, dtype=np.float64) * dy
        distance = np.sqrt(ox[:, None] ** 2 + oy[None, :] ** 2)
        kernel_np = np.maximum(0.0, radius - distance).astype(np.float32)
        kernel_np[distance >= radius] = 0.0
        self.kernel = torch.as_tensor(kernel_np, dtype=dtype, device=device).view(1, 1, *kernel_np.shape)
        self.padding = (rx, ry)  # conv2d padding order is (H=x, W=y)
        ones = torch.ones(1, 1, nelx, nely, dtype=dtype, device=device)
        self.normalizer = F.conv2d(ones, self.kernel, padding=self.padding)
        self.nnz = int(np.count_nonzero(kernel_np)) * nelx * nely

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        grid = values.reshape(1, 1, self.nelx, self.nely)
        filtered = F.conv2d(grid, self.kernel, padding=self.padding) / self.normalizer
        return filtered.reshape(-1)


def lower_boundary_y(x: np.ndarray, lx: float, pier_left: float, pier_right: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    mask = (x >= 0.0) & (x <= 30.0)
    result[mask] = 100.0 - (2.0 / 3.0) * x[mask]
    mask = (x > 30.0) & (x <= 95.0)
    result[mask] = 80.0
    mask = (x > 95.0) & (x <= pier_left)
    result[mask] = -40.0 / 94.0**2 * (x[mask] - 95.0) ** 2 + 80.0
    mask = (x > pier_left) & (x < pier_right)
    result[mask] = np.nan
    mask = (x >= pier_right) & (x <= 295.0)
    result[mask] = -40.0 / 94.0**2 * (x[mask] - 295.0) ** 2 + 80.0
    mask = (x > 295.0) & (x <= lx)
    result[mask] = 80.0
    return result


def arch_upper_boundary_y(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return -0.002 * (x - 195.0) ** 2 + 126.0


def _structured_adjacency(nnx: int, nny: int) -> tuple[np.ndarray, np.ndarray]:
    """Eight-neighbor plus self adjacency induced by four-node quadrilaterals."""
    grid = np.arange(nnx * nny, dtype=np.int64).reshape(nnx, nny)
    rows, cols = [], []
    for sx in (-1, 0, 1):
        for sy in (-1, 0, 1):
            xs = np.arange(max(0, -sx), min(nnx, nnx - sx))
            ys = np.arange(max(0, -sy), min(nny, nny - sy))
            if len(xs) == 0 or len(ys) == 0:
                continue
            source = grid[np.ix_(xs, ys)]
            target = grid[np.ix_(xs + sx, ys + sy)]
            rows.append(source.ravel())
            cols.append(target.ravel())
    return np.concatenate(rows), np.concatenate(cols)


class BridgeProblem:
    def __init__(self, config: BridgeConfig, device: torch.device):
        self.config = config
        self.device = device
        self.dtype = torch.float32
        c = config
        self.nelx, self.nely = c.nelx, c.nely
        self.lx, self.ly = c.lx, c.ly
        self.dx, self.dy = c.lx / c.nelx, c.ly / c.nely
        self.nnx, self.nny = c.nelx + 1, c.nely + 1
        self.num_elements = c.nelx * c.nely
        self.num_nodes = self.nnx * self.nny
        self.num_dofs = 2 * self.num_nodes

        self.x_elem = (np.arange(c.nelx) + 0.5) * self.dx
        self.y_elem = (np.arange(c.nely) + 0.5) * self.dy
        x = self.x_elem[:, None]
        y = self.y_elem[None, :]
        pier_center, pier_half_width = 195.0, 6.0
        pier_left, pier_right = 189.0, 201.0
        deck_bottom, deck_top = 100.0, 106.0

        deck = (y >= deck_bottom) & (y <= deck_top)
        deck = np.broadcast_to(deck, (c.nelx, c.nely))
        pier_fixed = (np.abs(x - pier_center) < pier_half_width) & (y <= 40.0)
        pier_design = ((np.abs(x - pier_center) < pier_half_width)
                       & (y > 40.0) & (y <= 100.0))
        pier = pier_fixed | pier_design
        lower = lower_boundary_y(self.x_elem, self.lx, pier_left, pier_right)[:, None]
        valid_lower = ~np.isnan(lower)
        design1 = (y >= lower) & (y < deck_bottom) & ~pier & valid_lower
        arch_top = arch_upper_boundary_y(self.x_elem)[:, None]
        design2 = ((x >= 95.0) & (x <= 295.0)
                   & (y >= deck_top) & (y <= arch_top))
        void_below = (y < lower) & valid_lower & ~pier
        void_arch_left = (x < 95.0) & (y > deck_top)
        void_arch_right = (x > 295.0) & (y > deck_top)
        void_arch_above = (x >= 95.0) & (x <= 295.0) & (y > arch_top)
        void = void_below | void_arch_left | void_arch_right | void_arch_above
        fixed = deck | pier_fixed
        design = ~fixed & ~void

        self.fixed_mask_np = fixed
        self.void_mask_np = void
        self.design_mask_np = design
        self.design1_mask_np = design1
        self.design2_mask_np = design2
        self.n_fixed = int(fixed.sum())
        self.n_void = int(void.sum())
        self.n_design = int(design.sum())
        self.target_volume = float(self.n_fixed + c.volfrac * self.n_design)
        self.fixed_elements = torch.as_tensor(np.flatnonzero(fixed.ravel()), dtype=torch.long, device=device)
        self.void_elements = torch.as_tensor(np.flatnonzero(void.ravel()), dtype=torch.long, device=device)
        self.design_elements = torch.as_tensor(np.flatnonzero(design.ravel()), dtype=torch.long, device=device)

        x_nodes = np.linspace(0.0, self.lx, self.nnx)
        y_nodes = np.linspace(0.0, self.ly, self.nny)
        xx, yy = np.meshgrid(x_nodes, y_nodes, indexing="ij")
        self.nodes_np = np.column_stack((xx.ravel(), yy.ravel()))
        self.nodes = torch.as_tensor(self.nodes_np, dtype=self.dtype, device=device)
        left_pin = ((self.nodes_np[:, 0] < 1e-6)
                    & (np.abs(self.nodes_np[:, 1] - 100.0) < 1e-6))
        right_symmetry = self.nodes_np[:, 0] > self.lx - 1e-6
        pier_base = ((np.abs(self.nodes_np[:, 0] - pier_center) < pier_half_width + 1e-6)
                     & (self.nodes_np[:, 1] <= 40.0))
        bc_type = np.zeros(self.num_nodes, dtype=np.int8)
        bc_type[left_pin | pier_base] = 3
        bc_type[right_symmetry] = 2
        self.bc_fixed = torch.as_tensor(bc_type == 3, dtype=torch.bool, device=device)
        self.bc_symmetry = torch.as_tensor(bc_type == 2, dtype=torch.bool, device=device)
        fixed_ux = int(np.count_nonzero((bc_type == 3) | (bc_type == 2)))
        fixed_uy = int(np.count_nonzero(bc_type == 3))
        self.free_dofs = self.num_dofs - fixed_ux - fixed_uy
        if not left_pin.any():
            raise ValueError("The mesh has no node at the left support (0,100); refine nely by an integer factor of 126.")
        load_mask = np.abs(self.nodes_np[:, 1] - deck_top) < 1e-6
        self.load_node_indices_np = np.flatnonzero(load_mask)
        if len(self.load_node_indices_np) == 0:
            raise ValueError("The mesh has no deck-top load nodes at y=106; refine nely by an integer factor of 126.")
        self.load_node_indices = torch.as_tensor(self.load_node_indices_np, dtype=torch.long, device=device)
        self.load_per_node = float(c.total_load / len(self.load_node_indices_np))

        i = np.arange(c.nelx, dtype=np.int64)[:, None]
        j = np.arange(c.nely, dtype=np.int64)[None, :]
        n1 = i * self.nny + j
        n2 = (i + 1) * self.nny + j
        n3 = (i + 1) * self.nny + j + 1
        n4 = i * self.nny + j + 1
        edof = np.stack((n1, n2, n3, n4), axis=-1).reshape(-1, 4)
        self.edof = torch.as_tensor(edof, dtype=torch.long, device=device)

        self.d0 = (c.E0 / (1.0 - c.nu**2)) * torch.tensor(
            [[1.0, c.nu, 0.0], [c.nu, 1.0, 0.0], [0.0, 0.0, (1.0 - c.nu) / 2.0]],
            dtype=self.dtype, device=device)
        gp = 1.0 / math.sqrt(3.0)
        gauss_points = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
        self.dn_dx, self.detj = [], []
        for xi, eta in gauss_points:
            dn = np.array([[-(1-eta), 1-eta, 1+eta, -(1+eta)],
                           [-(1-xi), -(1+xi), 1+xi, 1-xi]], dtype=np.float64) / 4.0
            jac = np.diag([self.dx / 2.0, self.dy / 2.0])
            self.dn_dx.append(torch.as_tensor(np.linalg.solve(jac, dn), dtype=self.dtype, device=device))
            self.detj.append(float(np.linalg.det(jac)))

        self.density_filter = MatrixFreeDensityFilter(
            c.nelx, c.nely, self.dx, self.dy, c.filter_radius_physical, device, self.dtype)
        e2n_row = torch.as_tensor(edof.ravel(), dtype=torch.long, device=device)
        e2n_col = torch.arange(self.num_elements, dtype=torch.long, device=device).repeat_interleave(4)
        e2n_indices = torch.stack((e2n_row, e2n_col))
        e2n_values = torch.ones(e2n_row.numel(), dtype=self.dtype, device=device)
        self.e2n = torch.sparse_coo_tensor(
            e2n_indices, e2n_values, size=(self.num_nodes, self.num_elements), device=device).coalesce()
        degree_np = np.bincount(edof.ravel(), minlength=self.num_nodes).astype(np.float32)
        self.node_degrees = torch.as_tensor(degree_np, dtype=self.dtype, device=device).unsqueeze(1)

        row_np, col_np = _structured_adjacency(self.nnx, self.nny)
        adjacency_degree = np.bincount(row_np, minlength=self.num_nodes).astype(np.float64)
        values_np = (1.0 / np.sqrt(adjacency_degree[row_np] * adjacency_degree[col_np])).astype(np.float32)
        indices = torch.as_tensor(np.stack((row_np, col_np)), dtype=torch.long, device=device)
        values = torch.as_tensor(values_np, dtype=self.dtype, device=device)
        self.a_hat = torch.sparse_coo_tensor(
            indices, values, size=(self.num_nodes, self.num_nodes), device=device).coalesce()
        self.graph_nnz = int(values.numel())
        self.symmetry_i, self.symmetry_j = self._build_symmetry_pairs()

    def _build_symmetry_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        left, right = [], []
        for i, x_i in enumerate(self.x_elem):
            if not (95.0 <= x_i <= 195.0):
                continue
            x_mirror = 390.0 - x_i
            j = int(np.argmin(np.abs(self.x_elem - x_mirror)))
            if not (195.0 <= self.x_elem[j] <= 295.0) or i >= j:
                continue
            left.append(i)
            right.append(j)
        return (torch.as_tensor(left, dtype=torch.long, device=self.device),
                torch.as_tensor(right, dtype=torch.long, device=self.device))

    def initial_density(self) -> torch.Tensor:
        c = self.config
        if c.initial_density_file:
            path = Path(c.initial_density_file)
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            if not path.is_file():
                raise FileNotFoundError(f"Initial density file not found: {path}")
            array = np.load(path).astype(np.float32)
            if array.ndim != 2:
                raise ValueError(f"Initial density must be a 2-D array, got shape {array.shape}")
            if array.shape == (self.nely, self.nelx) and array.shape != (self.nelx, self.nely):
                array = array.T
            elif array.shape[0] < array.shape[1] and self.nelx > self.nely:
                array = array.T
            tensor = torch.as_tensor(array, dtype=self.dtype, device=self.device).view(
                1, 1, *array.shape)
            if array.shape != (self.nelx, self.nely):
                tensor = F.interpolate(tensor, size=(self.nelx, self.nely),
                                       mode="bilinear", align_corners=False)
            rho = torch.clamp(tensor.reshape(-1), 0.0, 1.0)
            rho[self.fixed_elements] = 1.0
            rho[self.void_elements] = 0.0
            return self.apply_constraints_and_symmetry(rho)
        rho = torch.full((self.num_elements,), c.volfrac, dtype=self.dtype, device=self.device)
        rho[self.fixed_elements] = 1.0
        rho[self.void_elements] = 0.0
        return rho

    def nodal_density(self, rho: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.e2n, rho.unsqueeze(1)) / self.node_degrees

    def apply_constraints_and_symmetry(self, rho: torch.Tensor) -> torch.Tensor:
        c = self.config
        rho = rho.clone()
        if self.symmetry_i.numel() > 0:
            grid = rho.view(self.nelx, self.nely)
            average = 0.5 * (grid[self.symmetry_i] + grid[self.symmetry_j])
            grid[self.symmetry_i] = average
            grid[self.symmetry_j] = average
        rho[self.fixed_elements] = 1.0
        rho[self.void_elements] = 0.0
        current = rho[self.design_elements].sum()
        target = c.volfrac * self.n_design
        for _ in range(20):
            current = rho[self.design_elements].sum()
            if float(current) <= 0.0:
                break
            rho[self.design_elements] = torch.clamp(
                rho[self.design_elements] * (target / current), 0.001, 1.0)
            if abs(float(rho[self.design_elements].sum()) - target) < 1e-3 * self.n_design:
                break
        return rho

    def mirror_density(self, rho: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(rho, torch.Tensor):
            rho = rho.detach().cpu().numpy()
        half = rho.reshape(self.nelx, self.nely)
        return np.concatenate((half, half[::-1]), axis=0)

    def summary(self) -> dict[str, Any]:
        return {
            "elements": self.num_elements,
            "nodes": self.num_nodes,
            "displacement_dofs": self.num_dofs,
            "free_dofs": self.free_dofs,
            "dx": self.dx,
            "dy": self.dy,
            "fixed_elements": self.n_fixed,
            "void_elements": self.n_void,
            "design_elements": self.n_design,
            "design_domain_I_elements": int(self.design1_mask_np.sum()),
            "design_domain_II_elements": int(self.design2_mask_np.sum()),
            "load_nodes": len(self.load_node_indices_np),
            "load_per_node": self.load_per_node,
            "graph_nnz": self.graph_nnz,
            "filter_equivalent_nnz": self.density_filter.nnz,
            "filter_storage": "matrix-free convolution",
            "symmetry_pairs": int(self.symmetry_i.numel()),
        }
