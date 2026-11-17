"""Strain-energy physics for the million-DOF bridge scaling case."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from bridge_problem import BridgeProblem
from model import GNNdPINN


@dataclass
class PhysicsResult:
    potential_energy: torch.Tensor
    strain_energy: torch.Tensor
    base_element_energy: torch.Tensor
    external_work: torch.Tensor
    displacements: torch.Tensor
    von_mises: Optional[torch.Tensor]


def evaluate_physics(model: GNNdPINN, problem: BridgeProblem, rho: torch.Tensor,
                     nodal_rho: torch.Tensor, compute_stress: bool = False) -> PhysicsResult:
    """Evaluate total potential energy.

    The scalable engineering case uses ``strain_model='small'`` (service-load
    linear elasticity); ``strain_model='green_lagrange'`` retains the nonlinear
    St. Venant--Kirchhoff formulation used by the benchmark studies.
    """
    c = problem.config
    if c.strain_model not in ("small", "green_lagrange"):
        raise ValueError(f"Unknown strain_model: {c.strain_model}")
    u_nodes, _ = model(problem.nodes, nodal_rho)
    u_elems = u_nodes[problem.edof]
    base_energy = torch.zeros(problem.num_elements, dtype=problem.dtype, device=problem.device)
    vm = torch.zeros_like(base_energy) if compute_stress else None
    identity = torch.eye(2, dtype=problem.dtype, device=problem.device)

    for dn_dx, detj in zip(problem.dn_dx, problem.detj):
        grad_u = torch.matmul(u_elems.transpose(1, 2), dn_dx.T)
        if c.strain_model == "green_lagrange":
            deformation_gradient = identity + grad_u
            strain_tensor = 0.5 * (
                torch.matmul(deformation_gradient.transpose(1, 2), deformation_gradient)
                - identity)
        else:
            strain_tensor = 0.5 * (grad_u + grad_u.transpose(1, 2))
        e11 = strain_tensor[:, 0, 0]
        e22 = strain_tensor[:, 1, 1]
        e12 = strain_tensor[:, 0, 1]
        strain_voigt = torch.stack((e11, e22, 2.0 * e12), dim=1)
        stress_voigt = torch.matmul(strain_voigt, problem.d0.T)
        energy_density = 0.5 * torch.sum(strain_voigt * stress_voigt, dim=1)
        base_energy = base_energy + energy_density * detj
        if compute_stress:
            s11, s22, s12 = stress_voigt[:, 0], stress_voigt[:, 1], stress_voigt[:, 2]
            vm_gp = torch.sqrt(torch.clamp(
                s11**2 + s22**2 - s11*s22 + 3.0*s12**2, min=0.0))
            vm = vm + vm_gp / 4.0

    modulus = c.Emin + rho.flatten()**c.penalty * (c.E0 - c.Emin)
    strain_energy = torch.sum(base_energy * modulus)
    external_work = torch.sum(u_nodes[problem.load_node_indices, 1]) * problem.load_per_node
    potential_energy = strain_energy - external_work
    if compute_stress:
        vm = vm * rho.flatten()**c.penalty
    return PhysicsResult(
        potential_energy=potential_energy,
        strain_energy=strain_energy,
        base_element_energy=base_energy,
        external_work=external_work,
        displacements=u_nodes,
        von_mises=vm,
    )


def simp_sensitivity(problem: BridgeProblem, rho: torch.Tensor,
                     base_element_energy: torch.Tensor) -> torch.Tensor:
    """Partial compliance sensitivity at the approximately equilibrated state."""
    c = problem.config
    return (-c.penalty * rho**(c.penalty - 1.0)
            * (c.E0 - c.Emin) * base_element_energy)
