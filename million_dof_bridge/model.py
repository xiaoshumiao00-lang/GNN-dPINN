"""GNN-dPINN architecture with sparse message passing and optional checkpointing."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class GNNdPINN(nn.Module):
    def __init__(self, a_hat: torch.Tensor, bc_fixed: torch.Tensor,
                 bc_symmetry: torch.Tensor, lx: float, ly: float,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.a_hat = a_hat
        self.bc_fixed = bc_fixed
        self.bc_symmetry = bc_symmetry
        self.lx, self.ly = float(lx), float(ly)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.gcn1 = nn.Linear(3, 32)
        self.gcn2 = nn.Linear(32, 32)
        self.register_buffer("B", torch.randn(2, 64) * 1.5)
        self.decoder = nn.Sequential(
            nn.Linear(160, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, 2),
        )

    def _maybe_checkpoint(self, fn, *args):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _gcn_first(self, features):
        return torch.tanh(torch.sparse.mm(self.a_hat, self.gcn1(features)))

    def _gcn_second(self, features):
        return torch.tanh(torch.sparse.mm(self.a_hat, self.gcn2(features)))

    def _rff(self, xy):
        projected = 2.0 * np.pi * torch.matmul(xy, self.B)
        return torch.cat((torch.sin(projected), torch.cos(projected)), dim=-1)

    def forward(self, nodes: torch.Tensor, nodal_rho: torch.Tensor):
        x_norm = (nodes[:, 0:1] - self.lx / 2.0) / (self.lx / 2.0)
        y_norm = (nodes[:, 1:2] - self.ly / 2.0) / (self.ly / 2.0)
        xy = torch.cat((x_norm, y_norm), dim=1)
        graph_input = torch.cat((xy, nodal_rho), dim=1)
        h1 = self._maybe_checkpoint(self._gcn_first, graph_input)
        z_graph = self._maybe_checkpoint(self._gcn_second, h1)
        x_rff = self._maybe_checkpoint(self._rff, xy)
        fused = torch.cat((x_rff, z_graph), dim=-1)
        u_pred = self._maybe_checkpoint(self.decoder, fused)
        free_ux = ~(self.bc_fixed | self.bc_symmetry)
        free_uy = ~self.bc_fixed
        ux = u_pred[:, 0] * free_ux
        uy = u_pred[:, 1] * free_uy
        return torch.stack((ux, uy), dim=1), z_graph
