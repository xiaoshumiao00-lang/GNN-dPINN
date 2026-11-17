from __future__ import annotations

import unittest

import numpy as np
import torch

from bridge_problem import (
    BridgeConfig, BridgeProblem, MatrixFreeDensityFilter, _structured_adjacency)
from model import GNNdPINN
from optimize import oc_update, _capture_rng, _restore_rng
from physics import evaluate_physics, simp_sensitivity


class MillionBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = BridgeConfig.load("config_smoke.json")
        cls.config.device = "cpu"
        cls.problem = BridgeProblem(cls.config, torch.device("cpu"))

    def test_production_mesh_is_million_dof(self):
        config = BridgeConfig.load("config_million.json")
        nodes = (config.nelx + 1) * (config.nely + 1)
        dofs = 2 * nodes
        self.assertEqual(config.nelx * config.nely, 745920)
        self.assertEqual(nodes, 747905)
        self.assertEqual(dofs, 1495810)
        self.assertGreaterEqual(dofs, 1000000)
        self.assertEqual(config.nelx % 370, 0)
        self.assertEqual(config.nely % 126, 0)
        self.assertAlmostEqual(config.lx / config.nelx, 0.25)
        self.assertAlmostEqual(config.ly / config.nely, 0.25)

    def test_original_mesh_masks_match_public_bridge(self):
        config = BridgeConfig.load("config_smoke.json")
        config.nelx, config.nely = 370, 126
        config.device = "cpu"
        p = BridgeProblem(config, torch.device("cpu"))
        self.assertEqual(p.num_elements, 46620)
        self.assertEqual(p.num_nodes, 47117)
        self.assertEqual(p.num_dofs, 94234)
        self.assertEqual(p.free_dofs, 93039)
        self.assertEqual(p.n_fixed, 2700)
        self.assertEqual(p.n_void, 31172)
        self.assertEqual(p.n_design, 12748)
        self.assertEqual(int(p.design1_mask_np.sum()), 9362)
        self.assertEqual(int(p.design2_mask_np.sum()), 2666)
        self.assertEqual(len(p.load_node_indices_np), 371)
        self.assertAlmostEqual(p.target_volume, 9074.0)

    def test_matrix_free_filter_matches_dense(self):
        nelx, nely, dx, dy, radius = 5, 4, 2.0, 3.0, 4.5
        op = MatrixFreeDensityFilter(nelx, nely, dx, dy, radius, torch.device("cpu"), torch.float32)
        # Compare against an explicit dense cone filter in the same precision.
        values = np.random.default_rng(3).random(nelx * nely).astype(np.float32)
        x = (np.arange(nelx) + 0.5) * dx
        y = (np.arange(nely) + 0.5) * dy
        h = np.zeros((nelx*nely, nelx*nely), dtype=np.float32)
        for i in range(nelx):
            for j in range(nely):
                e1 = i*nely+j
                for k in range(nelx):
                    for l in range(nely):
                        e2 = k*nely+l
                        distance = np.hypot(x[i]-x[k], y[j]-y[l])
                        h[e1, e2] = max(0.0, radius-distance)
        expected = h @ values / h.sum(axis=1)
        actual = op(torch.from_numpy(values)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            op(torch.ones(nelx*nely, dtype=torch.float32)).numpy(),
            np.ones(nelx*nely, dtype=np.float32), rtol=2e-6, atol=2e-6)

    def test_structured_adjacency_matches_element_pairs(self):
        nnx, nny = 4, 3
        rows, cols = _structured_adjacency(nnx, nny)
        actual = set(zip(rows.tolist(), cols.tolist()))
        expected = set()
        for i in range(nnx-1):
            for j in range(nny-1):
                nodes = [i*nny+j, (i+1)*nny+j, (i+1)*nny+j+1, i*nny+j+1]
                for a in nodes:
                    for b in nodes:
                        expected.add((a, b))
        self.assertEqual(actual, expected)
        self.assertEqual(len(rows), len(expected))

    def test_boundary_nodes_and_dofs(self):
        p = self.problem
        self.assertGreater(p.free_dofs, 0)
        self.assertLess(p.free_dofs, p.num_dofs)
        self.assertEqual(len(p.load_node_indices_np), p.nnx)
        self.assertTrue(torch.any(p.bc_fixed))
        self.assertTrue(torch.any(p.bc_symmetry))
        self.assertEqual(p.symmetry_i.numel(), p.symmetry_j.numel())
        self.assertGreater(p.symmetry_i.numel(), 0)

    def test_nonlinear_physics_backward_and_hard_bc(self):
        p = self.problem
        torch.manual_seed(42)
        model = GNNdPINN(p.a_hat, p.bc_fixed, p.bc_symmetry, p.lx, p.ly, False)
        rho = p.initial_density()
        result = evaluate_physics(model, p, rho, p.nodal_density(rho), compute_stress=True)
        self.assertTrue(torch.isfinite(result.potential_energy))
        self.assertGreater(float(result.strain_energy.detach()), 0.0)
        self.assertEqual(result.von_mises.numel(), p.num_elements)
        fixed_rows = result.displacements[p.bc_fixed]
        self.assertTrue(torch.allclose(fixed_rows, torch.zeros_like(fixed_rows)))
        symmetry_ux = result.displacements[p.bc_symmetry, 0]
        self.assertTrue(torch.allclose(symmetry_ux, torch.zeros_like(symmetry_ux)))
        result.potential_energy.backward()
        gradients = [param.grad for param in model.parameters() if param.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        sensitivity = simp_sensitivity(p, rho, result.base_element_energy)
        self.assertTrue(torch.isfinite(sensitivity).all())

    def test_oc_volume_and_mirror(self):
        p = self.problem
        rho = p.initial_density()
        dc = -torch.ones_like(rho)
        updated = oc_update(p, rho, dc)
        updated = p.apply_constraints_and_symmetry(updated)
        self.assertAlmostEqual(float(updated.sum()), p.target_volume, delta=1e-2)
        full = p.mirror_density(updated)
        self.assertEqual(full.shape, (2*p.nelx, p.nely))
        np.testing.assert_allclose(full[:p.nelx], full[p.nelx:][::-1])

    def test_coarse_density_warm_start_upsamples_and_preserves_volume(self):
        config = BridgeConfig.load("config_smoke.json")
        config.device = "cpu"
        config.initial_density_file = "assets/bridge_coarse_370x126.npy"
        p = BridgeProblem(config, torch.device("cpu"))
        rho = p.initial_density()
        self.assertEqual(rho.shape, (p.num_elements,))
        self.assertTrue(torch.isfinite(rho).all())
        self.assertTrue(torch.allclose(rho[p.fixed_elements], torch.ones_like(rho[p.fixed_elements])))
        self.assertTrue(torch.allclose(rho[p.void_elements], torch.zeros_like(rho[p.void_elements])))
        self.assertAlmostEqual(float(rho[p.design_elements].mean()), config.volfrac, delta=0.012)

    def test_rng_restore_accepts_cuda_mapped_states(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        device = torch.device("cuda")
        state = _capture_rng(device)
        state["torch"] = state["torch"].cuda()
        state["cuda"] = [item.cuda() for item in state["cuda"]]
        _restore_rng(state, device)
        self.assertTrue(all(item.device.type == "cpu" for item in torch.cuda.get_rng_state_all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
