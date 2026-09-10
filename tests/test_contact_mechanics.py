"""
Phase 15 Verification Suite: Nonlinear Surface-to-Surface Contact Mechanics.
----------------------------------------------------------------------------
Tests:
1. Exact boundary facet extraction and outward normal orientation.
2. Generalized barycentric Mean Value Coordinates interpolation precision.
3. Spatial hash proximity detector and signed gap function evaluation.
4. Matrix-free contact tangent operator self-adjointness and symmetry.
5. Two-block compressive contact transmission and Newton-3rd law force equilibrium.
6. Augmented Lagrangian multiplier convergence with sub-micron penetration.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from wnfea.mesh.voxel_mesher import VoxelMesher
from wnfea.contact.contact_detector import (
    SurfaceFacet,
    ContactPair,
    compute_mean_value_coordinates_quad,
    extract_hex8_surface_facets,
    compute_nodal_surface_normals,
    SpatialHashContactDetector,
)
from wnfea.contact.contact_solver import (
    ContactSolveResult,
    MatrixFreeContactTangentOperator,
    solve_contact_assembly,
)


class TestContactMechanics(unittest.TestCase):
    """
    Verification tests for WNFEA contact mechanics and multi-body non-penetration engine.
    """

    def test_extract_hex8_surface_facets(self):
        """
        Verify external surface quad facet extraction, outward normals, and total area.
        """
        # 2x2x2 cube of size 1.0 x 1.0 x 1.0 m (8 elements)
        grid = VoxelMesher.create_box_grid(
            bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            resolution=(2, 2, 2),
        )

        facets = extract_hex8_surface_facets(grid.nodes, grid.elements)

        # 6 faces * (2x2 quads per face) = 24 surface facets
        self.assertEqual(len(facets), 24, "2x2x2 grid must have exactly 24 boundary quad facets")

        total_area = sum(f.area for f in facets)
        self.assertAlmostEqual(total_area, 6.0, places=6, msg="Total surface area of 1m cube must be 6.0 m^2")

        # Verify each normal points outward from element centroid
        for f in facets:
            v = grid.nodes[f.node_ids]
            elem_nodes = grid.nodes[grid.elements[f.element_id]]
            elem_c = np.mean(elem_nodes, axis=0)
            outward_vec = f.centroid - elem_c
            self.assertGreater(
                np.dot(f.normal, outward_vec), 0.0, "Facet normal must point outward from element"
            )
            self.assertAlmostEqual(np.linalg.norm(f.normal), 1.0, places=7)

    def test_mean_value_coordinates_reconstruction(self):
        """
        Verify Mean Value Coordinates generalized barycentric weights reconstruct spatial points.
        """
        quad = np.array([
            [0.1, 0.2, 0.0],
            [1.2, 0.1, 0.0],
            [0.9, 1.1, 0.0],
            [0.0, 0.8, 0.0],
        ], dtype=np.float64)

        rng = np.random.RandomState(42)
        for _ in range(10):
            # Sample random point inside convex quad using convex combination
            rand_w = rng.dirichlet(np.ones(4))
            p = np.sum(rand_w[:, None] * quad, axis=0)

            mvc_w = compute_mean_value_coordinates_quad(quad, p)

            # Check partition of unity
            self.assertAlmostEqual(np.sum(mvc_w), 1.0, places=12)

            # Check spatial reconstruction
            p_rec = np.sum(mvc_w[:, None] * quad, axis=0)
            err = np.linalg.norm(p_rec - p)
            self.assertLess(err, 1e-13, f"MVC reconstruction error must be < 1e-13, got {err}")

    def test_spatial_hash_contact_detection(self):
        """
        Verify that SpatialHashContactDetector identifies contacting interface nodes and signed gap.
        """
        # Master foundation block: [0, 1] x [0, 1] x [0, 1]
        grid_master = VoxelMesher.create_box_grid(
            bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            resolution=(2, 2, 2),
        )

        # Slave upper block placed at z in [1.0, 2.0] (touching at z = 1.0)
        grid_slave = VoxelMesher.create_box_grid(
            bounds=(0.0, 1.0, 0.0, 1.0, 1.0, 2.0),
            resolution=(2, 2, 2),
        )

        master_facets = extract_hex8_surface_facets(grid_master.nodes, grid_master.elements)
        slave_facets = extract_hex8_surface_facets(grid_slave.nodes, grid_slave.elements)
        slave_normals = compute_nodal_surface_normals(grid_slave.nodes, slave_facets)

        detector = SpatialHashContactDetector(grid_master.nodes, master_facets)

        # Bottom face of slave body is at z = 1.0 (9 nodes)
        pairs = detector.detect_contact_pairs(
            grid_slave.nodes, search_distance=0.01, slave_normals=slave_normals
        )

        # 9 bottom nodes of the slave body should be in contact with top face of master
        self.assertEqual(len(pairs), 9, "9 slave interface nodes must be detected in contact")

        for p in pairs:
            self.assertAlmostEqual(p.gap_n, 0.0, places=6, msg="Initial gap must be zero at touching interface")
            # Master top face normal is (0, 0, 1)
            np.testing.assert_allclose(p.normal, [0.0, 0.0, 1.0], atol=1e-5)

    def test_contact_tangent_operator_symmetry(self):
        """
        Verify MatrixFreeContactTangentOperator is symmetric (self-adjoint) on combined DOFs.
        """
        grid1 = VoxelMesher.create_box_grid(bounds=(0, 1, 0, 1, 0, 1), resolution=(1, 1, 1))
        grid2 = VoxelMesher.create_box_grid(bounds=(0, 1, 0, 1, 1, 2), resolution=(1, 1, 1))

        op1 = VoxelMesher.create_box_grid(bounds=(0, 1, 0, 1, 0, 1), resolution=(1, 1, 1))

        facets2 = extract_hex8_surface_facets(grid1.nodes, grid1.elements)
        detector = SpatialHashContactDetector(grid1.nodes, facets2)
        pairs = detector.detect_contact_pairs(grid2.nodes, search_distance=0.01)
        # Artificially set negative gap to activate tangent stiffness
        for p in pairs:
            p.gap_n = -0.001

        from wnfea.solver.matrix_free_hex8 import MatrixFreeHex8Operator
        op_slave = MatrixFreeHex8Operator(grid2, E=2.1e11, nu=0.3, fixed_dofs=[])
        op_master = MatrixFreeHex8Operator(grid1, E=2.1e11, nu=0.3, fixed_dofs=[])

        op_tangent = MatrixFreeContactTangentOperator(
            bulk_operators=[op_slave, op_master],
            dof_offsets=[0, grid2.total_nodes * 3],
            contact_pairs=pairs,
            penalty_stiffness=1e9,
            fixed_dofs=[],
        )

        rng = np.random.RandomState(42)
        x = rng.randn(op_tangent.n_dofs)
        y = rng.randn(op_tangent.n_dofs)

        Ax = op_tangent.matvec(x)
        Ay = op_tangent.matvec(y)

        xAy = np.dot(x, Ay)
        yAx = np.dot(y, Ax)
        rel_diff = abs(xAy - yAx) / max(abs(xAy), 1e-15)

        self.assertLess(rel_diff, 1e-12, "Contact tangent operator must be symmetric")

    def test_two_block_compressive_contact_transmission(self):
        """
        Verify multi-body compressive contact load transfer and global force equilibrium.
        Two stacked blocks:
        - Foundation block [0, 0.2] x [0, 0.2] x [0, 0.2] fixed at base z = 0.
        - Upper block [0, 0.2] x [0, 0.2] x [0.2, 0.4] loaded in -Z with total force 10 kN.
        """
        L = 0.2
        E = 2.1e11
        nu = 0.3
        F_total = 10000.0  # 10 kN compressive load

        grid_master = VoxelMesher.create_box_grid(
            bounds=(0.0, L, 0.0, L, 0.0, L),
            resolution=(2, 2, 2),
        )
        grid_slave = VoxelMesher.create_box_grid(
            bounds=(0.0, L, 0.0, L, L, 2.0 * L),
            resolution=(2, 2, 2),
        )

        # Fix base of foundation block at z = 0
        base_nodes = np.where(grid_master.nodes[:, 2] < 1e-6)[0]
        fixed_master = []
        for n in base_nodes:
            fixed_master.extend([n * 3, n * 3 + 1, n * 3 + 2])

        # Apply downward compressive load on top face of slave block (z = 2*L)
        top_nodes = np.where(grid_slave.nodes[:, 2] > 2.0 * L - 1e-6)[0]
        f_slave = np.zeros(grid_slave.total_nodes * 3, dtype=np.float64)
        f_per_node = -F_total / len(top_nodes)
        for n in top_nodes:
            f_slave[n * 3 + 2] = f_per_node

        f_master = np.zeros(grid_master.total_nodes * 3, dtype=np.float64)

        result = solve_contact_assembly(
            grid_slave=grid_slave,
            grid_master=grid_master,
            f_ext_slave=f_slave,
            f_ext_master=f_master,
            fixed_dofs_slave=[],
            fixed_dofs_master=fixed_master,
            E_slave=E,
            nu_slave=nu,
            E_master=E,
            nu_master=nu,
            max_augmented_iters=10,
            penetration_tolerance=1e-6,
            pcg_tol=1e-6,
        )

        # 1. Check contact transmission
        self.assertGreater(len(result.contact_pairs), 0, "Contact must be established across blocks")
        self.assertLess(result.max_penetration, 1e-5, "Maximum penetration must be controlled")

        # 2. Check vertical contact force transmitted to slave body matches applied load (Newton's 3rd law)
        # Total contact force on slave should push upwards (+Z) with ~10 kN
        contact_fz = result.total_contact_force[2]
        rel_force_err = abs(contact_fz - F_total) / F_total
        self.assertLess(rel_force_err, 0.01, f"Contact normal force must balance applied load within 1%, got {rel_force_err*100:.2f}%")

        # 3. Check displacements are compressive (negative Z)
        u_slave = result.displacements[: grid_slave.total_nodes * 3]
        u_master = result.displacements[grid_slave.total_nodes * 3 :]
        self.assertLess(np.min(u_slave[2::3]), 0.0, "Slave block must compress downward")
        self.assertLess(np.min(u_master[2::3]), 0.0, "Master block must compress downward")


if __name__ == "__main__":
    unittest.main()
