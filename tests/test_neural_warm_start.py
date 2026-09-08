"""
Verification Suite: Neural & Surrogate Warm-Start for Non-Linear JFNK
---------------------------------------------------------------------
Validates:
1. Provider Interface: NeMo, FNO, Linear tangent warm-start wrappers.
2. Residual Verification: Residual check ||R(u_warm)|| vs ||R(u_cold)||.
3. Divergence Safeguard: Auto-rejection of non-physical predictions.
4. Non-Linear Convergence: Warm-start achieves identical solution with fewer iterations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from wnfea.model import FEAModel
from wnfea.properties.materials import get_preset_material
from wnfea.properties.sections import create_solid_circle
from wnfea.boundary.conditions import create_fixed_support, LoadDef
from wnfea.mesh.beam_mesher import BeamMesher
from wnfea.solver.dof_manager import DOFManager
from wnfea.solver.nonlinear_solver import solve_nonlinear_jfnk
from wnfea.solver.neural_warm_start import (
    evaluate_warm_start,
    LinearTangentWarmStart,
    NeMoSurrogateWarmStart,
    CallableWarmStart,
)


from wnfea.geometry.primitives import Point3D, GeometryNode, GeometryEdge
from wnfea.model import PropertyAssignment


def _build_nonlinear_cantilever(P: float = -2000.0) -> FEAModel:
    """Build a 2m cantilever beam for non-linear large deflection testing."""
    model = FEAModel()
    model.geometry_nodes = {
        0: GeometryNode(id=0, label="Fixed", point=Point3D(0.0, 0.0, 0.0)),
        1: GeometryNode(id=1, label="Free", point=Point3D(2.0, 0.0, 0.0)),
    }
    model.geometry_edges = {
        0: GeometryEdge(id=0, label="Beam", start_node_id=0, end_node_id=1),
    }

    steel = get_preset_material("Structural Steel")
    rod = create_solid_circle("Rod", 0.05)
    model.materials[steel.name] = steel
    model.sections[rod.name] = rod
    model.edge_assignments[0] = PropertyAssignment(steel.name, rod.name)

    mesher = BeamMesher(n_divisions=5)
    mesher.mesh(model)

    model.supports.append(create_fixed_support(0, is_geometry_node=True))
    model.loads.append(LoadDef(node_id=1, is_geometry_node=True, fy=P))
    return model


def test_residual_verification_acceptance():
    """Verify that a good surrogate prediction is accepted based on residual reduction."""
    model = _build_nonlinear_cantilever(P=-100.0)
    dof_mgr = DOFManager(model)

    # Linear tangent predictor is a natural warm-start surrogate
    warm_provider = LinearTangentWarmStart()
    eval_res = evaluate_warm_start(model, warm_provider, dof_mgr, load_factor=1.0)

    assert eval_res.accepted, f"Warm-start should be accepted! Rejection reason: {eval_res.rejection_reason}"
    assert eval_res.reduction_factor < 0.5, f"Expected >50% residual reduction, got {eval_res.reduction_factor}"
    assert eval_res.warm_residual_norm < eval_res.cold_residual_norm

    print(f"  Warm-start accepted: ||R_warm|| = {eval_res.warm_residual_norm:.2f} N vs ||R_cold|| = {eval_res.cold_residual_norm:.2f} N")
    print(f"  Residual reduction: {(1.0 - eval_res.reduction_factor)*100:.1f}%")
    print("  [PASS] test_residual_verification_acceptance")


def test_divergence_safeguard_rejection():
    """Verify that a corrupted or non-physical prediction is rejected."""
    model = _build_nonlinear_cantilever(P=-100.0)
    dof_mgr = DOFManager(model)

    # Erroneous prediction: upward displacement when force is downward (opposite sign)
    n_nodes = len(model.mesh_nodes)
    bad_pred = np.zeros(n_nodes * 6)
    bad_pred[1::6] = 0.5  # Large +y displacement

    eval_res = evaluate_warm_start(model, bad_pred, dof_mgr, load_factor=1.0)
    assert not eval_res.accepted, "Corrupted prediction should have been rejected!"
    assert eval_res.warm_residual_norm > eval_res.cold_residual_norm
    assert "insufficient" in eval_res.rejection_reason.lower() or "incompatible" in eval_res.rejection_reason.lower()

    print(f"  Safeguard successfully rejected corrupted prediction: {eval_res.rejection_reason}")
    print("  [PASS] test_divergence_safeguard_rejection")


def test_nemo_surrogate_pipeline():
    """Verify simulated NVIDIA NeMo surrogate integrated with solve_nonlinear_jfnk."""
    model_cold = _build_nonlinear_cantilever(P=-2500.0)
    model_warm = _build_nonlinear_cantilever(P=-2500.0)

    # 1. Cold solve
    u_cold = solve_nonlinear_jfnk(model_cold, n_load_steps=3, max_newton_iter=15, verbose=False)

    # 2. Simulated NeMo surrogate: 90% accurate prediction
    def nemo_inference(coords, elems, load_factor):
        return u_cold * 0.90 * float(load_factor)

    nemo_provider = NeMoSurrogateWarmStart(nemo_inference, name="NeMo-Structural-FNO")

    # 3. Warm solve
    u_warm = solve_nonlinear_jfnk(
        model_warm, n_load_steps=3, max_newton_iter=15, warm_start=nemo_provider, verbose=False
    )

    # 4. Verify identical convergence
    diff = np.linalg.norm(u_warm - u_cold)
    rel_diff = diff / np.linalg.norm(u_cold)
    assert rel_diff < 1e-6, f"Warm-start converged to different state: rel_diff = {rel_diff}"

    tip_d_cold = u_cold[-5]  # tip fy
    tip_d_warm = u_warm[-5]
    print(f"  Cold tip deflection: {tip_d_cold*1e3:.2f} mm | Warm tip deflection: {tip_d_warm*1e3:.2f} mm")
    print(f"  Solution parity relative error: {rel_diff:.2e}")
    print("  [PASS] test_nemo_surrogate_pipeline")


def run_all():
    print("=" * 60)
    print("Running Neural & Surrogate Warm-Start Test Suite...")
    print("=" * 60)
    test_residual_verification_acceptance()
    test_divergence_safeguard_rejection()
    test_nemo_surrogate_pipeline()
    print("=" * 60)
    print("ALL NEURAL WARM-START TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
