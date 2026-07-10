"""build_shared_axis_from_axes: shape / unit-norm / mean & intermediate geometry.

Synthetic arrays only — no HF / sae_lens / model. The shared axis is the
Wave-2 COMMON-AXIS candidate: the SVD top-1 direction two per-indicator
behavioural axes most share.
"""
import numpy as np

from multilayer_causal.src import indicator_axes as ia


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def test_shape_and_unit_norm():
    rs = np.random.RandomState(0)
    a = np.stack([_unit(rs.randn(8)) for _ in range(3)])  # (L=3, d=8)
    b = np.stack([_unit(rs.randn(8)) for _ in range(3)])
    out = ia.build_shared_axis_from_axes([a, b], layers=[16, 18, 21])
    assert out["directions"].shape == (3, 8)
    assert np.allclose(np.linalg.norm(out["directions"], axis=1), 1.0, atol=1e-5)
    assert out["singular_energy"].shape == (3,)
    assert "provenance" in out


def test_nearly_identical_axes_shared_is_their_mean():
    """Two near-identical indicator axes => shared ~ their mean, and the shared
    component captures almost all the variance (singular_energy ~ 1)."""
    rs = np.random.RandomState(1)
    d = 16
    base = _unit(rs.randn(d))
    tilt = _unit(rs.randn(d))
    a_row = base
    b_row = _unit(base + 0.05 * tilt)
    a = a_row[None, :]  # (L=1, d)
    b = b_row[None, :]
    out = ia.build_shared_axis_from_axes([a, b], layers=[0])
    shared = _unit(out["directions"][0])
    mean = _unit(a_row + b_row)
    assert abs(float(shared @ mean)) > 0.99
    assert abs(float(shared @ a_row)) > 0.99
    assert abs(float(shared @ b_row)) > 0.99
    assert out["singular_energy"][0] > 0.99


def test_orthogonal_axes_shared_is_intermediate():
    """Orthogonal indicator axes => shared is equidistant (intermediate cos to
    each), NOT collapsed onto either, and singular_energy ~ 0.5 (no dominant
    shared component)."""
    rs = np.random.RandomState(2)
    q, _ = np.linalg.qr(rs.randn(12, 2))
    a_row, b_row = q[:, 0], q[:, 1]
    assert abs(float(a_row @ b_row)) < 1e-8  # genuinely orthogonal
    out = ia.build_shared_axis_from_axes([a_row[None, :], b_row[None, :]],
                                         layers=[0])
    shared = _unit(out["directions"][0])
    cos_a = abs(float(shared @ a_row))
    cos_b = abs(float(shared @ b_row))
    assert 0.3 < cos_a < 0.95, cos_a
    assert 0.3 < cos_b < 0.95, cos_b
    assert abs(cos_a - cos_b) < 0.1, (cos_a, cos_b)  # equidistant
    assert abs(out["singular_energy"][0] - 0.5) < 0.05


def test_orientation_is_sign_invariant_to_input_flips():
    """Flipping an indicator axis sign must not change the shared axis (up to a
    global sign) — the builder orients rows to their centroid first."""
    rs = np.random.RandomState(3)
    base = _unit(rs.randn(10))
    a_row = _unit(base + 0.1 * _unit(rs.randn(10)))
    b_row = _unit(base + 0.1 * _unit(rs.randn(10)))
    out1 = ia.build_shared_axis_from_axes([a_row[None, :], b_row[None, :]], [0])
    out2 = ia.build_shared_axis_from_axes([-a_row[None, :], b_row[None, :]], [0])
    cos = abs(float(_unit(out1["directions"][0]) @ _unit(out2["directions"][0])))
    assert cos > 0.999, cos
