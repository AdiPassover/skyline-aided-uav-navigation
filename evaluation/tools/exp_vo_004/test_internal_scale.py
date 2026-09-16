"""Known-answer tests for the EXP-VO-004 driver's own geometry: scale/Jacobian observables,
normalisation about the image centre, and the re-composition loop. Run alone::

    cd evaluation; python -m pytest tools/exp_vo_004 -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import internal_scale as isc  # noqa: E402

W, H = 1224, 1024


def sim(scale, deg, tx, ty):
    th = math.radians(deg)
    return np.array([[scale * math.cos(th), -scale * math.sin(th), tx],
                     [scale * math.sin(th), scale * math.cos(th), ty],
                     [0, 0, 1.0]])


def test_similarity_G_gives_reciprocal_magnification_and_zero_perspective():
    # G : L -> C scales by 0.5 => footprint magnification m = 2 (current px cover 2 logical px)
    o = isc.observables(sim(0.5, 30, 100, -40), W, H)
    assert math.isclose(o["m_centre"], 2.0, rel_tol=1e-12)
    assert math.isclose(o["m_area"], 2.0, rel_tol=1e-12)
    assert math.isclose(o["anisotropy"], 1.0, rel_tol=1e-12)
    assert o["perspective"] == 0.0
    assert o["det_sign"] == 1.0


def test_pure_rotation_about_centre_moves_centre_zero_and_reads_yaw():
    cx, cy = W / 2, H / 2
    th = math.radians(30)
    R = sim(1.0, 30, 0, 0)
    T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]]); Ti = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    G = T @ R @ Ti
    o = isc.observables(G, W, H)
    assert abs(o["logical_x"]) < 1e-9 and abs(o["logical_y"]) < 1e-9
    # M = G^-1 rotates by -30 deg; top edge of the footprint then has angle -30 -> 330
    assert math.isclose(o["yaw_edge_deg"], 330.0, abs_tol=1e-9)
    assert math.isclose(o["yaw_polar_deg"], 330.0, abs_tol=1e-9)
    assert math.isclose(o["m_centre"], 1.0, rel_tol=1e-12)
    del th


def test_homography_perspective_is_reported_and_centre_jacobian_differs_from_area():
    G = np.array([[1.0, 0, 0], [0, 1.0, 0], [2e-4, 0, 1.0]])
    o = isc.observables(G, W, H)
    assert math.isclose(o["perspective"], 2e-4 * W, rel_tol=1e-12)
    assert abs(o["m_area"] - o["m_centre"]) > 1e-3


def test_normalise_scale_only_keeps_centre_fixed_and_unit_scale():
    M = sim(3.0, 20, 50, 60)
    Mn = isc.normalise(M, W / 2, H / 2, "scale_only")
    assert np.allclose(isc.apply(Mn, W / 2, H / 2), isc.apply(M, W / 2, H / 2))
    J = isc.jacobian_at(Mn, W / 2, H / 2)
    assert math.isclose(math.sqrt(abs(np.linalg.det(J))), 1.0, rel_tol=1e-12)
    assert math.isclose(isc.polar_rotation_deg(J), 20.0, abs_tol=1e-9)


def test_normalise_se2_removes_shear_and_perspective_keeps_rotation_and_centre():
    M = np.array([[2.0, 0.6, 30], [-0.3, 1.5, 40], [1e-4, -2e-4, 1.0]])
    Mn = isc.normalise(M, W / 2, H / 2, "se2")
    assert np.allclose(isc.apply(Mn, W / 2, H / 2), isc.apply(M, W / 2, H / 2))
    J = isc.jacobian_at(Mn, W / 2, H / 2)
    sv = np.linalg.svd(J, compute_uv=False)
    assert np.allclose(sv, 1.0)
    assert math.isclose(isc.polar_rotation_deg(J), isc.polar_rotation_deg(isc.jacobian_at(M, W / 2, H / 2)), abs_tol=1e-9)
    assert Mn[2, 0] == 0 and Mn[2, 1] == 0


def test_recompose_without_normalisation_reproduces_logical_readout():
    rng = np.random.default_rng(0)
    G = [np.eye(3)]
    for _ in range(30):
        D = sim(1 + rng.normal(0, 0.01), rng.normal(0, 2), rng.normal(0, 3), rng.normal(0, 3))
        G.append(D @ G[-1])
    G = np.array(G)
    ev = ["none"] * len(G)
    rec = isc.recompose(G, ev, W, H, lambda k, e: False, "se2")
    for k in range(len(G)):
        o = isc.observables(G[k], W, H)
        assert math.isclose(rec[k, 0], o["logical_x"], abs_tol=1e-6)
        assert math.isclose(rec[k, 1], o["logical_y"], abs_tol=1e-6)
        assert math.isclose(rec[k, 3], o["m_centre"], rel_tol=1e-9)


def test_recompose_with_every_frame_se2_removes_a_pure_scale_drift():
    # Pure zoom per frame about the centre: the logical position must stay at 0 with or without
    # normalisation, and with every-frame SE(2) the readout scale stays exactly 1.
    cx, cy = W / 2, H / 2
    T = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1.0]]); Ti = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    G = [np.eye(3)]
    for _ in range(20):
        G.append(T @ sim(0.99, 0, 0, 0) @ Ti @ G[-1])
    G = np.array(G)
    rec = isc.recompose(G, ["none"] * len(G), W, H, lambda k, e: k > 0, "se2")
    assert np.all(np.abs(rec[:, :2]) < 1e-9)
    assert np.allclose(rec[:, 3], 1.0)
    raw = isc.recompose(G, ["none"] * len(G), W, H, lambda k, e: False, "se2")
    assert math.isclose(raw[-1, 3], 1 / 0.99 ** 20, rel_tol=1e-9)


def test_lag_variance_exponent_distinguishes_walk_from_drift():
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(0, 1, 20000))
    drift = np.arange(20000) * 1e-3 + rng.normal(0, 1e-3, 20000)
    assert abs(isc.lag_variance_exponent(walk, 1, 500)["log_log_slope"] - 1.0) < 0.15
    assert isc.lag_variance_exponent(drift, 1, 500)["log_log_slope"] > 1.9
