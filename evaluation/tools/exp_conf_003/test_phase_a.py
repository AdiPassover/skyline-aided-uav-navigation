"""Synthetic known-answer tests for the EXP-CONF-003 Phase A decomposition.

Run with pytest from this directory.
"""
from __future__ import annotations

import numpy as np
import pytest

from motion_decomp import DIR_MIN_FLOW_PX, decompose, similarity_approx

W, H = 2448, 2048
C = np.array([W / 2.0, H / 2.0])


def sim_h(rot_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    th = np.radians(rot_deg)
    S = np.eye(3)
    S[:2, :2] = scale * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S[:2, 2] = [tx, ty]
    return S


def proj_h(base: np.ndarray, px: float, py: float) -> np.ndarray:
    """base perturbed by raw perspective-row terms. NOTE: this also perturbs the centre
    Jacobian, so it is NOT invisible to the rigid readout (verified by
    test_raw_perspective_terms_leak_into_the_centre_readout)."""
    P = base.copy()
    P[2, 0] += px
    P[2, 1] += py
    return P


def proj_h_centre_neutral(base: np.ndarray, px: float, py: float) -> np.ndarray:
    """base composed with a projective perturbation that has a fixed point AND identity
    Jacobian at the image centre: P0(x) = x / (1 + p.x) conjugated by the centre translation.
    Invisible to the centre readout by construction."""
    T = np.eye(3); T[:2, 2] = C
    Ti = np.eye(3); Ti[:2, 2] = -C
    P0 = np.eye(3); P0[2, 0] = px; P0[2, 1] = py
    return base @ T @ P0 @ Ti


class TestSimilarityApprox:
    def test_similarity_is_its_own_approximation(self):
        Hm = sim_h(3.0, 1.01, 40.0, -25.0)
        S, ro = similarity_approx(Hm, W, H)
        assert np.allclose(S, Hm, atol=1e-9)
        assert ro["rot_deg"] == pytest.approx(3.0, abs=1e-9)

    def test_projective_part_is_removed(self):
        Hm = proj_h(sim_h(1.0, 1.0, 30.0, 10.0), 4e-6, -3e-6)
        S, _ = similarity_approx(Hm, W, H)
        assert abs(S[2, 0]) < 1e-12 and abs(S[2, 1]) < 1e-12
        # the approximation still matches H exactly at the centre
        hc = (Hm @ np.array([*C, 1.0]))
        hc = hc[:2] / hc[2]
        sc = (S @ np.array([*C, 1.0]))[:2]
        assert np.allclose(sc, hc, atol=1e-9)


class TestDecompose:
    def test_pure_similarity_disagreement_has_no_projective_part(self):
        Ha = sim_h(2.0, 1.00, 50.0, 0.0)
        Hb = sim_h(1.2, 1.01, 44.0, 6.0)
        d = decompose(Ha, Hb, W, H)
        assert d["rot_dis_deg"] == pytest.approx(0.8, abs=1e-9)
        assert d["d_proj_px"] == pytest.approx(0.0, abs=1e-9)
        assert d["d_sim_px"] == pytest.approx(d["d_tot_px"], abs=1e-9)
        assert d["nonrigid_a_px"] == pytest.approx(0.0, abs=1e-9)

    def test_pure_projective_disagreement_has_no_similarity_part(self):
        base = sim_h(1.5, 1.0, 40.0, -20.0)
        Ha = proj_h_centre_neutral(base, 8e-6, -6e-6)
        d = decompose(Ha, base, W, H)
        # identical rigid readout at the centre, so the navigation-consumed part vanishes
        assert d["rot_dis_deg"] < 1e-9
        assert d["flowvec_dis_px"] < 1e-6
        assert d["d_sim_px"] < 1e-6
        assert d["d_proj_px"] > 3.0            # but the image-space disagreement is large
        assert d["d_tot_px"] == pytest.approx(d["d_proj_px"], abs=1e-6)
        assert d["nonrigid_a_px"] > 3.0 and d["nonrigid_b_px"] == pytest.approx(0.0, abs=1e-9)

    def test_raw_perspective_terms_leak_into_the_centre_readout(self):
        """Documents why proj_h_centre_neutral exists: raw h20/h21 perturbation changes the
        centre Jacobian, so the rigid readout (and hence the navigator) sees part of it."""
        base = sim_h(1.5, 1.0, 40.0, -20.0)
        d = decompose(proj_h(base, 3e-6, -2e-6), base, W, H)
        assert d["rot_dis_deg"] > 0.05
        assert d["d_sim_px"] > 1.0

    def test_flow_vector_and_direction(self):
        Ha = sim_h(0.0, 1.0, 30.0, 0.0)
        Hb = sim_h(0.0, 1.0, 0.0, 30.0)
        d = decompose(Ha, Hb, W, H)
        assert d["dir_dis_deg"] == pytest.approx(90.0, abs=1e-9)
        assert d["flowvec_dis_px"] == pytest.approx(np.hypot(30.0, 30.0), abs=1e-9)
        assert d["flowmag_dis_px"] == pytest.approx(0.0, abs=1e-9)

    def test_direction_guard_below_min_flow(self):
        Ha = sim_h(0.0, 1.0, DIR_MIN_FLOW_PX * 0.4, 0.0)
        Hb = sim_h(0.0, 1.0, 0.0, 30.0)
        d = decompose(Ha, Hb, W, H)
        assert np.isnan(d["dir_dis_deg"])

    def test_h63_shape_large_projective_small_rigid(self):
        """The H6.3 signature by construction: transforms sharing rigid motion but with a
        projective difference produce large d_tot dominated by d_proj, not d_sim."""
        base = sim_h(1.0, 1.001, 35.0, -10.0)
        Ha = proj_h_centre_neutral(base, 1.5e-5, 1.2e-5)
        Hb = proj_h_centre_neutral(base, -8e-6, -6e-6)
        d = decompose(Ha, Hb, W, H)
        assert d["d_tot_px"] > 8.0
        assert d["d_proj_px"] > 0.99 * d["d_tot_px"]
        assert d["d_sim_px"] < 0.01 * d["d_tot_px"]
        assert d["rot_dis_deg"] < 1e-9
