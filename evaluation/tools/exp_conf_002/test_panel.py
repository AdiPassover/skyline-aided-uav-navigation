"""EXP-CONF-002: synthetic known-answer tests for the panel's mathematical transforms.

Constitution Principle XIII: descriptor/transform math gets tests. Runs with pytest from this
directory (same convention as evaluation/tools/exp_vo_007/test_analyse.py).
"""
from __future__ import annotations

import numpy as np
import pytest

from readout import (apply_h, circular_sd_deg, fit_h_ls, jacobian_at, readout,
                     symmetric_transfer_px)


def similarity(rot_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    c, s = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
    return np.array([[scale * c, -scale * s, tx], [scale * s, scale * c, ty], [0, 0, 1.0]])


class TestReadout:
    def test_known_similarity(self):
        H = similarity(5.0, 1.02, 7.0, -3.0)
        r = readout(H, 1224, 1024)
        assert r["proper"]
        assert r["rot_deg"] == pytest.approx(5.0, abs=1e-9)
        assert r["log_scale"] == pytest.approx(np.log(1.02), abs=1e-9)

    def test_identity_flow_zero(self):
        r = readout(np.eye(3), 1224, 1024)
        assert r["rot_deg"] == pytest.approx(0.0)
        assert r["flow_px"] == pytest.approx(0.0)

    def test_translation_flow(self):
        r = readout(similarity(0.0, 1.0, 30.0, 40.0), 1224, 1024)
        assert r["flow_px"] == pytest.approx(50.0, abs=1e-9)

    def test_improper_flagged(self):
        H = np.diag([1.0, -1.0, 1.0])   # reflection
        r = readout(H, 100, 100)
        assert not r["proper"]

    def test_jacobian_matches_numeric(self):
        rng = np.random.default_rng(1)
        H = np.eye(3) + rng.normal(0, 1e-3, (3, 3))
        H[2, 2] = 1.0
        x, y = 400.0, 300.0
        J = jacobian_at(H, x, y)
        eps = 1e-4
        for k, (dx, dy) in enumerate(((eps, 0.0), (0.0, eps))):
            num = (apply_h(H, np.array([[x + dx, y + dy]]))[0]
                   - apply_h(H, np.array([[x - dx, y - dy]]))[0]) / (2 * eps)
            assert np.allclose(J[:, k], num, atol=1e-6)


class TestSymmetricTransfer:
    def test_exact_homography_zero(self):
        rng = np.random.default_rng(2)
        H = similarity(2.0, 1.01, 5.0, 5.0)
        src = rng.uniform(0, 1000, (50, 2))
        dst = apply_h(H, src)
        assert np.max(symmetric_transfer_px(H, src, dst)) < 1e-9

    def test_perturbation_measured(self):
        rng = np.random.default_rng(3)
        H = np.eye(3)
        src = rng.uniform(0, 1000, (50, 2))
        dst = src + np.array([1.0, 0.0])   # 1 px offset in x
        d = symmetric_transfer_px(H, src, dst)
        assert np.allclose(d, 1.0, atol=1e-12)


class TestFitHLs:
    def test_recovers_known_h(self):
        rng = np.random.default_rng(4)
        H = similarity(3.0, 0.99, -12.0, 8.0)
        H[2, 0] = 1e-6                      # mild perspective
        src = rng.uniform(0, 1200, (100, 2))
        dst = apply_h(H, src)
        Hf = fit_h_ls(src, dst)
        assert np.max(symmetric_transfer_px(Hf, src, dst)) < 1e-6

    def test_noise_gives_small_error(self):
        rng = np.random.default_rng(5)
        H = similarity(1.0, 1.0, 4.0, 4.0)
        src = rng.uniform(0, 1200, (300, 2))
        dst = apply_h(H, src) + rng.normal(0, 0.3, (300, 2))
        Hf = fit_h_ls(src, dst)
        assert np.median(symmetric_transfer_px(Hf, src, dst)) < 0.6

    def test_too_few_points(self):
        assert fit_h_ls(np.zeros((3, 2)), np.zeros((3, 2))) is None


class TestCircularSd:
    def test_concentrated(self):
        assert circular_sd_deg(np.array([10.0, 10.1, 9.9])) < 0.2

    def test_wraparound(self):
        # 359.9 and 0.1 are 0.2 deg apart, not 359.8
        sd = circular_sd_deg(np.array([359.9, 0.1]))
        assert sd < 0.2


class TestBootstrapConditioning:
    """Concentrated support must yield larger readout dispersion than spread support."""

    def _boot_rot_sd(self, src: np.ndarray, rng) -> float:
        H = similarity(1.0, 1.0, 10.0, 0.0)
        dst = apply_h(H, src) + rng.normal(0, 0.5, src.shape)
        rots = []
        for _ in range(100):
            idx = rng.integers(0, len(src), len(src))
            Hb = fit_h_ls(src[idx], dst[idx])
            if Hb is not None:
                rots.append(readout(Hb, 1224, 1024)["rot_deg"])
        return circular_sd_deg(np.array(rots))

    def test_concentration_increases_dispersion(self):
        rng = np.random.default_rng(6)
        spread = rng.uniform(0, 1200, (200, 2))
        concentrated = rng.uniform(0, 120, (200, 2))   # 10% of the image
        sd_spread = self._boot_rot_sd(spread, rng)
        sd_conc = self._boot_rot_sd(concentrated, rng)
        assert sd_conc > 3 * sd_spread


class TestLkForwardBackward:
    """Synthetic shifted texture: FB error small, recovered flow equals the shift."""

    def test_pure_shift(self):
        cv2 = pytest.importorskip("cv2")
        rng = np.random.default_rng(7)
        base = rng.integers(0, 255, (400, 500), dtype=np.uint8)
        base = cv2.GaussianBlur(base, (7, 7), 2.0)
        shift = (6, 3)   # x, y
        cur = np.roll(np.roll(base, shift[1], axis=0), shift[0], axis=1)
        p0 = cv2.goodFeaturesToTrack(base, maxCorners=200, qualityLevel=0.01,
                                     minDistance=10, blockSize=7)
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01))
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(base, cur, p0, None, **lk)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur, base, p1, None, **lk)
        ok = (st1.ravel() == 1) & (st2.ravel() == 1)
        fb = np.linalg.norm(p0.reshape(-1, 2) - p0b.reshape(-1, 2), axis=1)[ok]
        flow = (p1.reshape(-1, 2) - p0.reshape(-1, 2))[ok]
        # interior points only (roll wraps at borders)
        inside = (p0.reshape(-1, 2)[ok][:, 0] > 40) & (p0.reshape(-1, 2)[ok][:, 0] < 460) \
                 & (p0.reshape(-1, 2)[ok][:, 1] > 40) & (p0.reshape(-1, 2)[ok][:, 1] < 360)
        assert np.median(fb[inside]) < 0.3
        assert np.allclose(np.median(flow[inside], axis=0), shift, atol=0.3)
