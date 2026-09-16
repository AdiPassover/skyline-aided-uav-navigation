"""Known-answer tests for the relative-position geometry helpers (EXP-SKY-009).

All T1 synthetic: about the *mechanism* (does the algebra recover a displacement it was built to
recover; is the vectorised lag search the frozen matcher), never about skylines.
"""

import math

import numpy as np
import pytest

from hsreloc.matchers import BoundedLagNccMatcher
from hsreloc.simret import relpose

CAM = relpose.ProfileCamera(width_px=512, height_px=512, fx=256.0, fy=256.0, cx=256.0, cy=256.0, n_samples=256)


def _profile(seed=0, n=256):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    p = (np.sin(2 * np.pi * x / 37.0) + 0.5 * np.sin(2 * np.pi * x / 11.0 + 1.0)
         + 0.3 * rng.standard_normal(n))
    return p - p.mean()


class TestCameraMapping:
    def test_centre_sample_is_on_axis_and_edges_reach_half_fov(self):
        # the profile's centre sample sits at column (W − 1) / 2 = 255.5, half a pixel left of cx
        assert abs(float(CAM.sample_to_azimuth_rad(127.5))) < 0.002
        assert math.degrees(float(CAM.sample_to_azimuth_rad(0))) == pytest.approx(-45.0, abs=0.2)
        assert math.degrees(float(CAM.sample_to_azimuth_rad(255))) == pytest.approx(45.0, abs=0.2)

    def test_positive_lag_means_feature_moved_left_i_e_negative_azimuth_shift(self):
        # q[i] = r[i + δ]: the feature the reference saw further right is now on axis → it moved left.
        assert float(CAM.lag_to_azimuth_shift_rad(127.5, +4.0)) < 0
        # dx/dα = fx·sec²α: near the edge the same pixel lag is worth FEWER degrees than on axis
        assert abs(float(CAM.lag_to_azimuth_shift_rad(250.0, 4.0))) < abs(float(CAM.lag_to_azimuth_shift_rad(127.5, 4.0)))


class TestBoundedLagSearchEqualsFrozenC1:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize("true_shift", [-20, -3, 0, 7, 31])
    def test_winner_and_score_bit_equal(self, seed, true_shift):
        r = _profile(seed)
        q = np.roll(r, -true_shift) + 0.05 * np.random.default_rng(seed + 100).standard_normal(r.size)
        frozen = BoundedLagNccMatcher(max_lag_samples=32, min_overlap_frac=0.6).match(q, r)
        ours = relpose.bounded_lag_search(q, r, max_lag=32, min_overlap_frac=0.6)
        assert ours.lag == frozen.shift
        assert ours.score == pytest.approx(frozen.score, abs=1e-12)
        assert ours.overlap == frozen.overlap
        assert ours.score_at_zero == pytest.approx(frozen.diagnostics["score_at_zero_lag"], abs=1e-12)

    def test_peak_diagnostics_are_retained(self):
        r = _profile(5)
        q = np.roll(r, -6)
        res = relpose.bounded_lag_search(q, r)
        assert res.lag == 6.0 and res.score == pytest.approx(1.0)
        assert res.secondary_margin > 0 and res.curvature > 0
        assert not res.saturated
        assert res.lags.size == 65 and np.isfinite(res.scores).sum() == res.n_searched

    def test_saturation_flag(self):
        # an aperiodic random walk shifted beyond the bound: the best admissible lag sits on the bound
        rng = np.random.default_rng(6)
        r = rng.standard_normal(256).cumsum()
        r -= r.mean()
        q = np.roll(r, -40)
        res = relpose.bounded_lag_search(q, r, max_lag=32)
        assert res.saturated and abs(res.lag) == 32.0


class TestWindowedLags:
    def test_uniform_shift_recovered_in_every_window(self):
        r = _profile(7)
        q = np.roll(r, -5)
        wins = relpose.windowed_lag_search(q, r, n_windows=8, max_lag=16)
        assert len(wins) == 8
        assert all(w["lag"] == 5.0 for w in wins), [w["lag"] for w in wins]

    def test_flat_window_is_not_scored(self):
        r = _profile(8)
        q = r.copy()
        q[:32] = 0.0
        wins = relpose.windowed_lag_search(q, r, n_windows=8, max_lag=8)
        assert wins[0]["flat"] and math.isnan(wins[0]["score"])
        assert not wins[1]["flat"]


class TestRangeFromVerticalParallax:
    def test_known_range_recovered(self):
        rows_ref = np.full(512, 200.0)
        D = np.linspace(500.0, 3000.0, 512)
        dz = 50.0
        rows_q = rows_ref + 256.0 * dz / D          # higher query → boundary lower in the frame
        est = relpose.range_from_vertical_parallax(rows_ref, rows_q, dz, fy=256.0)
        np.testing.assert_allclose(est, D, rtol=1e-9)

    def test_small_dz_and_wrong_sign_refused(self):
        rows = np.full(512, 200.0)
        assert np.isnan(relpose.range_from_vertical_parallax(rows, rows + 3.0, 5.0, fy=256.0)).all()
        assert np.isnan(relpose.range_from_vertical_parallax(rows, rows - 3.0, 50.0, fy=256.0)).all()

    def test_window_ranges_median(self):
        R = np.concatenate([np.full(256, 800.0), np.full(256, np.nan)])
        w = relpose.window_ranges(R, CAM, 8)
        assert w[0] == 800.0 and np.isnan(w[-1])


class TestSolveTranslation:
    @pytest.mark.parametrize("dE,dN", [(30.0, 0.0), (0.0, 40.0), (-25.0, 60.0), (80.0, -70.0)])
    def test_exact_first_order_pair_recovered_with_known_range(self, dE, dN):
        az = np.radians(np.linspace(-40, 40, 8))
        D = np.array([800, 1200, 900, 1500, 1500, 1000, 700, 2000.0])
        shifts = relpose.predicted_azimuth_shift_rad(dE, dN, az, D)
        est = relpose.solve_translation(az, shifts, D)
        assert est.valid
        assert est.delta_east_m == pytest.approx(dE, abs=1e-9)
        assert est.delta_north_m == pytest.approx(dN, abs=1e-9)
        assert est.residual_rad == pytest.approx(0.0, abs=1e-12)

    def test_too_few_windows_refuses(self):
        est = relpose.solve_translation([0.1, 0.2], [0.001, 0.002], [1000, 1000])
        assert not est.valid and est.refusal.startswith("too_few")

    def test_residual_gate_refuses(self):
        az = np.radians(np.linspace(-40, 40, 8))
        D = np.full(8, 1000.0)
        shifts = relpose.predicted_azimuth_shift_rad(30.0, 0.0, az, D)
        shifts[3] += 0.05                                     # one badly wrong window (≈2.9°)
        est = relpose.solve_translation(az, shifts, D, max_residual_rad=0.005)
        assert not est.valid and est.refusal == "residual_too_large"

    def test_global_lag_model_gives_east_only(self):
        # a lag of +4 samples at the centre ≈ 8 px ≈ 0.03125 rad at fx=256 → ΔE = D·Δα
        est = relpose.estimate_global_lag(4.0, CAM, 1000.0)
        assert est.valid and est.delta_north_m == 0.0
        expected = -1000.0 * float(CAM.lag_to_azimuth_shift_rad(127.5, 4.0))
        assert est.delta_east_m == pytest.approx(expected, rel=1e-9)
        assert est.delta_east_m > 0
        # ≈ 8 px at fx = 256 → ≈ 31 m at 1 km
        assert est.delta_east_m == pytest.approx(31.3, abs=0.5)
        assert not relpose.estimate_global_lag(32.0, CAM, 1000.0, saturated=True).valid


class TestEndToEndSyntheticWindows:
    def test_windowed_pipeline_recovers_translation_from_a_warped_profile(self):
        """Warp a rich reference profile by equation (1) with a known range map, then run the
        windowed lag search + solver: this is the mechanism the study applies to real curves."""
        rng = np.random.default_rng(11)
        n = 256
        cols = np.arange(512, dtype=np.float64)
        # a reference skyline in ROW space with structure at every scale
        rows_ref = 220 + 25 * np.sin(2 * np.pi * cols / 90) + 12 * np.sin(2 * np.pi * cols / 23 + 0.7) \
            + 5 * rng.standard_normal(512).cumsum() / 8
        D_col = 900.0 + 600.0 * (1 + np.sin(2 * np.pi * cols / 300))      # 300–2100 m
        dE, dN = 40.0, -30.0
        az_col = np.arctan((cols - 256.0) / 256.0)
        d_az = relpose.predicted_azimuth_shift_rad(dE, dN, az_col, D_col)
        # the query sees the feature that was at az at az + d_az: sample the reference at the source column
        src_cols = 256.0 + 256.0 * np.tan(az_col - d_az)
        rows_q = np.interp(src_cols, cols, rows_ref)
        from hsreloc.retrieval.skyline_curve import make_curve
        cq = make_curve("q", rows_q, 512, 512, "oracle:sim_exact", "t")
        cr = make_curve("r", rows_ref, 512, 512, "oracle:sim_exact", "t")
        pq, pr = relpose.profile_of(cq), relpose.profile_of(cr)
        wins = relpose.windowed_lag_search(pq, pr, n_windows=8, max_lag=32)
        Dw = relpose.window_ranges(D_col, CAM, 8)
        est = relpose.estimate_from_windows(wins, CAM, Dw)
        assert est.valid
        # integer-lag quantisation (~2 px ≈ 0.45° on axis) bounds the accuracy; require ≲ 15 m
        assert est.delta_east_m == pytest.approx(dE, abs=15.0)
        assert est.delta_north_m == pytest.approx(dN, abs=15.0)


class TestMetrics:
    def test_error_stats_and_snap_comparison_shapes(self):
        s = relpose.error_stats(np.array([1.0, 2.0, 50.0]))
        assert s["n"] == 3 and s["frac_over_catastrophic"] == pytest.approx(1 / 3)
        cmp = relpose.snap_comparison(np.array([10.0, 60.0]), np.array([5.0, 80.0]), [20.0])
        assert cmp["frac_refined_better"] == 0.5
        assert cmp["harmful_rate_vs_e_before"]["20"]["snap"] == 0.5
        assert cmp["harmful_rate_vs_e_before"]["20"]["refined"] == 0.5


class TestProposedInterface:
    def _pair(self, dE=25.0, dN=-15.0, seed=3):
        rng = np.random.default_rng(seed)
        cols = np.arange(512, dtype=np.float64)
        rows_ref = 220 + 25 * np.sin(2 * np.pi * cols / 90) + 12 * np.sin(2 * np.pi * cols / 23 + 0.7) \
            + 5 * rng.standard_normal(512).cumsum() / 8
        D_col = np.full(512, 1000.0)
        az = np.arctan((cols - 256.0) / 256.0)
        d_az = relpose.predicted_azimuth_shift_rad(dE, dN, az, D_col)
        rows_q = np.interp(256.0 + 256.0 * np.tan(az - d_az), cols, rows_ref)
        from hsreloc.retrieval.skyline_curve import make_curve
        pq = relpose.profile_of(make_curve("q", rows_q, 512, 512, "oracle:sim_exact", "t"))
        pr = relpose.profile_of(make_curve("r", rows_ref, 512, 512, "oracle:sim_exact", "t"))
        return pq, pr

    def test_valid_estimate_exposes_validity_variables_not_a_confidence(self):
        pq, pr = self._pair()
        e = relpose.relative_position_estimate("ref", "q", pq, pr, CAM, np.full(8, 1000.0), "reference_range_map")
        assert e.valid and e.refusal is None
        assert e.delta_east_m == pytest.approx(25.0, abs=15.0)
        assert e.delta_north_m == pytest.approx(-15.0, abs=15.0)
        assert e.c1_score > 0.9 and not e.c1_lag_saturated
        assert e.n_windows_used >= 3 and e.fit_residual_rad is not None and e.fit_residual_rad <= 0.01
        assert e.range_source == "reference_range_map" and e.range_median_m == 1000.0
        d = e.as_dict()
        assert "confidence" not in d and "heading" not in d and "yaw" not in d

    def test_no_range_refuses_rather_than_guessing(self):
        pq, pr = self._pair()
        e = relpose.relative_position_estimate("ref", "q", pq, pr, CAM, np.full(8, np.nan), "reference_range_map")
        assert not e.valid and e.refusal.startswith("too_few_windows")
        assert e.delta_east_m is None and e.delta_north_m is None
        assert e.c1_score > 0.9                                   # the match itself was fine

    def test_low_c1_score_refuses_first(self):
        pq, pr = self._pair()
        unrelated = _profile(99)
        e = relpose.relative_position_estimate("ref", "q", unrelated, pr, CAM, 1000.0, "scalar_prior")
        assert not e.valid and e.refusal == "c1_score_below_gate"

    def test_unknown_range_source_is_an_error(self):
        pq, pr = self._pair()
        with pytest.raises(relpose.RelPoseError):
            relpose.relative_position_estimate("ref", "q", pq, pr, CAM, 1000.0, "guess")
