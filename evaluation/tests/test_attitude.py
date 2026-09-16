"""Tests for `naveval.attitude` and the EXP-003 attitude ingest extension.

All T1 synthetic: they validate the *analysis instrument*, not any finding about the estimator.
The statistical machinery is tested against constructed series with known answers -- in
particular the autocorrelation defences are tested by feeding them data that would fool a naive
correlation, which is the specific failure mode EXP-003 is built to avoid.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from naveval import mcap_reader as mcap
from naveval.attitude import (
    AttitudeSeries,
    apply_holm,
    attitude_distribution,
    check_ingest_gates,
    circular_shift_null,
    decorrelation_block_length,
    effective_sample_size,
    holm_adjust,
    interpolate_to,
    lag1_autocorr,
    load_attitude_csv,
    moving_block_bootstrap_ci,
    pearson_r,
    permutation_pvalue,
    rankdata,
    relative_spread,
    run_correlation_test,
    spearman_rho,
    tercile_spread_analysis,
    tilt_rate_dps,
    write_correlations_csv,
)
from naveval.ingest_mars_lvig import (
    ATTITUDE_CSV_HEADER,
    AttitudeSample,
    IngestError,
    slice_attitude,
    write_attitude_sidecar,
)


# ------------------------------------------------------------------------------------------
# Quaternion -> roll/pitch (mcap_reader extension)
# ------------------------------------------------------------------------------------------

def _quat_from_rpy(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """ZYX intrinsic (yaw-pitch-roll) euler -> quaternion, the inverse of what the reader does."""
    r, p, y = (math.radians(v) / 2.0 for v in (roll_deg, pitch_deg, yaw_deg))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (
        sr * cp * cy - cr * sp * sy,   # x
        cr * sp * cy + sr * cp * sy,   # y
        cr * cp * sy - sr * sp * cy,   # z
        cr * cp * cy + sr * sp * sy,   # w
    )


class TestQuaternionToRollPitch:
    def test_identity_quaternion_is_level(self):
        roll, pitch = mcap.quaternion_to_roll_pitch_deg(0.0, 0.0, 0.0, 1.0)
        assert roll == pytest.approx(0.0, abs=1e-9)
        assert pitch == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("roll_deg,pitch_deg,yaw_deg", [
        (0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (9.06, -6.10, 137.0),
        (-4.56, 6.66, 310.0), (2.15, 0.60, 45.0), (-15.0, 12.0, 200.0),
    ])
    def test_round_trip_recovers_injected_roll_and_pitch(self, roll_deg, pitch_deg, yaw_deg):
        """The decisive test: build a quaternion from known angles, decode it, get them back.
        Covers the LIT-006 observed extremes (roll 9.06, pitch -6.10) and all four yaw quadrants,
        because a convention error typically shows up as a sign flip in only some quadrants."""
        x, y, z, w = _quat_from_rpy(roll_deg, pitch_deg, yaw_deg)
        got_roll, got_pitch = mcap.quaternion_to_roll_pitch_deg(x, y, z, w)
        assert got_roll == pytest.approx(roll_deg, abs=1e-6)
        assert got_pitch == pytest.approx(pitch_deg, abs=1e-6)

    def test_yaw_is_unaffected_by_roll_and_pitch_extraction(self):
        """Roll/pitch extraction must not disturb the already-verified compass conversion."""
        x, y, z, w = _quat_from_rpy(7.0, -3.0, 137.0)
        compass = mcap.quaternion_to_compass_deg(x, y, z, w)
        assert 0.0 <= compass < 360.0

    def test_pitch_is_clamped_at_gimbal_lock_rather_than_raising(self):
        """asin's argument can exceed 1 by an epsilon for a near-vertical attitude; the guard
        must return +/-90 rather than raise a domain error."""
        x, y, z, w = _quat_from_rpy(0.0, 90.0, 0.0)
        _, pitch = mcap.quaternion_to_roll_pitch_deg(x, y, z, w)
        assert pitch == pytest.approx(90.0, abs=1e-6)


class TestTiltMagnitude:
    def test_level_is_zero(self):
        assert mcap.tilt_from_roll_pitch_deg(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_pure_roll_equals_roll_magnitude(self):
        assert mcap.tilt_from_roll_pitch_deg(9.0, 0.0) == pytest.approx(9.0, abs=1e-9)

    def test_pure_pitch_equals_pitch_magnitude(self):
        assert mcap.tilt_from_roll_pitch_deg(0.0, -6.0) == pytest.approx(6.0, abs=1e-9)

    def test_sign_does_not_matter(self):
        assert mcap.tilt_from_roll_pitch_deg(-5.0, 3.0) == pytest.approx(
            mcap.tilt_from_roll_pitch_deg(5.0, -3.0))

    def test_agrees_with_small_angle_approximation_at_this_dataset_s_magnitudes(self):
        """acos(cos r cos p) vs sqrt(r^2+p^2) at this dataset's magnitudes. The documented bound
        is <0.02 deg, and the worst case is the observed extreme (roll 9.06, pitch 6.66) at
        0.0165 deg. Pinned so the claim in the docstring and in EXP-003 stays true rather than
        being asserted only in prose -- the first version of this test said 0.01 deg and caught
        that the prose was wrong."""
        worst = max(
            abs(mcap.tilt_from_roll_pitch_deg(r, p) - math.hypot(r, p))
            for r, p in [(9.06, 6.66), (5.0, 4.0), (2.15, 0.60)]
        )
        assert worst < 0.02


# ------------------------------------------------------------------------------------------
# Sidecar round-trip
# ------------------------------------------------------------------------------------------

class TestAttitudeSidecar:
    def _samples(self, n=5, t0=1000.0):
        return [
            AttitudeSample(timestamp_s=t0 + i * 0.01, roll_deg=1.0 * i, pitch_deg=-0.5 * i,
                           tilt_deg=float(i), yaw_compass_deg=130.0 + i)
            for i in range(n)
        ]

    def test_write_then_load_round_trips(self, tmp_path: Path):
        write_attitude_sidecar(tmp_path, self._samples())
        series = load_attitude_csv(tmp_path / "attitude.csv")
        assert len(series) == 5
        assert series.roll_deg[2] == pytest.approx(2.0)
        assert series.pitch_deg[2] == pytest.approx(-1.0)
        assert series.yaw_compass_deg[0] == pytest.approx(130.0)

    def test_header_matches_the_declared_contract(self, tmp_path: Path):
        path = write_attitude_sidecar(tmp_path, self._samples())
        with path.open(newline="", encoding="utf-8") as fh:
            assert tuple(next(csv.reader(fh))) == ATTITUDE_CSV_HEADER

    def test_refuses_missing_directory(self, tmp_path: Path):
        with pytest.raises(IngestError, match="does not exist"):
            write_attitude_sidecar(tmp_path / "nope", self._samples())

    def test_refuses_empty_sample_list(self, tmp_path: Path):
        with pytest.raises(IngestError, match="empty"):
            write_attitude_sidecar(tmp_path, [])

    def test_does_not_touch_contract_files(self, tmp_path: Path):
        """The sidecar must be additive: it exists precisely so groundtruth.csv's fixed seven
        columns stay untouched (EXP-003 Configuration, design choice 2)."""
        (tmp_path / "groundtruth.csv").write_text("sentinel", encoding="utf-8")
        (tmp_path / "dataset.json").write_text("{}", encoding="utf-8")
        write_attitude_sidecar(tmp_path, self._samples())
        assert (tmp_path / "groundtruth.csv").read_text(encoding="utf-8") == "sentinel"
        assert (tmp_path / "dataset.json").read_text(encoding="utf-8") == "{}"

    def test_slice_is_inclusive_and_order_preserving(self):
        s = self._samples(n=10)
        got = slice_attitude(s, 1000.02, 1000.05)
        assert [round(x.timestamp_s, 3) for x in got] == [1000.02, 1000.03, 1000.04, 1000.05]


# ------------------------------------------------------------------------------------------
# Rate and interpolation
# ------------------------------------------------------------------------------------------

def _series(t, roll=None, pitch=None, tilt=None, yaw=None):
    t = np.asarray(t, dtype=np.float64)
    z = np.zeros_like(t)
    return AttitudeSeries(
        timestamp_s=t,
        roll_deg=z if roll is None else np.asarray(roll, dtype=np.float64),
        pitch_deg=z if pitch is None else np.asarray(pitch, dtype=np.float64),
        tilt_deg=z if tilt is None else np.asarray(tilt, dtype=np.float64),
        yaw_compass_deg=z if yaw is None else np.asarray(yaw, dtype=np.float64),
    )


class TestTiltRate:
    def test_constant_tilt_has_zero_rate(self):
        t = np.arange(0, 5, 0.01)
        rate = tilt_rate_dps(_series(t, tilt=np.full_like(t, 4.0)))
        assert np.max(rate) == pytest.approx(0.0, abs=1e-9)

    def test_linear_ramp_recovers_its_slope(self):
        t = np.arange(0, 5, 0.01)
        rate = tilt_rate_dps(_series(t, tilt=2.0 * t))  # 2 deg/s
        interior = rate[100:-100]  # ignore boxcar edge effects
        assert np.median(interior) == pytest.approx(2.0, rel=0.02)

    def test_rate_is_non_negative(self):
        t = np.arange(0, 5, 0.01)
        rate = tilt_rate_dps(_series(t, tilt=np.sin(t) * 5.0))
        assert np.all(rate >= 0.0)

    def test_smoothing_suppresses_a_single_sample_spike(self):
        """EXP-003 threat 8: differentiating a 100 Hz EKF output amplifies noise, so one bad
        sample must not dominate the rate signal."""
        t = np.arange(0, 5, 0.01)
        tilt = np.full_like(t, 3.0)
        tilt[250] = 30.0
        smoothed = float(np.max(tilt_rate_dps(_series(t, tilt=tilt))))
        unsmoothed = float(np.max(tilt_rate_dps(_series(t, tilt=tilt), smooth_window_s=0.0)))
        assert smoothed < unsmoothed


class TestInterpolateTo:
    def test_interpolates_linearly_inside_the_span(self):
        got = interpolate_to(np.array([0.0, 1.0]), np.array([0.0, 10.0]), np.array([0.5]))
        assert got[0] == pytest.approx(5.0)

    def test_returns_nan_outside_the_span_rather_than_clamping(self):
        """np.interp alone would clamp to the endpoint value and silently fabricate attitude
        for frames the series does not cover."""
        got = interpolate_to(np.array([1.0, 2.0]), np.array([5.0, 6.0]), np.array([0.0, 3.0]))
        assert np.all(np.isnan(got))


# ------------------------------------------------------------------------------------------
# Rank statistics
# ------------------------------------------------------------------------------------------

class TestRankStatistics:
    def test_rankdata_averages_ties(self):
        assert list(rankdata(np.array([10.0, 20.0, 20.0, 30.0]))) == [1.0, 2.5, 2.5, 4.0]

    def test_spearman_is_one_for_a_monotone_nonlinear_relation(self):
        """The reason Spearman is the pre-registered statistic rather than Pearson."""
        x = np.arange(1.0, 21.0)
        assert spearman_rho(x, x ** 3) == pytest.approx(1.0)
        assert pearson_r(x, x ** 3) < 0.98

    def test_spearman_is_minus_one_when_reversed(self):
        x = np.arange(1.0, 21.0)
        assert spearman_rho(x, -x) == pytest.approx(-1.0)

    def test_zero_variance_gives_nan_not_a_crash(self):
        assert math.isnan(spearman_rho(np.ones(10), np.arange(10.0)))


class TestAutocorrelationDefences:
    def _ar1(self, n, phi, seed):
        rng = np.random.default_rng(seed)
        out = np.zeros(n)
        for i in range(1, n):
            out[i] = phi * out[i - 1] + rng.normal()
        return out

    def test_lag1_autocorr_recovers_a_known_ar1_coefficient(self):
        a = self._ar1(5000, 0.8, 1)
        assert lag1_autocorr(a) == pytest.approx(0.8, abs=0.05)

    def test_effective_n_is_much_smaller_than_n_for_correlated_series(self):
        x, y = self._ar1(1000, 0.95, 1), self._ar1(1000, 0.95, 2)
        assert effective_sample_size(x, y) < 100

    def test_effective_n_is_near_n_for_independent_noise(self):
        rng = np.random.default_rng(3)
        x, y = rng.normal(size=500), rng.normal(size=500)
        assert effective_sample_size(x, y) > 400

    def test_effective_n_never_exceeds_n(self):
        rng = np.random.default_rng(4)
        x = np.array([1.0, -1.0] * 250) + rng.normal(scale=0.01, size=500)  # anti-correlated
        assert effective_sample_size(x, x) <= 500

    def test_circular_shift_null_flags_a_pure_autocorrelation_artifact(self):
        """Two independent smooth random walks routinely correlate strongly by chance. The
        circular-shift null exists to catch exactly that, and this is its acceptance test."""
        x, y = np.cumsum(self._ar1(300, 0.9, 11)), np.cumsum(self._ar1(300, 0.9, 12))
        res = circular_shift_null(x, y)
        assert res["null_max_abs_rho"] > 0.5
        assert res["fraction_null_as_extreme"] > 0.05  # not distinguishable from the null

    def test_circular_shift_null_preserves_a_genuine_relationship(self):
        x = self._ar1(300, 0.9, 21)
        y = 3.0 * x + np.random.default_rng(22).normal(scale=0.05, size=300)
        res = circular_shift_null(x, y)
        assert res["rho_observed"] > 0.9
        assert res["fraction_null_as_extreme"] < 0.05

    def test_block_length_grows_with_autocorrelation(self):
        weak, strong = self._ar1(500, 0.1, 31), self._ar1(500, 0.95, 32)
        assert decorrelation_block_length(strong, strong) > decorrelation_block_length(weak, weak)

    def test_block_bootstrap_ci_brackets_a_moderate_correlation(self):
        """Deliberately a *moderate* relationship. A near-perfect one (rho > 0.998) sits against
        the rho = 1 boundary, where the bootstrap distribution is one-sided and a percentile CI
        can legitimately exclude the point estimate -- that is a known property of percentile
        bootstrap at a boundary, not a defect, and testing it there would pin the wrong thing."""
        x = np.arange(60.0)
        y = x + np.random.default_rng(41).normal(scale=25.0, size=60)
        rho = spearman_rho(x, y)
        assert 0.3 < rho < 0.95, "test setup should produce a moderate, non-boundary rho"
        lo, hi = moving_block_bootstrap_ci(x, y, block_length=5, n_resamples=400)
        assert lo <= rho <= hi

    def test_block_bootstrap_ci_is_wider_than_an_iid_pair_bootstrap_would_be(self):
        """The point of blocking: preserving within-block dependence must widen the interval,
        otherwise the CI understates uncertainty on autocorrelated data."""
        x = np.cumsum(self._ar1(200, 0.9, 81))
        y = x + np.cumsum(self._ar1(200, 0.9, 82))
        wide = moving_block_bootstrap_ci(x, y, block_length=25, n_resamples=400)
        narrow = moving_block_bootstrap_ci(x, y, block_length=1, n_resamples=400)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


class TestPermutationAndHolm:
    def test_permutation_p_is_small_for_a_strong_relation(self):
        x = np.arange(30.0)
        assert permutation_pvalue(x, x * 2.0, n_resamples=500) < 0.01

    def test_permutation_p_is_large_for_noise(self):
        rng = np.random.default_rng(51)
        assert permutation_pvalue(rng.normal(size=40), rng.normal(size=40), n_resamples=500) > 0.05

    def test_permutation_p_can_never_be_zero(self):
        x = np.arange(40.0)
        assert permutation_pvalue(x, x, n_resamples=100) > 0.0

    def test_holm_is_monotone_and_at_least_the_raw_p(self):
        raw = [0.001, 0.02, 0.03, 0.04, 0.5, 0.6, 0.7]
        adj = holm_adjust(raw)
        assert all(a >= r - 1e-12 for a, r in zip(adj, raw))
        ordered = [adj[i] for i in sorted(range(len(raw)), key=lambda i: raw[i])]
        assert ordered == sorted(ordered)

    def test_holm_first_step_multiplies_by_family_size(self):
        assert holm_adjust([0.001, 0.9, 0.9])[0] == pytest.approx(0.003)

    def test_holm_caps_at_one(self):
        assert all(p <= 1.0 for p in holm_adjust([0.5, 0.6, 0.7]))

    def test_holm_ignores_nan_tests(self):
        adj = holm_adjust([0.01, float("nan"), 0.02])
        assert math.isnan(adj[1])
        assert adj[0] == pytest.approx(0.02)  # family size 2, not 3


# ------------------------------------------------------------------------------------------
# Test harness and D1
# ------------------------------------------------------------------------------------------

class TestCorrelationHarness:
    def test_marks_underpowered_when_effective_n_is_tiny(self):
        t = np.arange(200.0)
        x = np.sin(t / 50.0)
        r = run_correlation_test("T0", "d", "H0", x, np.cos(t / 50.0), n_resamples=200)
        assert r.underpowered == (r.n_effective < 10)

    def test_degenerate_input_returns_nan_not_a_crash(self):
        r = run_correlation_test("T0", "d", "H0", np.array([1.0]), np.array([2.0]))
        assert math.isnan(r.rho) and r.underpowered

    def test_non_finite_pairs_are_dropped_pairwise(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        r = run_correlation_test("T0", "d", "H0", x, y, n_resamples=200)
        assert r.n == 3

    def test_apply_holm_fills_the_family(self):
        x = np.arange(30.0)
        results = [
            run_correlation_test("T1", "d", "H1", x, x, n_resamples=200),
            run_correlation_test("T2", "d", "H1", x, np.random.default_rng(7).normal(size=30),
                                 n_resamples=200),
        ]
        apply_holm(results)
        assert all(r.p_holm is not None for r in results)
        assert results[0].p_holm >= results[0].p_permutation

    def test_writes_a_csv_with_the_survival_column(self, tmp_path: Path):
        x = np.arange(30.0)
        results = apply_holm([run_correlation_test("T1", "d", "H1", x, x, n_resamples=200)])
        path = write_correlations_csv(results, tmp_path / "correlations.csv")
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows[0]["test_id"] == "T1"
        assert "survives_circular_shift" in rows[0]


class TestTercileSpreadD1:
    def test_spread_ratio_near_one_when_scale_is_unrelated_to_tilt(self):
        """The pre-registered H4 outcome: scale instability survives tilt control, so attitude
        is ruled out as the principal mechanism."""
        rng = np.random.default_rng(61)
        tilt = rng.uniform(0, 10, size=45)
        scale = rng.normal(1.0, 0.25, size=45)
        res = tercile_spread_analysis(tilt, scale)
        assert res["spread_ratio_sd"] > 0.5

    def test_spread_ratio_small_when_scale_deviation_is_driven_by_tilt(self):
        """The opposite outcome, which would implicate attitude."""
        rng = np.random.default_rng(62)
        tilt = np.linspace(0, 10, 45)
        scale = 1.0 + (tilt / 10.0) * rng.normal(0, 0.4, size=45)
        res = tercile_spread_analysis(tilt, scale)
        assert res["spread_ratio_sd"] < 0.8

    def test_reports_all_three_terciles_and_the_threshold(self):
        rng = np.random.default_rng(63)
        res = tercile_spread_analysis(rng.uniform(0, 10, 45), rng.normal(1.0, 0.2, 45))
        assert set(res["terciles"]) == {"low", "mid", "high"}
        assert res["h4_threshold"] == 0.8
        assert sum(res["terciles"][k]["n"] for k in res["terciles"]) == 45

    def test_refuses_too_few_segments(self):
        assert "error" in tercile_spread_analysis(np.arange(4.0), np.arange(4.0))

    def test_relative_spread_matches_a_hand_computed_case(self):
        a = np.array([1.0, 1.0, 1.0])
        assert relative_spread(a) == pytest.approx(0.0)


# ------------------------------------------------------------------------------------------
# Ingest gates
# ------------------------------------------------------------------------------------------

class TestIngestGates:
    def _good(self, n=30000, t0=1000.0):
        t = t0 + np.arange(n) * 0.01
        rng = np.random.default_rng(71)
        roll = rng.normal(2.15, 4.87, size=n)
        pitch = rng.normal(0.60, 4.13, size=n)
        tilt = np.hypot(roll, pitch)
        return _series(t, roll=roll, pitch=pitch, tilt=tilt, yaw=np.full(n, 130.0))

    def test_all_gates_pass_on_a_plausible_series(self):
        s = self._good()
        res = check_ingest_gates(s, (s.timestamp_s[0], s.timestamp_s[-1]), gt_heading_residual_deg=-1.95)
        assert res["all_pass"], res

    def test_g1_fails_on_a_coverage_gap(self):
        s = self._good(n=1000)
        t = s.timestamp_s.copy()
        t[500:] += 5.0
        gapped = _series(t, roll=s.roll_deg, pitch=s.pitch_deg, tilt=s.tilt_deg, yaw=s.yaw_compass_deg)
        res = check_ingest_gates(gapped, (t[0], t[-1]))
        assert not res["G1_coverage"]["pass"]

    def test_g2_fails_on_a_ninety_degree_convention_error(self):
        """The trap this dataset already sprang once, via rtk_yaw."""
        s = self._good(n=1000)
        res = check_ingest_gates(s, (s.timestamp_s[0], s.timestamp_s[-1]),
                                 gt_heading_residual_deg=90.0)
        assert not res["G2_yaw_self_consistency"]["pass"]
        assert not res["all_pass"]

    def test_g3_fails_on_an_implausible_tilt_distribution(self):
        n = 1000
        t = 1000.0 + np.arange(n) * 0.01
        s = _series(t, roll=np.full(n, 80.0), pitch=np.zeros(n), tilt=np.full(n, 80.0))
        res = check_ingest_gates(s, (t[0], t[-1]))
        assert not res["G3_plausibility"]["pass"]

    def test_distribution_reports_the_full_window_stats_lit006_lacked(self):
        d = attitude_distribution(self._good())
        assert d["roll_deg"]["sd"] == pytest.approx(4.87, abs=0.3)
        assert d["pitch_deg"]["sd"] == pytest.approx(4.13, abs=0.3)
        assert d["measured_rate_hz"] == pytest.approx(100.0, abs=1.0)
