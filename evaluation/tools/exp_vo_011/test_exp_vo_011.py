"""Tests for EXP-VO-011's analysis tooling.

These test the *instruments*, not the VO: known-answer statistics, the exact recovery of the
image-frame travel direction from the pose series, the row filter, and the sign conventions the
whole record rests on. Run from `evaluation/`:

    cd evaluation; python -m pytest tools/exp_vo_011 -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import covariates as cv          # noqa: E402
import flow_direction as fd      # noqa: E402
import scale_series as ss        # noqa: E402

REPO = Path(__file__).resolve().parents[3]
HAS_RUNS = (REPO / "runs" / "hkairport01-b-affine-rigid-v1" / "logical_transform.csv").exists()
HAS_REVERSED = (REPO / "runs" / "hkairport01-a-reversed-affine-rigid-v1"
                / "logical_transform.csv").exists()
needs_runs = pytest.mark.skipif(not HAS_RUNS, reason="committed rigid run records absent")


# --------------------------------------------------------------------- statistics


def test_mad_sigma_matches_sd_on_a_gaussian():
    x = np.random.default_rng(7).normal(0, 3.0, 200_000)
    assert ss.mad_sigma(x) == pytest.approx(3.0, rel=0.02)


def test_mad_sigma_ignores_heavy_tails_that_move_the_sd():
    base = np.random.default_rng(9).normal(0, 1.0, 20_000)
    contaminated = base.copy()
    contaminated[:200] += 50.0
    assert np.std(contaminated) > 1.5 * np.std(base)
    assert ss.mad_sigma(contaminated) == pytest.approx(ss.mad_sigma(base), rel=0.05)


def test_lag1_recovers_a_known_ar1_coefficient():
    rng = np.random.default_rng(3)
    rho, x = 0.6, [0.0]
    for _ in range(200_000):
        x.append(rho * x[-1] + rng.normal())
    assert ss.lag1(np.array(x[1:])) == pytest.approx(rho, abs=0.01)


def test_effective_sample_size_shrinks_under_positive_autocorrelation():
    rng = np.random.default_rng(5)
    white = rng.normal(0, 1, 20_000)
    corr = np.convolve(rng.normal(0, 1, 20_050), np.ones(50) / 50, mode="valid")[:20_000]
    assert ss.describe(white)["n_eff"] > 5 * ss.describe(corr)["n_eff"]


def test_describe_reports_the_sum_that_the_accumulation_uses():
    x = np.array([1.0, 2.0, -0.5])
    assert ss.describe(x)["sum"] == pytest.approx(2.5)


# --------------------------------------------------------------------- correlation machinery


def test_partial_correlation_removes_a_shared_linear_trend():
    t = np.linspace(0, 1, 4000)
    ctrl = np.column_stack([t, t ** 1.0])
    # Two series that share only the trend: raw correlation near 1, partial near 0.
    y = 5 * t + np.random.default_rng(1).normal(0, 0.05, len(t))
    x = 3 * t + np.random.default_rng(2).normal(0, 0.05, len(t))
    assert abs(np.corrcoef(y, x)[0, 1]) > 0.99
    assert abs(cv.partial_correlation(y, x, ctrl)["r"]) < 0.1


def test_partial_correlation_keeps_a_genuine_relation_that_survives_the_control():
    rng = np.random.default_rng(11)
    t = np.linspace(0, 1, 4000)
    z = rng.normal(0, 1, len(t))
    y = 2 * t + 0.8 * z + rng.normal(0, 0.1, len(t))
    r = cv.partial_correlation(y, z, np.column_stack([t, t]))["r"]
    assert r > 0.9


def test_partial_correlation_refuses_a_structurally_constant_covariate():
    y = np.random.default_rng(4).normal(0, 1, 100)
    out = cv.partial_correlation(y, np.zeros(100), np.column_stack([np.arange(100.0)] * 2))
    assert math.isnan(out["r"])


def test_spearman_is_invariant_to_a_monotone_transform():
    x = np.linspace(0.1, 5, 500)
    y = np.exp(x)
    assert cv.spearman(y, x) == pytest.approx(1.0, abs=1e-9)


def test_heading_test_separates_a_constant_from_a_harmonic():
    psi = np.linspace(0, 360, 4000, endpoint=False)
    y = 3e-4 + 5e-4 * np.cos(np.radians(psi))
    h = cv.heading_test(y, psi)
    assert h["harmonic"]["constant"] == pytest.approx(3e-4, rel=1e-6)
    assert h["harmonic"]["amplitude"] == pytest.approx(5e-4, rel=1e-6)
    # The harmonic integrates to zero over uniformly covered headings; the constant does not.
    assert h["accumulation_split"]["harmonic_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert h["accumulation_split"]["constant_fraction"] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------- circular statistics


def test_circular_stats_on_a_constant_direction():
    s = fd.circular_stats(np.full(500, 37.0))
    assert s["mean_deg"] == pytest.approx(37.0, abs=1e-6)
    assert s["resultant_R"] == pytest.approx(1.0, abs=1e-9)


def test_circular_stats_on_a_uniform_direction():
    s = fd.circular_stats(np.linspace(0, 360, 100_000, endpoint=False))
    assert s["resultant_R"] < 1e-4


def test_circular_stats_handles_the_wrap_at_zero():
    s = fd.circular_stats(np.array([359.0, 1.0]))
    assert min(s["mean_deg"], 360 - s["mean_deg"]) < 1e-6


# --------------------------------------------------------------------- real-record contracts


@needs_runs
def test_row_filter_drops_init_and_restart_and_keeps_recenter():
    arm = ss.load_arm("hkairport01-a", "homography")
    assert "init" not in set(arm["event"])
    assert "restart" not in set(arm["event"])
    assert "recenter" in set(arm["event"])
    assert arm["n_restart"] == 1          # the frame-922 failure, this lane's only real one
    assert arm["n_dropped"] == 2          # that restart plus the init row


@needs_runs
def test_accumulated_log_scale_reproduces_the_committed_column():
    """The sum of the increments must equal the estimator's own accumulator, up to the rows dropped.

    This is the gate that says the record's own arithmetic is the estimator's arithmetic.
    """
    arm = ss.load_arm("amtown01-c", "affine")     # no restarts, so nothing is dropped
    assert arm["n_restart"] == 0
    assert float(np.sum(arm["inc_log_scale"])) == pytest.approx(
        float(arm["rigid_accum_log_scale"][-1]), abs=1e-9)


@needs_runs
def test_image_flow_recovery_is_exact_against_the_stored_flow_magnitude():
    for window, model in (("hkairport01-b", "affine"), ("amtown01-c", "homography")):
        arm = ss.load_arm(window, model)
        assert fd.image_flow(arm)["max_abs_error_px"] < 1e-9


@needs_runs
def test_similarity_arms_have_structurally_degenerate_diagnostics():
    arm = ss.load_arm("amtown01-c", "similarity")
    assert np.allclose(arm["inc_anisotropy"], 1.0, atol=1e-9)
    assert np.allclose(arm["inc_perspective"], 0.0, atol=1e-12)


@needs_runs
def test_lidar_covariate_is_height_above_the_imaged_surface_not_the_takeoff_datum():
    """`LIT-VO-004` section 4's distinction, enforced rather than trusted.

    Note the ratio asserted here is 1.15, not `LIT-VO-004`'s 1.66. That figure is for the whole
    `AMtown01` cruise; `amtown01-c` is the truncated 14-of-27-sub-window prefix (`EXP-VO-007`), over
    which the LiDAR height above the imaged surface varies by 1.19. Quoting 1.66 for this window
    would overstate it, which is exactly the kind of slippage this test exists to prevent.
    """
    arm = ss.load_arm("amtown01-c", "affine")
    c = ss.covariates_for(arm)
    assert c["up_m"].max() - c["up_m"].min() < 0.5          # constant to centimetres
    assert c["lidar_height_m"].max() / c["lidar_height_m"].min() > 1.15   # and this one is not


@pytest.mark.skipif(not HAS_REVERSED, reason="the time-reversed replay runs are absent")
def test_time_reversal_reverses_the_sign_of_the_scale_bias():
    r = fd.reversal_pair("hkairport01-a", "affine")
    assert r["forward"]["mean"] > 0
    assert r["reversed"]["mean"] < 0
    assert r["sign_flipped"]
    # Mostly odd in the travel direction rather than an arrow-of-time property of the estimator.
    assert r["odd_fraction"] > 0.5
