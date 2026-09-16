"""`EXP-VO-012`'s pre-registered gates G1–G5, plus unit tests of the pieces they rest on.

The gates exist because six of this experiment's eleven hypotheses are **closed-form predictions**,
and a closed form compared against a subtly wrong implementation produces a confident wrong number
rather than an obvious failure. Each gate is a thing that, if broken, would make a whole family of
results meaningless:

===  =========================================================================================
G1   The ray-cast terrain renderer reduces EXACTLY to `EXP-VO-005`'s homography warp when the
     height field is zero. Without this the flat regimes are not comparable with the frozen
     record and the "did adding terrain change anything?" control is void.
G2   The ray/height-field intersection recovers a constructed terrain to < 1 mm, and reported
     `h_AGL` equals `h_camera − e(x_c)` exactly. Without this the oracle arm is not an oracle.
G3   The metric conversion, driven by TRUE height, reproduces `LIT-VO-003` §10.4's already
     verified table on `EXP-VO-005`'s committed runs. This pins the conversion against a prior
     result before it is used for anything new.
G4   The metric trajectory is BIT-IDENTICAL when the visual-scale series is replaced by
     arbitrary values. This is `DEC-VO-007` D4 — "visual scale has zero authority" — enforced
     rather than intended.
G5   Under the causal policy, no output frame depends on a height sample from its own future.
     Checked by corrupting the future and requiring the output not to move.
===  =========================================================================================

Tests needing rendered imagery or committed run records skip cleanly when absent.

Run from inside `evaluation/`: ``python -m pytest tools/exp_vo_012 -q``
"""

from __future__ import annotations

import importlib.util
import sys
import math
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    # Register before executing: `@dataclass` resolves string annotations through
    # `sys.modules[cls.__module__]`, which is None for a module that is not registered yet.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


st = _load("exp_vo_012_synth_terrain", "synth_terrain.py")
baro = _load("exp_vo_012_baro", "baro.py")
mr = _load("exp_vo_012_metric_readout", "metric_readout.py")
mm = _load("exp_vo_012_metric_metrics", "metric_metrics.py")


# ============================================================ G1 — renderer reduction

@pytest.mark.parametrize("east,north,up,heading", [
    (0.0, 0.0, 80.0, 0.0), (12.0, -30.0, 110.0, 0.0), (5.0, 20.0, 65.0, 37.0),
    (-40.0, 55.0, 95.0, 213.0),
])
def test_g1_ray_render_reduces_to_the_homography_warp(east, north, up, heading):
    """With `e ≡ 0` the two renderers must agree to within one grey level everywhere.

    They are different algorithms — a per-pixel ray/plane intersection versus a single projective
    warp — so exact bit equality is not expected; OpenCV's fixed-point interpolation differs by at
    most one level. Anything larger means the ray geometry disagrees with the homography, and every
    flat-regime comparison with `EXP-VO-005` would be invalid.
    """
    tex = st.sp.make_texture()
    new, residual = st.render_terrain(tex, np.array([east, north, up]), heading, st.terrain_flat)
    old = st.sp.render(tex, st.sp.ground_to_image(east, north, up, heading, 0.0))
    diff = np.abs(new.astype(int) - old.astype(int))
    assert residual == pytest.approx(0.0, abs=1e-9)
    assert (diff <= 1).mean() >= 0.999
    assert diff.max() <= 2


# ============================================================ G2 — known-answer geometry

def test_g2_intersection_recovers_a_constructed_plane():
    """A tilted plane is the one terrain whose intersection has a closed form, so it is the one that
    can be checked without trusting the solver that is being checked.

    The threshold is `EXP-VO-012`'s pre-registered **1 mm**, and the default six iterations clear it
    by an order of magnitude on the steepest terrain used (slope 0.25). One millimetre of ground at
    80 m through a 735 px focal length is 9 × 10⁻⁶ px, so the fixed-point tolerance is nowhere near
    the limiting error in any rendered frame.
    """
    slope = 0.20
    def elev(x, y):
        return slope * y
    camera = np.array([0.0, 0.0, 80.0])
    dirs = st.ray_directions(0.0)
    gx, gy, residual = st.intersect_terrain(camera, dirs, elev)
    assert residual < 1e-3
    # Converged, not merely small: iterating far past the default must not move the answer.
    gx2, gy2, r2 = st.intersect_terrain(camera, dirs, elev, tolerance_m=1e-12, min_iters=24,
                                        max_iters=40)
    assert r2 < 1e-9
    assert np.abs(gx - gx2).max() < 1e-3 and np.abs(gy - gy2).max() < 1e-3
    # Every returned point must also lie ON its ray. Checked as the agreement of the ray parameter
    # implied by x and by y separately — an exact property of the returned point that does not go
    # back through `elev`, so it is not contaminated by the fixed point's own residual. Masked to
    # rays whose x and y components are both well away from zero (they vanish on the centre row and
    # centre column, where the ratio is 0/0 rather than wrong).
    dx, dy = dirs[..., 0], dirs[..., 1]
    m = (np.abs(dx) > 0.05) & (np.abs(dy) > 0.05)
    assert m.sum() > 1e5
    t_from_x = (gx[m] - camera[0]) / dx[m]
    t_from_y = (gy[m] - camera[1]) / dy[m]
    assert np.abs(t_from_x - t_from_y).max() < 1e-9


def test_g2_intersection_recovers_a_curved_terrain():
    def elev(x, y):
        return 12.0 * np.sin(y / 40.0) + 3.0 * np.cos(x / 25.0)
    camera = np.array([7.0, -11.0, 90.0])
    dirs = st.ray_directions(23.0)
    gx, gy, residual = st.intersect_terrain(camera, dirs, elev)
    assert residual < st.RAY_TOLERANCE_M, f"intersection residual {residual:.3g} m"


def test_g2_solver_converges_on_the_steepest_terrain_actually_used():
    """The regression this gate caught: six fixed iterations left 6.9 mm on the ridge.

    The ridge's raised cosine reaches a 21.4 deg slope against the ramp's 14 deg, and the fixed-point
    convergence factor is the slope times the ray inclination — so the count a terrain needs is not a
    constant. The solver is tolerance-driven for exactly this reason, and this test pins the terrain
    that broke the constant.
    """
    elev = st.terrain_fn("ridge25")
    worst_fixed = 0.0
    for heading, camera in ((0.0, np.array([0.0, 0.0, 80.0])),
                            (0.0, np.array([0.0, -60.0, 80.0])),
                            (0.0, np.array([0.0, -100.0, 80.0])),
                            (0.0, np.array([0.0, 55.0, 80.0])),
                            (17.0, np.array([5.0, 40.0, 80.0]))):
        dirs = st.ray_directions(heading)
        _, _, residual = st.intersect_terrain(camera, dirs, elev)
        assert residual < st.RAY_TOLERANCE_M, f"ridge residual {residual:.3g} m at {camera}"
        _, _, fixed6 = st.intersect_terrain(camera, dirs, elev, min_iters=6, max_iters=6)
        worst_fixed = max(worst_fixed, fixed6)
    # The fix is load-bearing: somewhere on this terrain six fixed iterations miss the gate. Asserted
    # over the set rather than per position, because the ridge is only steep near its flanks -- which
    # is exactly why a per-frame assertion over a whole sequence caught it and a spot check did not.
    assert worst_fixed > st.RAY_TOLERANCE_M, (
        f"six fixed iterations cleared 1 mm everywhere tested ({worst_fixed:.3g} m); "
        "this test no longer pins the regression it was written for")


def test_g2_flat_and_ramp_still_take_exactly_the_minimum_iterations():
    """Reproduction guard: the committed flat/ramp run records were captured with six unconditional
    iterations, so this version must land on the same ground points for them."""
    for name in ("flat", "ramp30"):
        elev = st.terrain_fn(name)
        camera = np.array([0.0, 0.0, 80.0])
        dirs = st.ray_directions(0.0)
        adaptive = st.intersect_terrain(camera, dirs, elev)
        fixed6 = st.intersect_terrain(camera, dirs, elev, min_iters=6, max_iters=6)
        assert np.array_equal(adaptive[0], fixed6[0])
        assert np.array_equal(adaptive[1], fixed6[1])


def test_g2_reported_agl_is_camera_height_minus_terrain_under_the_camera():
    """The oracle arm's height must be exactly `up − e(x_c)`, not an approximation of it."""
    spec = st.SEQUENCES["bvo-ramp-constalt"]
    tr = st.trajectory(spec)
    elev = st.terrain_fn(spec["terrain"])
    under = np.array([float(elev(np.array([tr["east"][k]]), np.array([tr["north"][k]]))[0])
                      for k in range(len(tr["t"]))])
    agl = tr["up"] - under
    assert agl[0] == pytest.approx(80.0)
    assert agl.min() == pytest.approx(50.0, abs=1e-9)
    # And the barometer sees NONE of it -- which is the whole point of this regime.
    assert np.ptp(tr["up"]) == pytest.approx(0.0, abs=1e-12)


def test_g2_terrain_models_have_the_declared_shape():
    n = np.linspace(-150.0, 150.0, 601)
    e = np.zeros_like(n)
    ramp = st.terrain_ramp(e, n, rise_m=30.0, start_n=-60.0, end_n=60.0)
    assert ramp[0] == pytest.approx(0.0) and ramp[-1] == pytest.approx(30.0)
    assert np.all(np.diff(ramp) >= -1e-12), "the ramp must be monotone"
    ridge = st.terrain_ridge(e, n, rise_m=25.0, centre_n=0.0, width_n=200.0)
    assert ridge[0] == pytest.approx(0.0) and ridge[-1] == pytest.approx(0.0)
    assert ridge.max() == pytest.approx(25.0)
    # C1 at the feet: a raised cosine has zero derivative where it meets the flat ground.
    assert abs(ridge[1] - ridge[0]) < 1e-3


# ============================================================ G3 — conversion reproduction

#: `LIT-VO-003` §10.4, the already-verified table this conversion must reproduce. Endpoint error as a
#: percentage of path, converting with the TRUE height at the REFERENCE frame.
LIT_VO_003_TABLE = {
    ("synth-scale-climb", "affine"): 0.030,
    ("synth-scale-climb", "homography"): 0.083,
    ("synth-scale-descent", "affine"): 0.065,
    ("synth-scale-climbdesc", "affine"): 0.071,
}


def _synth_gt(dataset: str):
    import csv
    path = REPO / "datasets" / dataset / "groundtruth.csv"
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return (np.array([float(r["east_m"]) for r in rows]),
            np.array([float(r["north_m"]) for r in rows]),
            np.array([float(r["up_m"]) for r in rows]))


@pytest.mark.parametrize("dataset,model,expected_pct", sorted(
    (d, m, v) for (d, m), v in LIT_VO_003_TABLE.items()))
def test_g3_conversion_reproduces_lit_vo_003_section_10_4(dataset, model, expected_pct):
    run = REPO / "runs" / f"{dataset}-{model}-v1"
    gt = _synth_gt(dataset)
    if gt is None or not (run / "logical_transform.csv").exists():
        pytest.skip(f"{dataset}/{model} committed artifacts not present in this checkout")
    east, north, up = gt
    inc = mr.load_increments(run, st.W, st.H)
    # These EXP-VO-005 datasets are FLAT, so up_m IS the AGL. That equivalence is exactly
    # LIT-VO-006 eq. (4) with e == 0, and it is why this table is reusable here.
    track = mr.integrate_metric(inc, up, st.F_PX, arm="oracle")
    gt_xy = np.stack([east - east[0], north - north[0]], axis=1)
    est_xy = track.xy()
    path = float(np.sum(np.linalg.norm(np.diff(gt_xy, axis=0), axis=1)))
    end = float(np.linalg.norm(est_xy[-1] - gt_xy[-1]))
    assert 100.0 * end / path == pytest.approx(expected_pct, abs=0.006)


def test_g3_reference_frame_height_beats_current_frame_height():
    """`LIT-VO-003` §10.1's claim, re-checked here: the increment is in frame `k−1`'s pixels.

    Not a style question — using `h_k` is measurably worse on every altitude-varying arm, and the
    two coincide exactly when the altitude is constant, which is the control.
    """
    run = REPO / "runs" / "synth-scale-climb-affine-v1"
    gt = _synth_gt("synth-scale-climb")
    if gt is None or not (run / "logical_transform.csv").exists():
        pytest.skip("committed artifacts not present in this checkout")
    east, north, up = gt
    inc = mr.load_increments(run, st.W, st.H)
    gt_xy = np.stack([east - east[0], north - north[0]], axis=1)

    prev = mr.integrate_metric(inc, up, st.F_PX)
    curr_h = np.concatenate([up[1:], up[-1:]])          # shift so index k holds h_{k+1}
    curr = mr.integrate_metric(inc, curr_h, st.F_PX)
    e_prev = float(np.linalg.norm(prev.xy()[-1] - gt_xy[-1]))
    e_curr = float(np.linalg.norm(curr.xy()[-1] - gt_xy[-1]))
    assert e_prev < e_curr, f"reference-frame height {e_prev:.4f} m not better than {e_curr:.4f} m"


def test_g3_f_working_divides_by_the_downsample_factor():
    assert mr.f_working(1471.0653076171875, 2) == pytest.approx(735.5326538, abs=1e-6)
    assert mr.f_working(735.0, 1) == pytest.approx(735.0)
    with pytest.raises(ValueError):
        mr.f_working(735.0, 0)


def test_g3_forgetting_the_downsample_factor_doubles_every_distance():
    """The trap named in `LIT-VO-003` §10.2, made into a failing condition rather than a warning."""
    inc = mr.Increments(frame_index=np.arange(3), events=["init", "none", "none"],
                        dq=np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 0.0]]),
                        dtheta=np.zeros(3), inc_log_scale=np.zeros(3), timestamps_s=np.arange(3.0))
    h = np.full(3, 80.0)
    a = mr.integrate_metric(inc, h, mr.f_working(1470.0, 2))
    b = mr.integrate_metric(inc, h, mr.f_working(1470.0, 1))
    assert a.east_m[-1] == pytest.approx(2.0 * b.east_m[-1])


# ============================================================ G4 — visual-scale isolation

def test_g4_metric_track_is_bit_identical_under_an_arbitrary_visual_scale_series():
    """`DEC-VO-007` D4. If the visual channel ever acquired authority, this fails."""
    run = REPO / "runs" / "synth-scale-climb-affine-v1"
    gt = _synth_gt("synth-scale-climb")
    if gt is None or not (run / "logical_transform.csv").exists():
        pytest.skip("committed artifacts not present in this checkout")
    _, _, up = gt
    inc = mr.load_increments(run, st.W, st.H)
    base = mr.integrate_metric(inc, up, st.F_PX)

    rng = np.random.default_rng(4)
    for replacement in (np.zeros_like(inc.inc_log_scale),
                        np.full_like(inc.inc_log_scale, 12.5),
                        rng.normal(0, 3.0, inc.inc_log_scale.shape),
                        np.full_like(inc.inc_log_scale, np.nan)):
        poisoned = mr.Increments(inc.frame_index, inc.events, inc.dq, inc.dtheta,
                                 replacement, inc.timestamps_s)
        got = mr.integrate_metric(poisoned, up, st.F_PX)
        assert np.array_equal(got.east_m, base.east_m)
        assert np.array_equal(got.north_m, base.north_m)
        assert np.array_equal(got.gsd_m_per_px, base.gsd_m_per_px)


# ============================================================ G5 — causality

def test_g5_zoh_output_does_not_depend_on_future_samples():
    """Corrupt the future and require the causal output not to move — and the offline one to move.

    The second half matters as much as the first: if `linear` did *not* change, the test would be
    passing because the corruption was ineffective rather than because ZOH is causal.
    """
    t = np.arange(0, 20.0, 0.1)
    up = 80.0 + 0.5 * t
    spec = baro.BaroSpec(rate_hz=2.0)
    clean = baro.sample(t, up, spec)

    # Corrupt strictly BETWEEN two 2 Hz samples (10.0 and 10.5). Frames in (10.0, 10.2] are then in
    # the causal past of the last clean sample but in the interpolation span of a corrupted one, so
    # the two policies are genuinely distinguishable there. Corrupting exactly on a sample boundary
    # would make the test pass for both, which is the trap this comment exists to record.
    corrupted_truth = up.copy()
    corrupted_truth[t > 10.2] += 50.0
    poisoned = baro.sample(t, corrupted_truth, spec)

    past = t <= 10.2
    assert clean.causal is True
    assert np.allclose(clean.h_baro[past], poisoned.h_baro[past], equal_nan=True), \
        "a causal ZOH must not see the future"

    off = baro.sample(t, corrupted_truth, baro.with_(spec, policy="linear"))
    off_clean = baro.sample(t, up, baro.with_(spec, policy="linear"))
    assert off.causal is False
    assert not np.allclose(off.h_baro[past], off_clean.h_baro[past]), \
        "linear interpolation should have been contaminated, or the corruption did nothing"


def test_g5_latency_makes_the_channel_report_an_older_value():
    t = np.arange(0, 20.0, 0.1)
    up = 80.0 + 1.0 * t
    a = baro.sample(t, up, baro.BaroSpec(rate_hz=10.0))
    b = baro.sample(t, up, baro.BaroSpec(rate_hz=10.0, latency_s=1.0))
    late = t > 3.0
    assert np.nanmean(b.h_baro[late] - a.h_baro[late]) == pytest.approx(-1.0, abs=0.15)


# ============================================================ baro channel unit tests

def test_baro_ideal_channel_is_the_truth_relative_to_the_first_frame():
    t = np.arange(0, 30.0, 0.1)
    up = 80.0 + 20.0 * np.sin(t / 5.0)
    s = baro.ideal(t, up)
    assert np.allclose(s.h_baro, up - up[0], atol=1e-12)
    assert s.n_stale == 0 and s.n_invalid == 0 and s.causal


def test_baro_drift_is_linear_and_zero_at_the_reference_frame():
    t = np.arange(0, 60.0, 0.1)
    up = np.full_like(t, 80.0)
    s = baro.sample(t, up, baro.BaroSpec(drift_m=4.0))
    assert s.h_baro[0] == pytest.approx(0.0, abs=1e-12)
    assert s.h_baro[-1] == pytest.approx(4.0, rel=1e-9)
    assert np.mean(s.h_baro) == pytest.approx(2.0, rel=1e-6), "mean of a linear drift is D/2 (H5)"


def test_baro_noise_has_the_requested_sigma_and_is_reproducible():
    t = np.arange(0, 200.0, 0.1)
    up = np.full_like(t, 80.0)
    s = baro.sample(t, up, baro.BaroSpec(noise_m=0.25, seed=1))
    assert np.std(s.h_baro) == pytest.approx(0.25, rel=0.1)
    again = baro.sample(t, up, baro.BaroSpec(noise_m=0.25, seed=1))
    assert np.array_equal(s.h_baro, again.h_baro)
    other = baro.sample(t, up, baro.BaroSpec(noise_m=0.25, seed=2))
    assert not np.array_equal(s.h_baro, other.h_baro)


def test_baro_quantisation_lands_on_the_grid():
    t = np.arange(0, 20.0, 0.1)
    up = 80.0 + 3.0 * t / 20.0
    s = baro.sample(t, up, baro.BaroSpec(quantise_m=0.1))
    assert np.allclose(np.round(s.h_baro / 0.1), s.h_baro / 0.1, atol=1e-9)


def test_baro_dropout_marks_stale_then_invalid():
    t = np.arange(0, 40.0, 0.1)
    up = 80.0 + 0.5 * t
    s = baro.sample(t, up, baro.BaroSpec(rate_hz=10.0, dropouts=((10.0, 8.0),), tau_stale_s=2.0))
    # A value one sample-interval old is not yet a hold; staleness starts once it is older than
    # 1.5 intervals. So the check starts a couple of frames into the gap, deliberately.
    inside = (t >= 10.3) & (t < 18.0)
    assert s.stale[inside].all(), "every frame well inside the dropout must be marked stale"
    assert not s.stale[t < 9.0].any()
    # Beyond tau_stale the value is withdrawn rather than silently held.
    assert np.isnan(s.h_baro[(t > 12.5) & (t < 17.5)]).all()
    assert not np.isnan(s.h_baro[t < 9.0]).any()


def test_baro_zoh_lags_a_climb_by_half_a_sample_interval():
    """H6's mechanism, isolated — and its exact DISCRETE form.

    H6 predicts a mean height lag of `v/(2f)`. That is the continuous-time answer. Sampled at a
    finite frame interval `dt` the exact mean lag is `v·(T − dt)/2` with `T = 1/f`, which reduces to
    `v/(2f)` only for `T ≫ dt` and is **exactly zero** when the channel and the camera run at the
    same rate — as they must, since then every frame has its own fresh sample. Recorded here rather
    than fudged, because the R5 sweep spans both ends of that range.
    """
    t = np.arange(0, 60.0, 0.1)
    dt = 0.1
    v = 0.8
    up = 80.0 + v * t
    for rate in (10.0, 5.0, 2.0, 1.0, 0.5):
        s = baro.sample(t, up, baro.BaroSpec(rate_hz=rate))
        lag_m = np.mean((up - up[0]) - s.h_baro)
        expected = v * max(0.0, (1.0 / rate - dt)) / 2.0
        assert lag_m == pytest.approx(expected, abs=0.01 + 0.05 * expected)
    # And the continuous-time form H6 quotes is recovered once the channel is much slower than the
    # camera, which is the regime that matters for a real telemetry link.
    s = baro.sample(t, up, baro.BaroSpec(rate_hz=0.5))
    assert np.mean((up - up[0]) - s.h_baro) == pytest.approx(v / (2 * 0.5), rel=0.10)


# ============================================================ metric-metrics unit tests

def test_alignments_fit_what_they_claim_and_nothing_else():
    rng = np.random.default_rng(9)
    gt = np.cumsum(rng.normal(0, 1, (400, 2)), axis=0)
    a = math.radians(37.0)
    r = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    est = (gt @ r) * 0.7 + np.array([12.0, -5.0])       # rotate, SCALE by 0.7, translate

    sim2 = mm.align_sim2(est, gt)
    assert sim2.scale == pytest.approx(1 / 0.7, rel=1e-6)
    assert np.linalg.norm(sim2.apply(est) - gt, axis=1).max() < 1e-8

    se2 = mm.align_se2(est, gt)
    assert se2.scale == 1.0, "SE(2) must never fit scale"
    assert np.linalg.norm(se2.apply(est) - gt, axis=1).max() > 1.0, \
        "with scale held at 1 a 30 % scale error must remain visible"


def test_reference_initialised_fits_nothing():
    gt = np.array([[5.0, 7.0], [5.0, 17.0], [15.0, 17.0]])
    est = np.array([[0.0, 0.0], [0.0, 10.0], [10.0, 10.0]])   # same shape, own datum, heading0 = 0
    al = mm.reference_initialised(est, gt, 0.0)
    assert al.n_fitted == 0 and al.scale == 1.0
    assert np.allclose(al.apply(est), gt)


def test_reference_initialised_uses_the_known_initial_heading():
    est = np.array([[0.0, 0.0], [10.0, 0.0]])
    gt = np.array([[0.0, 0.0], [0.0, 10.0]])                  # east-going estimate, north-going GT
    al = mm.reference_initialised(est, gt, 90.0)
    assert np.allclose(al.apply(est), gt, atol=1e-9)


def test_path_length_step_is_mandatory_and_changes_the_answer():
    rng = np.random.default_rng(2)
    truth = np.stack([np.arange(1000.0), np.zeros(1000)], axis=1)
    noisy = truth + rng.normal(0, 0.5, truth.shape)
    assert mm.path_length(noisy, 1) > mm.path_length(noisy, 25) > 0
    assert mm.path_length(truth, 1) == pytest.approx(999.0)
    with pytest.raises(ValueError):
        mm.path_length(truth, 0)


def test_scale_error_shows_up_in_the_metric_family_and_not_in_sim2():
    """The methodological claim of Phase 5, as a test rather than an argument."""
    gt = np.stack([np.linspace(0, 300, 500), 0.3 * np.linspace(0, 300, 500)], axis=1)
    est = gt * 1.20                                          # a clean 20 % scale error
    s = mm.score(est, gt, heading0_deg=0.0, path_step=5)
    assert s["sim2"]["ate_rmse_normalised"] < 1e-9, "Sim(2) fits the scale away -- that is the point"
    assert s["ref_init"]["endpoint_error_pct_of_path"] > 15.0
    assert s["se2"]["endpoint_error_m"] > 1.0
    assert s["path_length_ratio"] == pytest.approx(1.20, rel=1e-6)
