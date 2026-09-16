"""Synthetic known-answer tests for the candidate matcher family (Part G).

Ten transformations with known answers — identical, vertical offset, horizontal shift, horizontal
scale, shift + scale, a small local nonlinear warp, partial overlap, noise, a distractor, and a flat
profile — checked against constructed truth rather than against each other.

These are **correctness** tests, not tuning. No bound here is chosen because it makes a score better;
each one is set to whatever the constructed transformation needs, and the assertion is that the
matcher recovers the parameter it was given. Nothing in this file measures retrieval performance and
nothing in it may be cited as evidence about one.

The property that matters most is the last group: a matcher with more freedom must not turn unrelated
curves into confident matches. That is the failure mode ``LIT-SKY-005`` warns about for C3 and C4, and
it is asserted here as a comparison of the *gain* the extra freedom buys on genuinely transformed
pairs against the gain it buys on unrelated ones.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hsreloc.matchers import (C0, C1, C3, C4, AngularProfileConfig, BoundedLagNccMatcher,
                              CameraModel, ConstrainedDtwMatcher, FrozenNccMatcher, MatcherError,
                              RepresentationError, ShiftScaleNccMatcher, angular_profile,
                              azimuth_grid, banded_dtw, build_matcher, curvature_profile,
                              degrees_per_sample, derivative_profile, lag_samples_for_degrees,
                              local_extrema, multiscale_profiles, roughness, smoothed_profile)
from hsreloc.retrieval.baselines import ncc as frozen_ncc

N = 256


def signal(t):
    """A smooth, structured, non-periodic-at-N profile. Defined for all real ``t`` so a transformed
    copy can be constructed exactly rather than resampled from a discrete one."""
    t = np.asarray(t, dtype=np.float64)
    return (np.sin(2 * np.pi * t / 37.0) + 0.5 * np.sin(2 * np.pi * t / 13.0 + 1.0)
            + 0.25 * np.sin(2 * np.pi * t / 71.0 - 0.4))


def centred(p):
    p = np.asarray(p, dtype=np.float64)
    return p - p.mean()


@pytest.fixture
def q():
    return centred(signal(np.arange(N)))


# -- 1. identical ------------------------------------------------------------------------------

def test_identical_profiles_score_one_in_every_variant(q):
    for m in (FrozenNccMatcher(), BoundedLagNccMatcher(max_lag_samples=8),
              ShiftScaleNccMatcher(max_lag_samples=4, scales=[0.98, 1.0, 1.02]),
              ConstrainedDtwMatcher(band=6)):
        res = m.match(q, q)
        assert res.score == pytest.approx(1.0, abs=1e-9), m.variant
        assert res.accepted


def test_identical_profiles_choose_the_identity_transform(q):
    assert BoundedLagNccMatcher(max_lag_samples=8).match(q, q).shift == 0.0
    ss = ShiftScaleNccMatcher(max_lag_samples=4, scales=[0.98, 1.0, 1.02]).match(q, q)
    assert (ss.shift, ss.scale) == (0.0, 1.0)
    assert ConstrainedDtwMatcher(band=6).match(q, q).warp_magnitude == pytest.approx(0.0)


# -- 2. vertical offset ------------------------------------------------------------------------

def test_a_constant_vertical_offset_does_not_change_any_score(q):
    """Pitch and altitude add a constant; the profile stage removes the mean and NCC removes it again."""
    for m in (FrozenNccMatcher(), BoundedLagNccMatcher(max_lag_samples=8),
              ShiftScaleNccMatcher(max_lag_samples=2, scales=[1.0]), ConstrainedDtwMatcher(band=4)):
        assert m.match(q, q + 3.7).score == pytest.approx(1.0, abs=1e-9), m.variant


# -- 3. horizontal shift -----------------------------------------------------------------------

@pytest.mark.parametrize("k", [-9, -3, -1, 1, 4, 12])
def test_c1_recovers_a_known_integer_shift(k):
    """``r`` is ``q`` sampled ``k`` further along, so the recovered shift is ``-k`` under the module's
    declared sign (positive shift = the reference is sampled further right)."""
    q = centred(signal(np.arange(N)))
    r = centred(signal(np.arange(N) + k))
    res = BoundedLagNccMatcher(max_lag_samples=16, min_overlap_frac=0.7).match(q, r)
    assert res.shift == pytest.approx(-k)
    assert res.score > 0.999
    assert res.overlap == N - abs(k)
    assert res.diagnostics["score_at_zero_lag"] < res.score


def test_c0_does_not_recover_a_shift_and_c1_does(q):
    r = centred(signal(np.arange(N) + 8))
    c0 = FrozenNccMatcher().match(q, r)
    c1 = BoundedLagNccMatcher(max_lag_samples=16, min_overlap_frac=0.7).match(q, r)
    assert c0.shift == 0.0 and c0.score < 0.9
    assert c1.score > 0.99
    assert c1.score > c0.score


def test_c1_recovers_a_sub_sample_shift():
    q = centred(signal(np.arange(N)))
    r = centred(signal(np.arange(N) + 2.5))
    res = BoundedLagNccMatcher(max_lag_samples=6, subsample_step=0.5, min_overlap_frac=0.7).match(q, r)
    assert res.shift == pytest.approx(-2.5)
    assert res.score > 0.999


def test_a_shift_beyond_the_declared_bound_is_not_found(q):
    """The bound is a commitment, not a hint: a 20-sample shift is simply not in a 5-sample search."""
    r = centred(signal(np.arange(N) + 20))
    res = BoundedLagNccMatcher(max_lag_samples=5, min_overlap_frac=0.7).match(q, r)
    assert abs(res.shift) <= 5
    assert res.score < 0.9


# -- 4 & 5. horizontal scale, shift + scale ----------------------------------------------------

def _scaled_reference(scale, lag, n=N):
    """``r`` such that ``r(centre + scale*(i-centre) + lag) == q[i]`` exactly."""
    centre = (n - 1) / 2.0
    j = np.arange(n, dtype=np.float64)
    return centred(signal(centre + (j - centre - lag) / scale))


@pytest.mark.parametrize("scale", [0.94, 0.97, 1.03, 1.06])
def test_c3_recovers_a_known_horizontal_scale(scale):
    q = centred(signal(np.arange(N)))
    r = _scaled_reference(scale, 0.0)
    grid = sorted({round(s, 4) for s in ShiftScaleNccMatcher.scale_grid(0.08, 17)} | {scale})
    res = ShiftScaleNccMatcher(max_lag_samples=0, scales=grid, min_overlap_frac=0.7).match(q, r)
    assert res.scale == pytest.approx(scale)
    assert res.score > 0.999
    assert res.diagnostics["score_at_identity"] < res.score


def test_c3_recovers_a_known_shift_and_scale_together():
    q = centred(signal(np.arange(N)))
    r = _scaled_reference(1.05, 6.0)
    grid = sorted({round(s, 4) for s in ShiftScaleNccMatcher.scale_grid(0.08, 17)} | {1.05})
    res = ShiftScaleNccMatcher(max_lag_samples=10, scales=grid, min_overlap_frac=0.6).match(q, r)
    assert (res.shift, res.scale) == (pytest.approx(6.0), pytest.approx(1.05))
    assert res.score > 0.999
    assert res.warp_magnitude > 0


def test_c1_alone_cannot_absorb_a_scale_and_c3_can():
    q = centred(signal(np.arange(N)))
    r = _scaled_reference(1.06, 0.0)
    c1 = BoundedLagNccMatcher(max_lag_samples=12, min_overlap_frac=0.6).match(q, r)
    grid = sorted({round(s, 4) for s in ShiftScaleNccMatcher.scale_grid(0.08, 17)} | {1.06})
    c3 = ShiftScaleNccMatcher(max_lag_samples=12, scales=grid, min_overlap_frac=0.6).match(q, r)
    assert c3.score > c1.score + 0.01


# -- 6. small local nonlinear warp --------------------------------------------------------------

def _warped_reference(amplitude, n=N):
    """A mild monotone nonlinearity — what a depth-dependent (near-field) viewpoint change looks like
    when it is *not* a uniform shift or a uniform scale."""
    j = np.arange(n, dtype=np.float64)
    return centred(signal(j + amplitude * np.sin(2 * np.pi * j / n)))


def test_c4_absorbs_a_local_warp_that_shift_and_scale_cannot():
    q = centred(signal(np.arange(N)))
    r = _warped_reference(5.0)
    c0 = FrozenNccMatcher().match(q, r)
    c1 = BoundedLagNccMatcher(max_lag_samples=8, min_overlap_frac=0.7).match(q, r)
    c3 = ShiftScaleNccMatcher(max_lag_samples=8, scales=ShiftScaleNccMatcher.scale_grid(0.06, 13),
                              min_overlap_frac=0.7).match(q, r)
    c4 = ConstrainedDtwMatcher(band=10, max_mean_warp=8.0).match(q, r)
    assert c4.score > c3.score >= c0.score
    assert c4.score > c1.score
    assert 0.0 < c4.warp_magnitude <= 8.0
    assert c4.diagnostics["n_non_diagonal_steps"] > 0


def test_c4_reports_and_rejects_a_pathological_alignment():
    q = centred(signal(np.arange(N)))
    r = _warped_reference(9.0)
    res = ConstrainedDtwMatcher(band=12, max_mean_warp=0.5).match(q, r)
    assert res.accepted is False
    assert res.score == -math.inf
    assert "exceeds the declared maximum" in res.diagnostics["reason"]


def test_banded_dtw_preserves_left_to_right_order_and_stays_inside_its_band():
    q = centred(signal(np.arange(64)))
    r = _warped_reference(4.0, 64)
    out = banded_dtw(q, r, band=6)
    path = out["path"]
    assert np.all(np.diff(path[:, 0]) >= 0) and np.all(np.diff(path[:, 1]) >= 0)
    assert np.any(np.diff(path[:, 0]) + np.diff(path[:, 1]) > 0)     # strictly advancing
    assert out["max_warp"] <= 6
    assert tuple(path[0]) == (0, 0) and tuple(path[-1]) == (63, 63)


def test_a_band_that_cannot_span_the_length_difference_is_refused():
    with pytest.raises(MatcherError, match="cannot span the length difference"):
        banded_dtw(np.zeros(10), np.zeros(30), band=2)


# -- 7. partial overlap ------------------------------------------------------------------------

def test_overlap_is_reported_and_the_floor_is_enforced(q):
    r = centred(signal(np.arange(N) + 60))
    res = BoundedLagNccMatcher(max_lag_samples=100, min_overlap_frac=0.9).match(q, r)
    assert res.overlap >= 0.9 * N
    assert abs(res.shift) <= N * 0.1 + 1
    assert res.score < 0.99                       # the true alignment is outside the allowed overlap


def test_a_tiny_overlap_alignment_cannot_win(q):
    """Without a floor, a similarity maximised over ever-smaller windows drifts to 1.0 on anything."""
    r = np.asarray(np.random.default_rng(11).normal(size=N))
    loose = BoundedLagNccMatcher(max_lag_samples=N - 4, min_overlap_frac=0.02).match(q, centred(r))
    strict = BoundedLagNccMatcher(max_lag_samples=N - 4, min_overlap_frac=0.6).match(q, centred(r))
    assert loose.overlap_frac < strict.overlap_frac
    assert loose.score > strict.score             # exactly the inflation the floor exists to stop
    assert strict.overlap_frac >= 0.6


def test_no_alignment_clearing_the_floor_is_a_refusal_not_a_score(q):
    res = ShiftScaleNccMatcher(max_lag_samples=0, scales=[3.0], min_overlap_frac=1.0).match(q, q)
    assert res.accepted is False
    assert res.score == -math.inf
    assert res.overlap == 0


def test_c1_with_a_zero_bound_is_exactly_c0(q):
    r = centred(signal(np.arange(N) + 3))
    assert (BoundedLagNccMatcher(max_lag_samples=0, min_overlap_frac=1.0).match(q, r).score
            == FrozenNccMatcher().match(q, r).score)


# -- 8. noise ----------------------------------------------------------------------------------

def test_noise_degrades_the_score_gracefully_and_does_not_move_the_recovered_shift():
    rng = np.random.default_rng(5)
    q = centred(signal(np.arange(N)))
    r = centred(signal(np.arange(N) + 4) + rng.normal(scale=0.05, size=N))
    res = BoundedLagNccMatcher(max_lag_samples=12, min_overlap_frac=0.7).match(q, r)
    assert res.shift == pytest.approx(-4)
    assert 0.9 < res.score < 1.0


# -- 9. distractor / unrelated profiles ---------------------------------------------------------

def _unrelated(rng, n=N):
    """A smooth but independent profile — a different place, not a transformed same place."""
    t = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    for _ in range(4):
        out += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * t / rng.uniform(9, 90) + rng.uniform(0, 6.3))
    return centred(out)


def _ladder():
    return {C0: FrozenNccMatcher(),
            C1: BoundedLagNccMatcher(max_lag_samples=12, min_overlap_frac=0.7),
            C3: ShiftScaleNccMatcher(max_lag_samples=12,
                                     scales=ShiftScaleNccMatcher.scale_grid(0.06, 13),
                                     min_overlap_frac=0.7),
            C4: ConstrainedDtwMatcher(band=10, max_mean_warp=8.0)}


def _unrelated_scores(matchers, n_pairs=40, seed=2026):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in matchers}
    for _ in range(n_pairs):
        a, b = _unrelated(rng), _unrelated(rng)
        for key, m in matchers.items():
            res = m.match(a, b)
            if res.accepted:                       # a refused alignment is not a score
                out[key].append(res.score)
    return out


def test_flexible_matchers_do_not_turn_unrelated_curves_into_confident_matches():
    scores = _unrelated_scores(_ladder())
    for key, values in scores.items():
        assert max(values) < 0.95, f"{key} reached {max(values):.3f} on an unrelated pair"
        assert float(np.mean(values)) < 0.8, key


def test_added_freedom_inflates_unrelated_scores_and_the_ladder_order_reflects_it():
    """The measured basis for the conservative ordering C0 → C1 → C3 → C4.

    On this synthetic corpus every variant recovers a genuinely shifted pair (C1/C3 to 1.000, C4 to
    0.99), so the difference between them is not what they gain on a true match — it is what they give
    away on a false one. Measured here: mean score on **unrelated** pairs rises from -0.02 (C0) to
    0.19 (C1) to 0.27 (C3), with worst cases of 0.56, 0.63 and 0.88; the separation between the worst
    true match and the best false one therefore shrinks monotonically along the ladder.

    That is exactly ``LIT-SKY-005``'s argument, and it is why a rung is only worth climbing once the
    geometry shows the previous one is insufficient. These are numbers about *this constructed
    corpus*, not about skylines: they bound the mechanism, they do not predict a retrieval rate.
    """
    matchers = _ladder()
    q = centred(signal(np.arange(N)))
    true_scores = {k: [] for k in matchers}
    for k in (-8, -4, 4, 8):
        r = centred(signal(np.arange(N) + k))
        for key, m in matchers.items():
            true_scores[key].append(m.match(q, r).score)
    fake_scores = _unrelated_scores(matchers, seed=7)

    # C0 cannot separate an 8-sample shift of the same place from a different place at all — the
    # rigidity EXP-SKY-006 ran into, reproduced here on constructed data.
    assert min(true_scores[C0]) < max(fake_scores[C0])
    margins = {}
    for key in (C1, C3, C4):
        assert min(true_scores[key]) > 0.95, key                       # the true pair is recovered
        assert min(true_scores[key]) > max(fake_scores[key]), key      # and still separated
        assert np.mean(fake_scores[key]) > np.mean(fake_scores[C0]), key   # but false pairs rose too
        margins[key] = min(true_scores[key]) - max(fake_scores[key])
    assert margins[C1] > margins[C3] > margins[C4]


# -- 10. flat profile --------------------------------------------------------------------------

def test_a_flat_profile_carries_no_information_in_any_variant(q):
    flat = np.zeros(N)
    for m in (FrozenNccMatcher(), BoundedLagNccMatcher(max_lag_samples=8),
              ShiftScaleNccMatcher(max_lag_samples=4, scales=[1.0]),
              ConstrainedDtwMatcher(band=6, max_mean_warp=8.0)):
        assert abs(m.match(flat, q).score) < 1e-6, m.variant
        assert abs(m.match(flat, flat).score) < 1e-6, m.variant


# -- C0 equivalence and reporting ---------------------------------------------------------------

def test_c0_is_bit_identical_to_the_frozen_baseline_scorer():
    rng = np.random.default_rng(99)
    c0 = FrozenNccMatcher()
    for _ in range(50):
        a, b = centred(rng.normal(size=N)), centred(rng.normal(size=N))
        assert c0.score(a, b) == frozen_ncc(a, b)


def test_every_variant_reports_every_transform_parameter(q):
    r = centred(signal(np.arange(N) + 3))
    for m in (FrozenNccMatcher(), BoundedLagNccMatcher(max_lag_samples=8),
              ShiftScaleNccMatcher(max_lag_samples=4, scales=[0.99, 1.0]),
              ConstrainedDtwMatcher(band=6, max_mean_warp=8.0)):
        d = m.match(q, r).as_dict()
        for key in ("variant", "score", "shift", "scale", "overlap", "overlap_frac",
                    "warp_magnitude", "accepted"):
            assert key in d, (m.variant, key)
        assert m.describe()["scoring_primitive"].startswith("hsreloc.retrieval.baselines.ncc")


def test_matchers_are_drop_in_scorers_for_the_unchanged_ranker(q):
    """``rank_candidates`` takes any ``(query, reference) -> float``; a variant is exactly that."""
    from hsreloc.retrieval.rank import RankedCandidate, sort_candidates
    m = BoundedLagNccMatcher(max_lag_samples=6, min_overlap_frac=0.7)
    refs = {"a": centred(signal(np.arange(N) + 3)), "b": _unrelated(np.random.default_rng(1))}
    ranked = sort_candidates([RankedCandidate(k, 0.0, 0.0, m(q, v)) for k, v in refs.items()])
    assert ranked[0].reference_id == "a"


def test_tie_breaking_prefers_the_least_transformed_alignment():
    """A perfectly periodic profile matches at several lags; the smallest one must win, every time."""
    period = 32
    p = centred(np.sin(2 * np.pi * np.arange(N) / period))
    res = BoundedLagNccMatcher(max_lag_samples=period + 4, min_overlap_frac=0.5).match(p, p)
    assert res.shift == 0.0
    for _ in range(5):
        assert BoundedLagNccMatcher(max_lag_samples=period + 4,
                                    min_overlap_frac=0.5).match(p, p).shift == 0.0


def test_misconfiguration_is_refused(q):
    with pytest.raises(MatcherError, match="min_overlap_frac"):
        BoundedLagNccMatcher(max_lag_samples=4, min_overlap_frac=0.0)
    with pytest.raises(MatcherError, match="max_lag_samples must be >= 0"):
        BoundedLagNccMatcher(max_lag_samples=-1)
    with pytest.raises(MatcherError, match="scales must be"):
        ShiftScaleNccMatcher(scales=[])
    with pytest.raises(MatcherError, match="same length"):
        FrozenNccMatcher().match(q, q[:-1])
    with pytest.raises(MatcherError, match="unknown matcher variant"):
        build_matcher("c9_nonsense")


def test_build_matcher_constructs_each_rung():
    assert build_matcher(C0).variant == C0
    assert build_matcher(C1, {"max_lag_samples": 5}).max_lag_samples == 5
    assert build_matcher(C1, {"max_lag_deg": 2.0, "fov_deg": 90.0, "n_samples": 256}).max_lag_samples == 6
    assert build_matcher(C3, {"max_scale_deviation": 0.05, "n_scale_steps": 5}).scales[0] < 1.0
    assert build_matcher(C4, {"band": 3}).band == 3


def test_a_lag_bound_in_degrees_converts_through_the_field_of_view():
    assert lag_samples_for_degrees(2.0, 90.0, 256) == 6        # 2 deg / (90/256 deg per sample)
    assert lag_samples_for_degrees(0.0, 90.0, 256) == 0
    with pytest.raises(MatcherError):
        lag_samples_for_degrees(1.0, 0.0, 256)


# -- C2: the angular representation --------------------------------------------------------------

CAM = CameraModel(width_px=256, height_px=256, fx=128.0, fy=128.0, cx=128.0, cy=128.0)


def test_azimuth_and_elevation_match_the_analytic_pinhole_values():
    assert CAM.azimuth_deg(128.0) == pytest.approx(0.0)
    assert CAM.azimuth_deg(256.0) == pytest.approx(45.0)       # x - cx = fx -> atan(1) = 45 deg
    assert CAM.azimuth_deg(0.0) == pytest.approx(-45.0)
    assert CAM.elevation_deg(128.0) == pytest.approx(0.0)
    assert CAM.elevation_deg(0.0) == pytest.approx(45.0)       # top of frame is *above* the axis
    assert CAM.elevation_deg(256.0) == pytest.approx(-45.0)


def test_column_and_row_inverses_round_trip():
    cols = np.array([0.0, 40.0, 128.0, 200.0, 255.0])
    assert np.allclose(CAM.column_of_azimuth(CAM.azimuth_deg(cols)), cols)
    rows = np.array([1.0, 60.0, 128.0, 250.0])
    assert np.allclose(CAM.row_of_elevation(CAM.elevation_deg(rows)), rows)


def test_a_yaw_offset_is_a_uniform_shift_in_angular_units_but_not_in_pixel_units():
    """The whole claim of C2, checked on constructed geometry: a distant skyline described in azimuth,
    imaged twice with the camera rotated by a known yaw."""
    def sky_elevation(az_deg):                       # a fixed skyline, in world azimuth
        return 6.0 * np.sin(np.radians(az_deg) * 3.0) + 2.0 * np.sin(np.radians(az_deg) * 11.0)

    cols = np.arange(CAM.width_px, dtype=np.float64)
    yaw = 3.0
    rows_a = CAM.row_of_elevation(sky_elevation(CAM.azimuth_deg(cols)))
    rows_b = CAM.row_of_elevation(sky_elevation(CAM.azimuth_deg(cols) + yaw))
    cfg = AngularProfileConfig(n_samples=181, common_span_deg=36.0)

    ang_a, ang_b = angular_profile(rows_a, CAM, cfg), angular_profile(rows_b, CAM, cfg)
    step = degrees_per_sample(CAM, cfg)
    lag_samples = int(round(yaw / step))
    m = BoundedLagNccMatcher(max_lag_samples=lag_samples + 6, min_overlap_frac=0.7)
    angular = m.match(ang_a, ang_b)
    assert angular.shift == pytest.approx(-lag_samples, abs=1)
    assert angular.score > 0.995                    # one shift explains the whole profile

    # In pixel units the same yaw is not one shift: the best single lag leaves a visible residual.
    pix_a = centred(1.0 - rows_a / CAM.height_px)
    pix_b = centred(1.0 - rows_b / CAM.height_px)
    pixel = BoundedLagNccMatcher(max_lag_samples=40, min_overlap_frac=0.7).match(pix_a, pix_b)
    assert pixel.score < angular.score


def test_angular_profiles_agree_across_two_cameras_of_different_resolution_and_fov():
    def sky_elevation(az_deg):
        return 5.0 * np.sin(np.radians(az_deg) * 2.5) + 1.5 * np.cos(np.radians(az_deg) * 7.0)

    wide = CameraModel(256, 256, 128.0, 128.0, 128.0, 128.0)               # 90 deg HFOV
    narrow = CameraModel(512, 384, 512 / (2 * math.tan(math.radians(30))), 400.0, 256.0, 192.0)
    cfg = AngularProfileConfig(n_samples=201, common_span_deg=40.0)
    a = angular_profile(wide.row_of_elevation(sky_elevation(wide.azimuth_deg(np.arange(wide.width_px)))),
                        wide, cfg)
    b = angular_profile(narrow.row_of_elevation(sky_elevation(narrow.azimuth_deg(np.arange(narrow.width_px)))),
                        narrow, cfg)
    assert frozen_ncc(a, b) > 0.999


def test_a_common_span_wider_than_the_field_of_view_is_refused_not_extrapolated():
    cfg = AngularProfileConfig(n_samples=64, common_span_deg=120.0)
    rows = np.full(CAM.width_px, 100.0)
    with pytest.raises(RepresentationError, match="exceeds this camera's field of view"):
        angular_profile(rows, CAM, cfg)


def test_the_angular_profile_needs_a_curve_at_column_resolution():
    with pytest.raises(RepresentationError, match="cannot be applied to a resampled profile"):
        angular_profile(np.zeros(64), CAM, AngularProfileConfig())


def test_camera_model_from_session_refuses_a_session_without_intrinsics():
    ok = CameraModel.from_session({"camera": {"resolution_px": [256, 256],
                                              "intrinsics": {"fx": 128.0, "fy": 128.0,
                                                             "cx": 128.0, "cy": 128.0}}})
    assert ok.hfov_deg == pytest.approx(CAM.azimuth_deg(255.0) - CAM.azimuth_deg(0.0))
    with pytest.raises(RepresentationError, match="would be a guess about the camera"):
        CameraModel.from_session({"camera": {"resolution_px": [256, 256], "intrinsics": {}}})


def test_azimuth_grid_is_uniform_and_matches_the_declared_span():
    grid = azimuth_grid(CAM, AngularProfileConfig(n_samples=91, common_span_deg=45.0))
    assert grid[0] == pytest.approx(-22.5) and grid[-1] == pytest.approx(22.5)
    assert np.allclose(np.diff(grid), grid[1] - grid[0])


# -- diagnostic representations (never the primary curve) ----------------------------------------

def test_the_derivative_profile_removes_a_linear_trend(q):
    ramp = np.linspace(-1.0, 1.0, N)
    assert frozen_ncc(derivative_profile(q), derivative_profile(q + ramp)) > 0.99
    assert frozen_ncc(q, q + ramp) < 0.99          # the raw profile is not trend-invariant


def test_smoothing_and_multiscale_behave_as_declared(q):
    assert np.allclose(smoothed_profile(q, 0.0), q)
    assert roughness(smoothed_profile(q, 8.0)) < roughness(q)
    scales = multiscale_profiles(q, sigmas=(0.0, 2.0, 8.0))
    assert set(scales) == {"sigma_0", "sigma_2", "sigma_8"}
    assert all(v.shape == q.shape for v in scales.values())


def test_curvature_and_extrema_are_descriptive_only(q):
    assert curvature_profile(q).shape == q.shape
    ext = local_extrema(q)
    assert ext["n_extrema"] == len(ext["maxima"]) + len(ext["minima"])
    assert local_extrema(np.zeros(N))["n_extrema"] == 0
