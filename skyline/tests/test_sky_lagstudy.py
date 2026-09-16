"""EXP-SKY-013 — the nested-bound instrument and the positional-resolution bookkeeping.

The load-bearing claim these tests defend: reading five lag bounds off ONE bound-32 scoring pass is
*exact*, not an approximation. If that ever stops holding, every comparison in EXP-SKY-013 becomes
a comparison between two different instruments rather than between two lag bounds.
"""
from __future__ import annotations

import numpy as np
import pytest

from hsreloc.matchers import BoundedLagNccMatcher, FrozenNccMatcher
from hsreloc.simret.lagstudy import (DEFAULT_BOUNDS, LagStudyError, NestedBoundBank, binned,
                                     bound_label, degrees_per_sample, elevation_offset_deg,
                                     eligibility, implied_range_m, mean_elevation, nested_slices,
                                     positional_fields, quantiles, recency_mask, verify_bounds)

N = 256


def _profile(seed: int) -> np.ndarray:
    r = np.random.default_rng(seed)
    x = np.linspace(0.0, 6.0 * np.pi, N)
    p = np.sin(x) + 0.4 * np.sin(3.0 * x + r.uniform(0, 3)) + 0.05 * r.standard_normal(N)
    return p - p.mean()


@pytest.fixture(scope="module")
def corpus():
    base = _profile(100)
    refs = [_profile(i) for i in range(8)] + [np.roll(base, s) for s in (-31, -9, -3, 5, 17, 32)]
    ids = [f"r{i:03d}" for i in range(len(refs))]
    return base, refs, ids


# --------------------------------------------------------------------------------------------------
# the nested read-off is exact
# --------------------------------------------------------------------------------------------------

def test_nested_slices_cover_the_symmetric_subgrid():
    sl = nested_slices([0, 4, 32], 32)
    lags = np.arange(-32, 33)
    assert list(lags[sl[0]]) == [0]
    assert list(lags[sl[4]]) == list(range(-4, 5))
    assert list(lags[sl[32]]) == list(range(-32, 33))


def test_a_bound_outside_the_scored_grid_is_refused():
    with pytest.raises(LagStudyError):
        nested_slices([64], 32)
    with pytest.raises(LagStudyError):
        nested_slices([-1], 32)


def test_every_bound_equals_the_frozen_matcher_object(corpus):
    """Winning lag exactly, score to 1e-12 — against BoundedLagNccMatcher itself."""
    base, refs, ids = corpus
    bank = NestedBoundBank(ids, refs, bounds=DEFAULT_BOUNDS, max_lag=32, min_overlap_frac=0.6)
    for q in (base, _profile(900), _profile(901)):
        got = bank.score(q)
        for b in DEFAULT_BOUNDS:
            m = BoundedLagNccMatcher(max_lag_samples=b, min_overlap_frac=0.6)
            s, lg = got[b]
            for i, r in enumerate(refs):
                res = m.match(q, r)
                assert abs(float(s[i]) - res.score) <= 1e-12
                assert float(lg[i]) == float(res.shift)


def test_bound_zero_is_the_frozen_c0_scorer(corpus):
    base, refs, ids = corpus
    bank = NestedBoundBank(ids, refs, bounds=(0,), max_lag=32, min_overlap_frac=0.6)
    s, lg = bank.score(base)[0]
    for i, r in enumerate(refs):
        assert abs(float(s[i]) - FrozenNccMatcher().match(base, r).score) <= 1e-12
        assert float(lg[i]) == 0.0


def test_score_is_non_decreasing_in_the_bound(corpus):
    """A larger search space can never find a worse best — if this fails the slicing is wrong."""
    base, refs, ids = corpus
    bank = NestedBoundBank(ids, refs, bounds=DEFAULT_BOUNDS, max_lag=32)
    got = bank.score(_profile(555))
    for lo, hi in zip(DEFAULT_BOUNDS, DEFAULT_BOUNDS[1:]):
        assert np.all(got[lo][0] <= got[hi][0] + 1e-12)


def test_verify_bounds_reports_agreement(corpus):
    base, refs, ids = corpus
    bank = NestedBoundBank(ids, refs, max_lag=32, min_overlap_frac=0.6)
    v = verify_bounds(bank, base, refs, indices=range(len(refs)))
    assert v["n_lag_mismatch"] == 0
    assert v["max_abs_score_delta"] <= 1e-12
    assert v["n_checked"] == len(refs) * len(DEFAULT_BOUNDS)


def test_verify_bounds_raises_when_the_bank_is_tampered_with(corpus):
    """A silent divergence must stop the run, not appear in a table."""
    base, refs, ids = corpus
    bank = NestedBoundBank(ids, refs, max_lag=32, min_overlap_frac=0.6)
    # the tamper has to change the profile *shape*: the frozen NCC removes each window's own mean
    # and divides by its norm, so an offset or a positive rescale would be invisible (see below)
    bank.S_chunks[0] = bank.S_chunks[0] + np.linspace(0.0, 3.0, bank.n_samples)[None, :]
    with pytest.raises(LagStudyError):
        verify_bounds(bank, base, refs, indices=[0])


def test_the_score_is_blind_to_a_constant_offset_and_a_positive_rescale(corpus):
    """The H6 mechanism, asserted rather than asserted-about.

    ``profile_distance`` removes each window's mean and divides by its norm, so the absolute
    elevation of a profile never reaches the score. That is exactly the component a pure altitude
    change shifts — which is why EXP-SKY-013 tests mean elevation as an *auxiliary scalar* and not
    by deleting the mean subtraction.
    """
    base, refs, ids = corpus
    bank = NestedBoundBank(ids, refs, bounds=(0, 32), max_lag=32)
    plain = bank.score(base)
    shifted = bank.score(base + 0.25)
    scaled = bank.score(base * 3.0)
    for b in (0, 32):
        assert np.allclose(plain[b][0], shifted[b][0], atol=1e-12)
        assert np.allclose(plain[b][0], scaled[b][0], atol=1e-12)


def test_profiles_must_share_one_length(corpus):
    base, refs, ids = corpus
    with pytest.raises(LagStudyError):
        NestedBoundBank(["a", "b"], [refs[0], refs[1][:100]])
    with pytest.raises(LagStudyError):
        NestedBoundBank([], [])
    bank = NestedBoundBank(ids, refs)
    with pytest.raises(LagStudyError):
        bank.score(np.zeros(10))


def test_score_full_returns_every_lag(corpus):
    base, refs, ids = corpus
    bank = NestedBoundBank(ids, refs, max_lag=32)
    full = bank.score_full(base)
    assert full.shape == (len(refs), 65)
    # the bound-32 winner is the max over the whole row
    s32, _ = bank.score(base)[32]
    assert np.allclose(np.max(full, axis=1), s32, atol=1e-12)


# --------------------------------------------------------------------------------------------------
# positional bookkeeping
# --------------------------------------------------------------------------------------------------

def test_positional_fields_read_the_selection_back_in_metres():
    ref_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    ref_h = np.array([50.0, 70.0])
    outcome = {"top1_id": "far", "nearest_id": "near", "top1_distance_m": 100.0,
               "d_near_m": 5.0, "top1_score": 0.99, "nearest_score": 0.95, "exact_rank": 4,
               "exact_top1": False}
    pf = positional_fields((0.0, 0.0), 60.0, ref_xy, ref_h, outcome,
                           {"near": 0, "far": 1}, catastrophic_m=500.0)
    assert pf["selected_distance_m"] == 100.0
    assert pf["selection_excess_m"] == pytest.approx(95.0)
    assert pf["top1_minus_nearest_score"] == pytest.approx(0.04)
    assert pf["nearest_rank"] == 4
    assert pf["catastrophic"] is False
    assert pf["selected_dh_m"] == pytest.approx(-10.0)      # 60 - 70
    assert pf["nearest_dh_m"] == pytest.approx(10.0)        # 60 - 50


def test_catastrophic_uses_the_declared_threshold():
    ref_xy = np.array([[0.0, 0.0]])
    o = {"top1_id": "a", "nearest_id": "a", "top1_distance_m": 600.0, "d_near_m": 600.0,
         "top1_score": 0.99, "nearest_score": 0.99, "exact_rank": 1, "exact_top1": True}
    assert positional_fields((0, 0), None, ref_xy, None, o, {"a": 0}, 500.0)["catastrophic"] is True
    assert positional_fields((0, 0), None, ref_xy, None, o, {"a": 0}, 700.0)["catastrophic"] is False


def test_eligibility_is_a_mask_over_one_scoring_pass():
    m = eligibility(5, self_index=2, exclude_indices=[0])
    assert list(m) == [False, True, False, True, True]


def test_recency_mask_excludes_a_window_along_the_trajectory():
    orders = np.arange(10)
    assert list(recency_mask(5, orders, 0)) == [True] * 5 + [False] + [True] * 4
    m = recency_mask(5, orders, 2)
    assert not m[3:8].any() and m[0] and m[9]
    with pytest.raises(LagStudyError):
        recency_mask(0, orders, -1)


def test_quantiles_and_binned_tolerate_empty_input():
    q = quantiles([])
    assert q["n"] == 0 and q["p95"] is None
    q = quantiles([1.0, 2.0, 3.0, None, float("nan")])
    assert q["n"] == 3 and q["max"] == 3.0
    rows = binned([1.0, 5.0], [0.0, 50.0], [0.0, 10.0, 100.0])
    assert rows[0]["n"] == 1 and rows[1]["n"] == 1
    assert binned([], [], [0.0, 10.0])[0]["p50"] is None


# --------------------------------------------------------------------------------------------------
# altitude helpers
# --------------------------------------------------------------------------------------------------

def test_mean_elevation_is_the_scalar_the_profile_stage_discards():
    rows = np.full(8, 256.0)
    assert mean_elevation(rows, 512) == pytest.approx(0.5)
    # a skyline higher in the frame (smaller row) has a larger elevation
    assert mean_elevation(np.full(8, 128.0), 512) == pytest.approx(0.75)
    with pytest.raises(LagStudyError):
        mean_elevation(np.array([]), 512)
    with pytest.raises(LagStudyError):
        mean_elevation(np.full(4, 10.0), 0)


def test_elevation_offset_and_implied_range_are_first_order_readings():
    assert elevation_offset_deg(0.51, 0.50, 90.0) == pytest.approx(0.9)
    # dh / tan-ish: 1 m of altitude over 0.01 deg implies a far scene
    assert implied_range_m(1.0, 0.01) == pytest.approx(1.0 / np.radians(0.01))
    assert implied_range_m(1.0, 0.0) is None
    assert implied_range_m(1.0, None) is None


def test_labels_and_sample_geometry():
    assert bound_label(0) == "C0"
    assert bound_label(32) == "C1-32"
    assert degrees_per_sample(90.0, 256) == pytest.approx(0.3515625)
    with pytest.raises(LagStudyError):
        degrees_per_sample(0.0, 256)
