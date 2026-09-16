"""Unit tests for the SKY matcher's building blocks: the seam, profiles, scorers, ranking, fixes.

Covers tasks T004, T006, T008, T010, T014. Each block tests the property that actually matters rather
than that the code runs -- e.g. the profile tests pin *amplitude retention*, because whether a
representation keeps vertical scale is the question ``LIT-009`` showed determines what a horizon
matcher can express.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from hsreloc.retrieval import baselines, rank
from hsreloc.retrieval.fix import GeodeticOrigin, derive_fix, enu_to_geodetic
from hsreloc.retrieval.profile import ProfileConfig, ProfileError, is_degenerate, normalize
from hsreloc.retrieval.rank import RankedCandidate
from hsreloc.retrieval.skyline_curve import CurveError, curve_digest, make_curve

from naveval.frames import geodetic_to_enu

H = 96
W = 128


def _curve(values, width=W, height=H, provenance="oracle:manual"):
    return make_curve("obs_x", values, width, height, provenance, "test")


def _ramp(width=W, height=H):
    return np.linspace(0.2 * height, 0.7 * height, width)


# --- T004: the matcher-facing seam ------------------------------------------------------------

def test_curve_rejects_row_outside_image():
    with pytest.raises(CurveError, match="rows outside"):
        _curve(np.full(W, H + 1.0))


def test_curve_rejects_negative_row():
    values = _ramp().copy()
    values[3] = -0.5
    with pytest.raises(CurveError, match="rows outside"):
        _curve(values)


def test_curve_rejects_nan():
    values = _ramp().copy()
    values[10] = np.nan
    with pytest.raises(CurveError, match="NaN or Inf"):
        _curve(values)


def test_curve_rejects_length_mismatch():
    with pytest.raises(CurveError, match="does not cover every column"):
        make_curve("obs_x", _ramp(width=W - 1), W, H, "oracle:manual", "test")


def test_curve_rejects_unknown_provenance():
    with pytest.raises(CurveError, match="unknown provenance"):
        _curve(_ramp(), provenance="guessed")


def test_curve_accepts_future_automatic_provenance():
    """The seam must already accept what a later extractor will declare, without a change here."""
    curve = _curve(_ramp(), provenance="automatic:some-future-method")
    assert curve.provenance == "automatic:some-future-method"


def test_digest_is_stable_and_content_sensitive():
    a = _ramp()
    b = a.copy()
    b[7] += 1e-6
    assert curve_digest(a) == curve_digest(a.copy())
    assert curve_digest(a) != curve_digest(b)


# --- T006: profile normalization --------------------------------------------------------------

def test_profile_has_fixed_length_regardless_of_image_width():
    cfg = ProfileConfig(n_samples=256)
    narrow = normalize(_curve(_ramp(width=64), width=64), cfg)
    wide = normalize(_curve(_ramp(width=640), width=640), cfg)
    assert narrow.shape == wide.shape == (256,)


def test_profile_removes_the_mean_when_configured():
    prof = normalize(_curve(_ramp()), ProfileConfig(normalize_mean=True))
    assert abs(float(prof.mean())) < 1e-12


def test_profile_is_invariant_to_a_constant_vertical_offset():
    """Pitch or altitude shifts the whole skyline; mean removal is what absorbs that."""
    cfg = ProfileConfig(normalize_mean=True)
    base = _ramp()
    a = normalize(_curve(base), cfg)
    b = normalize(_curve(base + 7.0), cfg)
    assert np.allclose(a, b, atol=1e-9)


def test_profile_retains_amplitude():
    """The profile keeps vertical scale; only the *scorer* may discard it (research R6)."""
    cfg = ProfileConfig(normalize_mean=True)
    centre = H * 0.5
    small = normalize(_curve(centre + 0.2 * (_ramp() - centre)), cfg)
    large = normalize(_curve(centre + 0.8 * (_ramp() - centre)), cfg)
    assert np.std(large) > 3.0 * np.std(small)


def test_detrend_false_preserves_a_genuine_slope():
    """Nordland has no roll data and a rigid camera, so a ramp here is terrain, not tilt."""
    prof = normalize(_curve(_ramp()), ProfileConfig(detrend=False))
    assert np.std(prof) > 0.05
    flat = normalize(_curve(_ramp()), ProfileConfig(detrend=True))
    assert np.std(flat) < np.std(prof)


def test_profile_rejects_units_it_cannot_support():
    with pytest.raises(ProfileError, match="elevation-angle"):
        ProfileConfig(units="elevation_angle_deg").validate()


def test_degenerate_detection():
    cfg = ProfileConfig()
    assert is_degenerate(normalize(_curve(np.full(W, 40.0)), cfg), 1e-6)
    assert not is_degenerate(normalize(_curve(_ramp()), cfg), 1e-6)


# --- T008: scorers ----------------------------------------------------------------------------

def _wave(width=W, height=H, freq=3.0, phase=0.0):
    x = np.linspace(0.0, 2.0 * np.pi, width)
    return 0.5 * height + 0.2 * height * np.sin(freq * x + phase)


@pytest.mark.parametrize("name", baselines.REAL_BASELINES)
def test_scorer_identity_is_maximal(name):
    """A different *shape*, not merely a different offset or scale -- otherwise NCC ties at 1.0."""
    scorer = baselines.get_scorer(name)
    p = normalize(_curve(_wave(freq=3.0)), ProfileConfig())
    q = normalize(_curve(_wave(freq=7.0, phase=1.1)), ProfileConfig())
    assert scorer(p, p) > scorer(p, q)


@pytest.mark.parametrize("name", baselines.REAL_BASELINES)
def test_scorer_is_symmetric(name):
    scorer = baselines.get_scorer(name)
    p = normalize(_curve(_wave(freq=3.0)), ProfileConfig())
    q = normalize(_curve(_wave(freq=5.0, phase=0.7)), ProfileConfig())
    assert scorer(p, q) == pytest.approx(scorer(q, p), abs=1e-12)


def test_ncc_discards_amplitude_but_l1_and_l2_do_not():
    """The deliberate bracket: the baseline family measures the amplitude question, not assumes it."""
    cfg = ProfileConfig()
    p = normalize(_curve(_ramp()), cfg)
    scaled = 0.5 * p
    assert baselines.ncc(p, scaled) == pytest.approx(1.0, abs=1e-9)
    assert baselines.l1(p, scaled) < -1e-6
    assert baselines.l2(p, scaled) < -1e-6


def test_mock_ignores_curve_content_and_is_seed_reproducible():
    a = baselines.MockScorer(seed=3)
    b = baselines.MockScorer(seed=3)
    c = baselines.MockScorer(seed=4)
    assert a.score_pair("q1", "r1") == b.score_pair("q1", "r1")
    assert a.score_pair("q1", "r1") != c.score_pair("q1", "r1")
    assert a.score_pair("q1", "r1") != a.score_pair("q1", "r2")


def test_mock_is_not_reachable_as_a_curve_scorer():
    with pytest.raises(baselines.BaselineError):
        baselines.get_scorer(baselines.MOCK)


def test_unknown_baseline_is_rejected():
    with pytest.raises(baselines.BaselineError, match="unknown baseline"):
        baselines.get_scorer("netvlad")


# --- T010: ranking ----------------------------------------------------------------------------

def _cands(pairs):
    return [RankedCandidate(rid, 0.0, 0.0, score) for rid, score in pairs]


def test_candidates_are_ordered_best_first():
    ordered = rank.sort_candidates(_cands([("a", 0.1), ("b", 0.9), ("c", 0.5)]))
    assert [c.reference_id for c in ordered] == ["b", "c", "a"]


def test_ties_break_deterministically_by_reference_id():
    ordered = rank.sort_candidates(_cands([("z", 0.5), ("a", 0.5), ("m", 0.5)]))
    assert [c.reference_id for c in ordered] == ["a", "m", "z"]


def test_shortlist_is_capped_and_tolerates_a_small_reference_set():
    cands = _cands([("a", 0.3), ("b", 0.2)])
    assert len(rank.shortlist(cands, 5)) == 2
    assert len(rank.shortlist(cands, 1)) == 1


def test_shortlist_rejects_nonsense_k():
    with pytest.raises(ValueError):
        rank.shortlist(_cands([("a", 1.0)]), 0)


# --- T014: the geographic fix -----------------------------------------------------------------

ORIGIN = GeodeticOrigin(63.4401, 10.45111, 42.0)


@pytest.mark.parametrize("east,north", [(0, 0), (1500, -2200), (-53448.02, -69711.63), (250_000, 190_000)])
def test_enu_geodetic_round_trip_matches_naveval(east, north):
    """The inverse must agree with naveval's forward model, not merely be self-consistent."""
    lat, lon, alt = enu_to_geodetic(east, north, 0.0, ORIGIN)
    back_e, back_n, _ = geodetic_to_enu(lat, lon, alt, ORIGIN.lat_deg, ORIGIN.lon_deg, ORIGIN.alt_m)
    assert back_e == pytest.approx(east, abs=1e-6)
    assert back_n == pytest.approx(north, abs=1e-6)


def test_fix_is_the_matched_reference_verbatim_with_no_interpolation():
    fix = derive_fix("ref_3", 1234.5, -678.25, ORIGIN)
    assert (fix.east_m, fix.north_m) == (1234.5, -678.25)
    assert fix.matched_reference_id == "ref_3"


def test_fix_never_carries_a_heading():
    assert derive_fix("ref_0", 10.0, 20.0, ORIGIN).heading_deg is None


def test_origin_mismatch_is_detectable():
    other = GeodeticOrigin(64.06304019047619, 11.49989019047619, 51.4847619047619)
    assert not ORIGIN.matches(other)
    assert ORIGIN.matches(GeodeticOrigin(63.4401, 10.45111, 42.0))
