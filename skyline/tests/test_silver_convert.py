"""The declared label-map -> silver-curve conversion (EXP-SKY-005, hsreloc.extraction.silver.convert).

Known-answer tests on synthetic ADE20K-style label maps: every case the rule promises to handle
explicitly (no sky, several sky regions, sky through branches, enclosed sky, speckle, no defensible
boundary) is asserted here rather than argued for in prose.
"""

from __future__ import annotations

import numpy as np

from hsreloc.extraction.silver import convert as cv

SKY = cv.SKY_CLASS_INDEX
BUILDING = 1
TREE = 4
H, W = 60, 80


def _blank() -> np.ndarray:
    return np.full((H, W), BUILDING, dtype=np.uint8)


def test_simple_horizon_boundary_is_the_last_sky_row():
    lab = _blank()
    lab[:20, :] = SKY
    res = cv.silver_curve(lab)
    assert res["status"] == cv.STATUS_OK
    assert res["valid"].all()
    assert np.allclose(res["rows"], 19.5)


def test_columns_without_sky_are_invalid_not_invented():
    lab = _blank()
    lab[:20, :40] = SKY                      # sky only on the left half
    res = cv.silver_curve(lab, min_valid_frac=0.5)
    assert res["valid"][:40].all() and not res["valid"][40:].any()
    assert np.isnan(res["rows"][40:]).all()
    assert res["valid_frac"] == 0.5
    assert res["status"] == cv.STATUS_OK     # exactly at the floor


def test_below_the_floor_is_an_explicit_status():
    lab = _blank()
    lab[:20, :20] = SKY
    res = cv.silver_curve(lab, min_valid_frac=0.5)
    assert res["status"] == cv.STATUS_INSUFFICIENT
    assert res["valid_frac"] == 0.25


def test_two_top_connected_regions_are_both_used():
    lab = _blank()
    lab[:15, :30] = SKY
    lab[:25, 50:] = SKY                      # a tower between them reaches the top row
    res = cv.silver_curve(lab, min_valid_frac=0.4)
    assert res["status"] == cv.STATUS_OK
    assert np.allclose(res["rows"][:30], 14.5)
    assert np.allclose(res["rows"][50:], 24.5)
    assert not res["valid"][30:50].any()
    assert res["n_sky_components"] == 2


def test_enclosed_sky_never_defines_a_boundary():
    """Sky seen through a window/arch is a component that does not touch the top row."""
    lab = _blank()
    lab[:20, :] = SKY
    lab[40:50, 20:30] = SKY                  # enclosed patch far below the horizon
    res = cv.silver_curve(lab)
    assert np.allclose(res["rows"], 19.5)    # the patch is ignored entirely
    assert res["n_sky_components"] == 2
    assert res["sky_frac_top_connected"] < res["sky_frac_raw"]


def test_speck_below_the_boundary_does_not_move_it():
    lab = _blank()
    lab[:20, :] = SKY
    lab[30, 10] = SKY                        # 1-pixel speck, connected to nothing
    res = cv.silver_curve(lab)
    assert res["rows"][10] == 19.5


def test_no_top_connected_sky_is_reported_not_guessed():
    lab = _blank()
    lab[30:40, 10:20] = SKY                  # only enclosed sky
    res = cv.silver_curve(lab)
    assert res["status"] == cv.STATUS_NO_TOP_SKY
    assert not res["valid"].any()
    assert np.isnan(res["rows"]).all()


def test_sky_through_branches_topmost_run_stops_at_the_first_branch():
    lab = _blank()
    lab[:30, :] = SKY
    lab[10, :] = TREE                        # a branch crossing the whole width at row 10
    res = cv.silver_curve(lab, closing_px=0)
    assert np.allclose(res["rows"], 9.5)     # the run ends at the branch, it is not bridged
    bridged = cv.silver_curve(lab, closing_px=5)
    assert np.allclose(bridged["rows"], 29.5)   # closing bridges the 1-px branch -> canopy envelope
    assert bridged["closing_px"] == 5


def test_boundary_is_clipped_inside_the_image():
    lab = np.full((H, W), SKY, dtype=np.uint8)          # all sky
    res = cv.silver_curve(lab)
    assert res["rows"].max() == H - 1.0
    assert res["status"] == cv.STATUS_OK


def test_conversion_is_deterministic():
    rng = np.random.default_rng(7)
    lab = _blank()
    lab[:25, :] = SKY
    lab[24, rng.integers(0, W, 12)] = TREE
    first = cv.silver_curve(lab, closing_px=3)
    second = cv.silver_curve(lab, closing_px=3)
    assert np.array_equal(np.nan_to_num(first["rows"], nan=-1), np.nan_to_num(second["rows"], nan=-1))
    assert first["convert_version"] == cv.CONVERT_VERSION
