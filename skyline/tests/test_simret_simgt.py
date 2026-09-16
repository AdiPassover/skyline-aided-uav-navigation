"""Known-answer tests for the simulator-GT mask → skyline curve conversion (Part B).

Every mask here is drawn by hand, so the correct curve is known before the code runs. The cases are
the ones the brief named — flat horizon, sloped skyline, multiple peaks, a tower reaching the top of
the frame, a sky hole, a disconnected white region, no sky at all, all sky — plus the binarity gate,
because a mask that is not binary is not ground truth and must be refused rather than thresholded.
"""

from __future__ import annotations

import numpy as np
import pytest

from hsreloc.extraction.curves import REASON_NO_SKY_AT_TOP, REASON_NONE
from hsreloc.simret import simgt

H, W = 20, 12


def mask_from_rows(rows) -> np.ndarray:
    """Sky strictly above ``rows[x]``; boundary therefore expected at ``rows[x] - 0.5``."""
    m = np.zeros((H, W), dtype=bool)
    for x, r in enumerate(rows):
        if r > 0:
            m[:int(r), x] = True
    return m


def test_flat_horizon():
    res = simgt.sim_gt_curve(mask_from_rows([10] * W))
    assert res["status"] == simgt.STATUS_OK
    assert res["valid"].all()
    assert np.allclose(res["rows"], 9.5)


def test_sloped_skyline():
    rows = [4 + x for x in range(W)]
    res = simgt.sim_gt_curve(mask_from_rows(rows))
    assert res["status"] == simgt.STATUS_OK
    assert np.allclose(res["rows"], np.array(rows) - 0.5)


def test_multiple_peaks():
    rows = [12, 6, 3, 6, 12, 12, 5, 2, 5, 12, 12, 12]
    res = simgt.sim_gt_curve(mask_from_rows(rows))
    assert np.allclose(res["rows"], np.array(rows) - 0.5)
    assert res["valid"].all()


def test_tower_reaching_the_top_of_frame_leaves_an_invalid_column():
    """A column with no sky above it has no boundary. None is invented, and the seam curve does not
    exist for the image — the refusal the matcher turns into EXTRACTION_FAILURE."""
    rows = [10] * W
    rows[5] = 0                                   # structure occupies the column from row 0 down
    res = simgt.sim_gt_curve(mask_from_rows(rows))
    assert res["status"] == simgt.STATUS_INSUFFICIENT
    assert res["valid"][5] is np.False_
    assert res["valid"].sum() == W - 1
    assert np.isnan(res["rows"][5])
    assert simgt.seam_curve_from_result("obs", res, W, H) is None
    canonical = simgt.canonical_from_result(res, W, H)
    assert canonical.invalid_reasons[5] == REASON_NO_SKY_AT_TOP
    assert canonical.invalid_reasons[0] == REASON_NONE
    assert canonical.n_valid == W - 1


def test_sky_hole_stops_the_topmost_run_at_the_hole():
    """A non-sky blob enclosed in the sky (a balloon, a bird, a rendering speck) ends the topmost
    contiguous run above it. The sky below the hole is not used to push the boundary down."""
    m = mask_from_rows([14] * W)
    m[6:9, 4:7] = False                            # the hole
    res = simgt.sim_gt_curve(m)
    assert res["valid"].all()
    assert np.allclose(res["rows"][4:7], 5.5)      # run ends at row 5 -> 5.5
    assert np.allclose(res["rows"][0:4], 13.5)     # untouched columns keep the real horizon


def test_disconnected_white_region_below_the_terrain_is_ignored():
    """Sky through an arch, a window, or a puddle reflection is enclosed sky: it touches no top row,
    so it never defines a boundary."""
    rows = [8] * W
    m = mask_from_rows(rows)
    m[15:18, 3:6] = True                           # a bright patch far below the horizon
    res = simgt.sim_gt_curve(m)
    assert np.allclose(res["rows"], 7.5)           # unchanged by the disconnected region
    assert res["n_enclosed_sky_px"] == 9
    assert res["sky_frac_top_connected"] < res["sky_frac_raw"]


def test_no_sky_image():
    res = simgt.sim_gt_curve(np.zeros((H, W), dtype=bool))
    assert res["status"] == simgt.STATUS_NO_TOP_SKY
    assert not res["valid"].any()
    assert res["valid_frac"] == 0.0
    assert simgt.seam_curve_from_result("obs", res, W, H) is None


def test_all_sky_image():
    res = simgt.sim_gt_curve(np.ones((H, W), dtype=bool))
    assert res["status"] == simgt.STATUS_OK
    assert res["valid"].all()
    assert np.allclose(res["rows"], H - 1.0)       # clamped inside the frame, never at/over the edge
    curve = simgt.seam_curve_from_result("obs", res, W, H)
    assert curve is not None and curve.provenance == "oracle:sim_exact"


def test_seam_curve_exists_only_when_every_column_is_valid():
    full = simgt.sim_gt_curve(mask_from_rows([9] * W))
    curve = simgt.seam_curve_from_result("obs_ok", full, W, H)
    assert curve is not None
    assert curve.row_per_col.size == W
    assert curve.provenance == simgt.SIM_EXACT_PROVENANCE
    curve.validate()


# -- binarity ----------------------------------------------------------------------------------

def test_a_clean_binary_mask_passes_and_reports_two_levels():
    img = (mask_from_rows([9] * W).astype(np.uint8)) * 255
    sky, report = simgt.sky_mask_from_image(img)
    assert report["intermediate_frac"] == 0.0
    assert report["n_distinct_values"] == 2
    assert sky.sum() == 9 * W


def test_an_anti_aliased_mask_is_refused_not_thresholded():
    img = (mask_from_rows([9] * W).astype(np.uint8)) * 255
    img[9, :] = 128                                # one row of grey: 1/20 of the pixels
    with pytest.raises(simgt.SimGtError, match="not binary enough"):
        simgt.sky_mask_from_image(img)


def test_a_tiny_amount_of_grey_is_tolerated_when_the_declared_tolerance_allows_it():
    img = (mask_from_rows([9] * W).astype(np.uint8)) * 255
    img[9, 0] = 128
    sky, report = simgt.sky_mask_from_image(img, max_intermediate_frac=0.01)
    assert report["n_intermediate"] == 1
    assert sky[9, 0]                               # >= 128 counts as sky under the declared rule


def test_rgb_masks_are_accepted():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:9, :, :] = 255
    sky, _ = simgt.sky_mask_from_image(img)
    assert sky[:9].all() and not sky[9:].any()


# -- on-disk forms -----------------------------------------------------------------------------

def test_gt_curve_round_trips_including_invalid_columns(tmp_path):
    rows = [11] * W
    rows[2] = 0
    res = simgt.sim_gt_curve(mask_from_rows(rows))
    path = simgt.write_gt_curve(tmp_path / "gt.csv", res)
    back = simgt.read_gt_curve(path)
    assert np.array_equal(back["valid"], res["valid"])
    assert np.allclose(back["rows"][back["valid"]], res["rows"][res["valid"]])
    assert np.isnan(back["rows"][2])


def test_seam_curve_file_is_the_frozen_col_row_format(tmp_path):
    res = simgt.sim_gt_curve(mask_from_rows([9] * W))
    path = simgt.write_seam_curve(tmp_path / "obs.csv", res)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "col,row"
    assert len(lines) == W + 1
    assert lines[1].startswith("0,")
