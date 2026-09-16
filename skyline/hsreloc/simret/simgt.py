"""Simulator ground-truth sky mask → skyline curve (PROT-SKY-001 §1.2 field 13–14, Part B).

This is **actual ground truth**, not a silver label: the mask comes from the renderer's own sky pass,
so the sky/non-sky assignment is exact by construction rather than inferred from pixels. The
provenance it carries is ``oracle:sim_exact``, which the frozen ``skyline_curve`` module has always
listed among the oracle provenances — no seam change is needed for it.

The conversion rule is deliberately the **same rule** the silver pipeline uses, imported rather than
copied (``hsreloc.extraction.silver.convert.top_connected`` / ``curve_from_mask``, unchanged):

1. threshold the 8-bit mask to a boolean sky mask — white = sky, black = non-sky;
2. keep only the 8-connected sky components that touch **row 0** (top-connected sky). Sky seen
   through an arch, a window, or a gap under a canopy is enclosed sky and never defines a skyline;
3. per column, take the topmost contiguous run of top-connected sky; the boundary is the transition
   just below its last pixel (``r1 + 0.5``);
4. a column with no top-connected sky is **invalid**. No row is invented for it;
5. per image, a status: ``ok`` / ``insufficient_sky`` / ``no_top_connected_sky``.

Two products, on purpose
------------------------
* the **full-fidelity GT curve** (``col,row,valid,reason``) is always written, invalid columns
  included, and is what extraction accuracy is measured against (``COMP-SKY-002`` metrics need the
  valid mask);
* the **seam curve** (``skylines_oracle/<id>.csv``, ``col,row``) is written **only when every column
  is valid**, because the matcher seam requires full-width contiguous coverage and a partial curve
  stored as if complete would fabricate boundaries. A missing seam curve is a *database hole* for a
  reference and an ``EXTRACTION_FAILURE`` for a query — the contracted propagation, unchanged.

That is the brief's "do not force a full-width curve when the mask says no sky exists", made
structural: the fabricating path does not exist.

Binarity
--------
A mask that is not actually binary is not ground truth. ``binarity`` measures the fraction of pixels
that are neither near-black nor near-white; above the declared tolerance the mask is refused with the
measured fraction and a histogram summary, never silently thresholded. Anti-aliased or JPEG-compressed
mask exports are exactly what this catches.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from hsreloc.extraction.curves import (REASON_NO_SKY_AT_TOP, REASON_NONE, make_curve as make_canonical)
from hsreloc.extraction.silver.convert import curve_from_mask, top_connected
from hsreloc.retrieval.skyline_curve import make_curve as make_seam_curve

GT_CONVERT_VERSION = "1.0.0"
SIM_EXACT_PROVENANCE = "oracle:sim_exact"
GT_PRODUCER = "sim_exact_mask"

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_sky"
STATUS_NO_TOP_SKY = "no_top_connected_sky"

#: Default binarity tolerance: at most this fraction of pixels may sit between the two levels.
DEFAULT_MAX_INTERMEDIATE_FRAC = 1e-3
#: Pixels within this distance of 0 or 255 count as a declared level.
LEVEL_TOLERANCE = 8


class SimGtError(Exception):
    """A simulator mask cannot be treated as ground truth."""


def binarity(mask_image: np.ndarray, level_tolerance: int = LEVEL_TOLERANCE) -> dict:
    """How binary an 8-bit mask actually is. Reported, never used to repair."""
    arr = np.asarray(mask_image)
    if arr.ndim == 3:
        arr = arr[..., :3].max(axis=2)
    arr = arr.astype(np.int32)
    low = arr <= int(level_tolerance)
    high = arr >= 255 - int(level_tolerance)
    intermediate = ~(low | high)
    return {
        "intermediate_frac": float(intermediate.mean()),
        "n_intermediate": int(intermediate.sum()),
        "min": int(arr.min()), "max": int(arr.max()),
        "n_distinct_values": int(np.unique(arr).size),
        "level_tolerance": int(level_tolerance),
    }


def sky_mask_from_image(mask_image: np.ndarray, max_intermediate_frac: float = DEFAULT_MAX_INTERMEDIATE_FRAC,
                        level_tolerance: int = LEVEL_TOLERANCE) -> tuple:
    """8-bit mask image → (boolean sky mask, binarity report). Raises if it is not binary enough."""
    report = binarity(mask_image, level_tolerance)
    if report["intermediate_frac"] > float(max_intermediate_frac):
        raise SimGtError(
            f"mask is not binary enough to be ground truth: {report['intermediate_frac']:.4%} of "
            f"pixels are neither black nor white (tolerance {max_intermediate_frac:.4%}); "
            f"{report['n_distinct_values']} distinct values in [{report['min']}, {report['max']}]. "
            f"A mask exported with anti-aliasing, filtering or lossy compression is not exact truth — "
            f"re-export it from the sky render target without post-processing rather than thresholding "
            f"it here.")
    arr = np.asarray(mask_image)
    if arr.ndim == 3:
        arr = arr[..., :3].max(axis=2)
    return (arr.astype(np.int32) >= 128), report


def sim_gt_curve(sky_mask: np.ndarray, min_valid_frac: float = 1.0) -> dict:
    """Top-connected-sky curve for one boolean simulator mask.

    ``min_valid_frac`` defaults to **1.0**: for simulator ground truth over a horizon camera, any
    column without sky is a real property of the scene (a tower filling the column, terrain above the
    frame) and the caller decides what to do with it — but the *seam* curve then does not exist, so a
    less-than-full-width GT image never quietly becomes a full-width curve.
    """
    mask = np.asarray(sky_mask, dtype=bool)
    if mask.ndim != 2:
        raise SimGtError(f"sky mask must be 2-D, got shape {mask.shape}")
    top = top_connected(mask)
    rows, valid = curve_from_mask(top)
    valid_frac = float(valid.mean())
    if not top.any():
        status = STATUS_NO_TOP_SKY
    elif valid_frac < float(min_valid_frac):
        status = STATUS_INSUFFICIENT
    else:
        status = STATUS_OK
    return {
        "rows": rows, "valid": valid, "status": status, "valid_frac": valid_frac,
        "sky_frac_raw": float(mask.mean()),
        "sky_frac_top_connected": float(top.mean()),
        "n_enclosed_sky_px": int((mask & ~top).sum()),
        "gt_convert_version": GT_CONVERT_VERSION,
        "min_valid_frac": float(min_valid_frac),
    }


def canonical_from_result(result: dict, width_px: int, height_px: int):
    """The GT curve in the extraction bench's ``CanonicalCurve`` form (invalid columns preserved)."""
    rows = np.asarray(result["rows"], dtype=np.float64).copy()
    valid = np.asarray(result["valid"], dtype=bool)
    rows[~valid] = np.nan
    reasons = tuple(REASON_NONE if v else REASON_NO_SKY_AT_TOP for v in valid)
    return make_canonical(width_px, height_px, rows, reasons, "ground_truth", GT_PRODUCER,
                          meta={k: v for k, v in result.items() if k not in ("rows", "valid")})


def seam_curve_from_result(observation_id: str, result: dict, width_px: int, height_px: int):
    """The matcher-facing ``SkylineCurve``, or ``None`` when the mask does not cover every column."""
    valid = np.asarray(result["valid"], dtype=bool)
    if not valid.all():
        return None
    return make_seam_curve(observation_id, np.asarray(result["rows"], dtype=np.float64),
                           width_px, height_px, SIM_EXACT_PROVENANCE, f"sim_exact_mask:{observation_id}")


# --------------------------------------------------------------------------------------------------
# on-disk forms
# --------------------------------------------------------------------------------------------------

GT_CURVE_COLUMNS = ["col", "row", "valid", "reason"]


def write_gt_curve(path: Path, result: dict) -> Path:
    """Full-fidelity GT curve: every column, with its validity and reason. Always written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(result["rows"], dtype=np.float64)
    valid = np.asarray(result["valid"], dtype=bool)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GT_CURVE_COLUMNS)
        w.writeheader()
        for c in range(rows.size):
            w.writerow({"col": c, "row": ("" if not valid[c] else f"{float(rows[c]):.4f}"),
                        "valid": "true" if valid[c] else "false",
                        "reason": REASON_NONE if valid[c] else REASON_NO_SKY_AT_TOP})
    return path


def read_gt_curve(path: Path) -> dict:
    """Inverse of :func:`write_gt_curve`."""
    rows_by_col: dict = {}
    valid_by_col: dict = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            c = int(r["col"])
            ok = r["valid"] == "true"
            valid_by_col[c] = ok
            rows_by_col[c] = float(r["row"]) if ok else np.nan
    n = len(rows_by_col)
    if sorted(rows_by_col) != list(range(n)):
        raise SimGtError(f"{path}: GT curve columns are not 0..{n - 1} contiguous")
    return {"rows": np.array([rows_by_col[c] for c in range(n)], dtype=np.float64),
            "valid": np.array([valid_by_col[c] for c in range(n)], dtype=bool)}


def write_seam_curve(path: Path, result: dict) -> Path:
    """``col,row`` full-width curve for the matcher seam. Caller must have checked validity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.asarray(result["rows"], dtype=np.float64)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("col,row\n")
        for c in range(rows.size):
            f.write(f"{c},{float(rows[c])}\n")
    return path
