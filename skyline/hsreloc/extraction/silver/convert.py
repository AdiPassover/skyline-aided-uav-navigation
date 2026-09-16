"""ADE20K label map -> silver skyline curve (EXP-SKY-005).

A **silver label**, never ground truth: the deterministic conversion of an independent semantic
segmentor's output into the same per-column representation our extractor produces, so the two can be
compared. The rule is deliberately simple and is declared in full before the trial runs; it is
chosen by looking at images only ("is this what a person would call the sky/non-sky boundary?") and
never against our extractor's output.

The rule — ``top-connected sky region, topmost contiguous run per column``:

1. ``sky = (label == ADE20K class 2 'sky')`` — one class, no merging.
2. optional morphological closing of ``sky`` (structuring element declared in the config) to bridge
   sky specks between branches, so a canopy reads as an envelope rather than a lace curtain;
3. 8-connected components of ``sky``; keep only components with a pixel in row 0 — **top-connected**.
   Enclosed sky (through windows, arches, gaps under a canopy, reflections) never defines a boundary;
4. per column: the topmost contiguous run of top-connected sky; the boundary is the transition below
   its last pixel (``r1 + 0.5``);
5. columns with no top-connected sky are **invalid** — no boundary is invented;
6. per image: ``ok`` when the valid fraction reaches the declared floor, otherwise an explicit
   status (``insufficient_sky`` / ``no_top_connected_sky``) — the analogue of the extractor's refusal.

Handled explicitly by construction: columns with no sky (5); multiple disconnected sky regions (3);
sky through trees/branches/windows (2 + 3); top-connected vs enclosed sky (3); buildings and
vegetation (they are simply not class 2, so their silhouette *is* the boundary); segmentation
speckle (2, and the topmost-run rule ignores lower specks); columns without a defensible boundary
(5, 6).
"""

from __future__ import annotations

import numpy as np

try:                                                    # cv2 is a project dependency; keep the
    import cv2                                          # import local-failure message obvious
except ImportError as exc:                              # pragma: no cover
    raise ImportError("hsreloc.extraction.silver.convert needs opencv-python") from exc

CONVERT_VERSION = "1.0.0"
SKY_CLASS_INDEX = 2
STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_sky"
STATUS_NO_TOP_SKY = "no_top_connected_sky"


def sky_mask(label: np.ndarray, closing_px: int = 0) -> np.ndarray:
    """Binary sky mask from an ADE20K label map, with an optional closing of `closing_px`."""
    mask = (np.asarray(label) == SKY_CLASS_INDEX)
    if closing_px and closing_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(closing_px), int(closing_px)))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return mask


def top_connected(mask: np.ndarray) -> np.ndarray:
    """Union of the 8-connected components of `mask` that touch the top image row."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return np.zeros_like(mask)
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    keep = np.unique(lab[0][mask[0]]) if mask[0].any() else np.array([], dtype=lab.dtype)
    keep = keep[keep != 0]
    if keep.size == 0:
        return np.zeros_like(mask)
    return np.isin(lab, keep)


def curve_from_mask(mask: np.ndarray) -> tuple:
    """(rows, valid) for a top-connected sky mask: topmost contiguous run per column."""
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    rows = np.full(w, np.nan, dtype=np.float64)
    valid = np.zeros(w, dtype=bool)
    for x in range(w):
        col = mask[:, x]
        idx = np.flatnonzero(col)
        if idx.size == 0:
            continue
        r0 = idx[0]
        breaks = np.flatnonzero(np.diff(idx) > 1)
        r1 = idx[breaks[0]] if breaks.size else idx[-1]
        rows[x] = min(float(r1) + 0.5, h - 1.0)
        valid[x] = True
    return rows, valid


def silver_curve(label: np.ndarray, closing_px: int = 0, min_valid_frac: float = 0.5) -> dict:
    """Full conversion for one label map. Returns rows/valid/status and descriptive statistics."""
    raw = sky_mask(label, closing_px)
    top = top_connected(raw)
    rows, valid = curve_from_mask(top)
    valid_frac = float(valid.mean())
    if not top.any():
        status = STATUS_NO_TOP_SKY
    elif valid_frac < min_valid_frac:
        status = STATUS_INSUFFICIENT
    else:
        status = STATUS_OK
    n_comp = int(cv2.connectedComponents(raw.astype(np.uint8), connectivity=8)[0] - 1)
    return {
        "rows": rows, "valid": valid, "status": status,
        "valid_frac": valid_frac,
        "sky_frac_raw": float(raw.mean()),
        "sky_frac_top_connected": float(top.mean()),
        "n_sky_components": n_comp,
        "convert_version": CONVERT_VERSION,
        "closing_px": int(closing_px),
        "min_valid_frac": float(min_valid_frac),
    }
