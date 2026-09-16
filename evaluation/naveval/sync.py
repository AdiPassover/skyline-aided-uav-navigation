"""Frame <-> ground-truth time synchronisation (FR-009 through FR-012).

Ground truth is generally sampled well below the frame rate, so interpolation is
unavoidable. Position is interpolated linearly; heading is interpolated along the
shortest arc (frames.py), because linear interpolation of raw degree values is wrong
across the 0/360 wrap. Frames outside the ground-truth time span are excluded and
counted, never extrapolated -- an extrapolated position is not a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from naveval.alignment import rotation_matrix
from naveval.frames import interpolate_heading_deg, normalize_heading_deg


@dataclass
class SyncResult:
    """Ground truth interpolated onto the frame timestamps that could be synchronized."""

    frame_indices: np.ndarray  # indices into the original frame arrays that were synced
    east: np.ndarray
    north: np.ndarray
    heading: np.ndarray  # NaN where no ground-truth heading exists at all
    has_heading: bool

    excluded_out_of_span_count: int
    excluded_insufficient_valid_gt_count: int

    synchronization_uncertainty_s: float  # residual timing uncertainty to report alongside metrics (FR-011)


def apply_clock_offset(gt_timestamps: np.ndarray, clock_offset_s: float) -> np.ndarray:
    """Shift ground-truth timestamps onto the frame clock's time axis.

    Per contracts/dataset.md: frame_time = gt_time + clock_offset_s.
    """
    return np.asarray(gt_timestamps, dtype=np.float64) + clock_offset_s


def synchronize(
    frame_timestamps: np.ndarray,
    gt_timestamps: np.ndarray,
    gt_east: np.ndarray,
    gt_north: np.ndarray,
    gt_heading: Optional[np.ndarray] = None,
    gt_valid: Optional[np.ndarray] = None,
    clock_offset_s: float = 0.0,
    clock_drift_s_per_s: float = 0.0,
    synchronization_uncertainty_s: float = 0.0,
) -> SyncResult:
    """Interpolate ground truth onto frame timestamps.

    A frame is excluded (and counted, not silently dropped) when it falls outside the
    valid ground-truth time span, or when the ground-truth samples bracketing it were
    marked invalid, leaving no valid coverage to interpolate from.
    """
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.float64)
    gt_timestamps = np.asarray(gt_timestamps, dtype=np.float64)
    gt_east = np.asarray(gt_east, dtype=np.float64)
    gt_north = np.asarray(gt_north, dtype=np.float64)
    n_gt = gt_timestamps.shape[0]

    if gt_valid is None:
        gt_valid = np.ones(n_gt, dtype=bool)
    else:
        gt_valid = np.asarray(gt_valid, dtype=bool)

    has_heading = gt_heading is not None and not np.all(np.isnan(gt_heading))
    if gt_heading is None:
        gt_heading = np.full(n_gt, np.nan)
    else:
        gt_heading = np.asarray(gt_heading, dtype=np.float64)

    # gt_true_time = elapsed-time-scaled by drift, then shifted onto the frame clock.
    # A pure clock_offset_s (no drift) reduces to the simple shift documented above.
    if n_gt > 0 and clock_drift_s_per_s != 0.0:
        anchor = gt_timestamps[0]
        gt_on_frame_clock = anchor + (gt_timestamps - anchor) * (1.0 + clock_drift_s_per_s) + clock_offset_s
    else:
        gt_on_frame_clock = apply_clock_offset(gt_timestamps, clock_offset_s)

    valid_mask = gt_valid
    valid_times = gt_on_frame_clock[valid_mask]
    valid_east = gt_east[valid_mask]
    valid_north = gt_north[valid_mask]
    valid_heading = gt_heading[valid_mask]

    if valid_times.size < 2:
        # No usable ground truth at all: every frame is excluded.
        return SyncResult(
            frame_indices=np.array([], dtype=np.int64),
            east=np.array([]),
            north=np.array([]),
            heading=np.array([]),
            has_heading=has_heading,
            excluded_out_of_span_count=0,
            excluded_insufficient_valid_gt_count=int(frame_timestamps.size),
            synchronization_uncertainty_s=synchronization_uncertainty_s,
        )

    span_lo, span_hi = valid_times[0], valid_times[-1]

    kept_indices = []
    east_out = []
    north_out = []
    heading_out = []
    excluded_out_of_span = 0
    excluded_insufficient_valid = 0

    for i, t in enumerate(frame_timestamps):
        if t < span_lo or t > span_hi:
            excluded_out_of_span += 1
            continue

        # Locate the bracketing valid samples for this frame time.
        j = int(np.searchsorted(valid_times, t))
        if j == 0:
            lo_idx = hi_idx = 0
        elif j >= valid_times.size:
            lo_idx = hi_idx = valid_times.size - 1
        else:
            lo_idx, hi_idx = j - 1, j

        t_lo, t_hi = valid_times[lo_idx], valid_times[hi_idx]
        frac = 0.0 if t_hi == t_lo else (t - t_lo) / (t_hi - t_lo)

        east_i = valid_east[lo_idx] + frac * (valid_east[hi_idx] - valid_east[lo_idx])
        north_i = valid_north[lo_idx] + frac * (valid_north[hi_idx] - valid_north[lo_idx])

        if has_heading:
            heading_i = interpolate_heading_deg(float(valid_heading[lo_idx]), float(valid_heading[hi_idx]), frac)
        else:
            heading_i = float("nan")

        kept_indices.append(i)
        east_out.append(east_i)
        north_out.append(north_i)
        heading_out.append(heading_i)

    return SyncResult(
        frame_indices=np.array(kept_indices, dtype=np.int64),
        east=np.array(east_out, dtype=np.float64),
        north=np.array(north_out, dtype=np.float64),
        heading=np.array(heading_out, dtype=np.float64),
        has_heading=has_heading,
        excluded_out_of_span_count=excluded_out_of_span,
        excluded_insufficient_valid_gt_count=excluded_insufficient_valid,
        synchronization_uncertainty_s=synchronization_uncertainty_s,
    )


def estimate_clock_offset(
    frame_timestamps: np.ndarray,
    gt_timestamps: np.ndarray,
    frame_signal: np.ndarray,
    gt_signal: np.ndarray,
    search_range_s: float,
    search_step_s: float = 0.01,
) -> tuple:
    """Estimate a constant clock offset by cross-correlating two matched scalar signals.

    Returns (best_offset_s, residual_uncertainty_s). A minimal, deterministic
    implementation: grid-search offsets in [-search_range_s, search_range_s], for each
    candidate resample gt_signal onto frame_timestamps - offset and score by sum of
    squared differences against frame_signal; pick the minimum. The step size is the
    reported residual uncertainty, since that is the resolution of the search.

    Requires frame_signal and gt_signal to be defined on a comparable scale (e.g. both
    a speed profile derived from consecutive positions) -- reconciling their scales is
    the caller's responsibility.
    """
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.float64)
    gt_timestamps = np.asarray(gt_timestamps, dtype=np.float64)
    frame_signal = np.asarray(frame_signal, dtype=np.float64)
    gt_signal = np.asarray(gt_signal, dtype=np.float64)

    candidates = np.arange(-search_range_s, search_range_s + search_step_s, search_step_s)
    best_offset = 0.0
    best_score = np.inf

    for offset in candidates:
        resampled = np.interp(frame_timestamps - offset, gt_timestamps, gt_signal, left=np.nan, right=np.nan)
        mask = ~np.isnan(resampled)
        if mask.sum() < 2:
            continue
        score = float(np.sum((resampled[mask] - frame_signal[mask]) ** 2))
        if score < best_score:
            best_score = score
            best_offset = float(offset)

    return best_offset, search_step_s


def normalize_to_start_pose(
    east: np.ndarray, north: np.ndarray, heading: np.ndarray
) -> tuple:
    """Re-express a trajectory relative to the pose at its first sample (FR-012).

    The first sample becomes the origin with heading 0. A compass-frame rotation by the
    first heading h0 corresponds to the standard 2D rotation matrix rotation_matrix(h0)
    applied to (east, north) -- verified against heading_to_unit_vector directly rather
    than assumed, given how easily this class of sign error creeps in (see gap G13a).
    """
    east = np.asarray(east, dtype=np.float64)
    north = np.asarray(north, dtype=np.float64)
    heading = np.asarray(heading, dtype=np.float64)

    e0, n0, h0 = east[0], north[0], heading[0]
    de = east - e0
    dn = north - n0
    pts = np.stack([de, dn], axis=1) @ rotation_matrix(h0).T

    heading_new = np.array([normalize_heading_deg(h - h0) for h in heading])
    return pts[:, 0], pts[:, 1], heading_new
