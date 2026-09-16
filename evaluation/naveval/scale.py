"""Per-segment fitted scale series (FR-035) -- DIAGNOSTIC ONLY (FR-067, DEC-003).

The mechanism that makes the `COMP-001` §4 recentering question empirically
answerable: comparing fitted scale across segments that do and do not span a recenter
is precisely the measurement. Nothing here feeds back into the primary metrics -- the
single global Sim(2) fit computed once over the whole trajectory (alignment.fit_sim2)
remains authoritative regardless of what this module reports. A per-segment fit here
existing alongside an unrelated global fit is the entire point: their disagreement is
the observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from naveval.alignment import AlignmentRefusedError, fit_sim2
from naveval.metrics import _cumulative_distance


@dataclass(frozen=True)
class ScaleSeriesPoint:
    segment_id: int
    time_start_s: float
    distance_start_m: float
    scale: Optional[float]
    scale_ratio_to_global: Optional[float]
    spans_recenter: bool
    n_poses: int
    refusal_reason: Optional[str] = None


def compute_scale_series(
    segments,
    raw_est_points: np.ndarray,
    gt_points: np.ndarray,
    timestamps_s: np.ndarray,
    global_scale: float,
    recenter_positions=(),
) -> list[ScaleSeriesPoint]:
    """One `ScaleSeriesPoint` per segment: a fresh LOCAL Sim(2) fit over the segment's
    own raw estimator points vs. ground truth, reported against both elapsed time and
    distance travelled (FR-035).

    `spans_recenter` is True only when a recenter/restart occurred STRICTLY INSIDE the
    segment (not merely at its start boundary) -- segments derived from reference-id
    boundaries (`segments.segments_from_reference_changes`) therefore always report
    False here by construction, since every recenter in that scheme falls exactly on a
    segment boundary; the flag is informative for metadata- or config-defined segments
    that may straddle one.

    A segment with fewer than 2 poses, or whose geometry the alignment refuses (e.g.
    near-stationary), reports `scale=None` with `refusal_reason` set -- never a
    fabricated or silently-zero scale.
    """
    raw_est_points = np.asarray(raw_est_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    recenter_positions = set(recenter_positions)
    cumdist = _cumulative_distance(gt_points)

    points = []
    for seg in segments:
        sl = slice(seg.start_index, seg.end_index + 1)
        seg_est = raw_est_points[sl]
        seg_gt = gt_points[sl]
        n_poses = int(seg_est.shape[0])

        scale = None
        ratio = None
        refusal_reason = None
        if n_poses >= 2:
            try:
                local_fit = fit_sim2(seg_est, seg_gt)
                scale = local_fit.scale
                ratio = (scale / global_scale) if global_scale not in (0.0, None) else None
            except AlignmentRefusedError as exc:
                refusal_reason = str(exc)
        else:
            refusal_reason = f"segment has only {n_poses} pose(s); need >= 2 to fit a local scale"

        spans_recenter = any(seg.start_index < p <= seg.end_index for p in recenter_positions)

        points.append(ScaleSeriesPoint(
            segment_id=seg.segment_id,
            time_start_s=float(timestamps_s[seg.start_index]),
            distance_start_m=float(cumdist[seg.start_index]),
            scale=scale,
            scale_ratio_to_global=ratio,
            spans_recenter=spans_recenter,
            n_poses=n_poses,
            refusal_reason=refusal_reason,
        ))
    return points
