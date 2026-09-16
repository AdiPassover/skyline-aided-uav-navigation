"""Trajectory error metrics (FR-030 onward).

Every metric here operates on ALREADY-ALIGNED points -- the output of
`alignment.fit_sim2(...).apply(...)`. This module has no alignment logic of its own and
does not know or care how the alignment was obtained. That separation is what makes
FR-067 ("no primary metric may re-estimate scale per segment") a property of *which*
alignment result is passed in, not of the metric code itself.

Phase 3 added absolute trajectory error (ATE). Phase 5 adds the rest of the primary set
(FR-030): relative pose error (RPE) under the single global scale, translational drift
normalised by distance travelled, yaw error/drift, and endpoint error including its
closed-loop variant. `relative_pose_error_per_segment_rescaled` is the sole exception to
the "already-aligned points" rule -- it is an explicitly labelled DIAGNOSTIC (FR-067,
DEC-003) that performs its own local fit per segment, precisely so its difference against
the global-scale RPE can quantify scale drift. It takes RAW estimator points for exactly
that reason; every other function in this module takes already-aligned points and must
never import `alignment.fit_sim2`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from naveval.frames import shortest_angle_diff_deg


def _cumulative_distance(points: np.ndarray) -> np.ndarray:
    """Cumulative Euclidean distance travelled up to and including each point, starting
    at 0.0 for the first point. The building block for every length- or
    distance-normalised metric in this module."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return np.zeros(0)
    deltas = np.diff(points, axis=0)
    step_lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate([[0.0], np.cumsum(step_lengths)])


def path_length(points: np.ndarray) -> float:
    """Cumulative Euclidean distance travelled along a sequence of (N, 2) points."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        return 0.0
    deltas = np.diff(points, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def absolute_trajectory_error(aligned_points: np.ndarray, gt_points: np.ndarray) -> dict:
    """Absolute Trajectory Error: per-point Euclidean error after alignment (FR-030),
    summarised as RMSE. Also reports the RMSE normalised by ground-truth path length
    (FR-033), so the result stays interpretable given that the raw estimate is
    non-metric.

    A metric computed after scale alignment demonstrates trajectory *shape* fidelity,
    not metric accuracy (FR-032) -- callers reporting this value must state that.
    """
    aligned_points = np.asarray(aligned_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)
    if aligned_points.shape != gt_points.shape:
        raise ValueError(
            f"aligned_points and gt_points must have the same shape, got "
            f"{aligned_points.shape} and {gt_points.shape}"
        )
    if aligned_points.shape[0] == 0:
        raise ValueError("Cannot compute ATE over zero points")

    per_point_errors = np.linalg.norm(aligned_points - gt_points, axis=1)
    rmse = float(np.sqrt((per_point_errors ** 2).mean()))
    length = path_length(gt_points)

    return {
        "rmse": rmse,
        "mean": float(per_point_errors.mean()),
        "max": float(per_point_errors.max()),
        "per_point": per_point_errors,
        "path_length_m": length,
        "rmse_normalized": rmse / length if length > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Relative pose error (FR-030, FR-034, FR-067)
# ---------------------------------------------------------------------------


def relative_pose_error(aligned_points: np.ndarray, gt_points: np.ndarray, length_m: float) -> Optional[dict]:
    """Relative pose error over one sub-trajectory length, under the single GLOBAL
    Sim(2) scale (FR-067) -- `aligned_points` must already be the output of one
    whole-trajectory alignment, exactly like `absolute_trajectory_error`. This function
    fits nothing; it only measures.

    For each starting frame, finds the first later frame whose ground-truth path
    distance from the start reaches `length_m`, then compares the estimated and
    ground-truth displacement vectors between those two frames. Because the global
    scale is fixed for every segment, a scale that drifts over the flight shows up as a
    real, non-cancelling error here -- which is the entire point of computing RPE this
    way rather than re-fitting per segment (contrast `relative_pose_error_per_segment_rescaled`).

    Returns None when the trajectory's total path length is shorter than `length_m`, or
    no segment could be formed -- there is nothing to measure, not a zero-error result.
    """
    aligned_points = np.asarray(aligned_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)
    if aligned_points.shape != gt_points.shape:
        raise ValueError(
            f"aligned_points and gt_points must have the same shape, got "
            f"{aligned_points.shape} and {gt_points.shape}"
        )
    n = gt_points.shape[0]
    if n < 2:
        return None

    cumdist = _cumulative_distance(gt_points)
    total = float(cumdist[-1])
    if total < length_m:
        return None

    errors = []
    for i in range(n):
        target = cumdist[i] + length_m
        if target > total:
            break
        j = int(np.searchsorted(cumdist, target))
        if j >= n or j == i:
            continue
        delta_est = aligned_points[j] - aligned_points[i]
        delta_gt = gt_points[j] - gt_points[i]
        errors.append(float(np.linalg.norm(delta_est - delta_gt)))

    if not errors:
        return None

    errors_arr = np.array(errors)
    rmse = float(np.sqrt((errors_arr ** 2).mean()))
    return {
        "rmse": rmse,
        "rmse_normalized": rmse / length_m,
        "n_segments": len(errors),
        "length_m": length_m,
        "scale_basis": "global_fitted",
    }


def relative_pose_error_per_segment_rescaled(
    raw_est_points: np.ndarray, gt_points: np.ndarray, length_m: float
) -> Optional[dict]:
    """DIAGNOSTIC ONLY (FR-067, DEC-003). Re-fits a fresh Sim(2) transform per segment
    from RAW (unaligned) estimator points, so any scale drift *within* a segment is
    locally absorbed and made invisible -- the opposite of `relative_pose_error`. It
    exists so the difference between the two can be reported as the scale-drift
    contribution (FR-067's last sentence), never as a primary error figure on its own.

    Segments with fewer than 3 points are skipped: a 2-point Sim(2) fit is always
    exactly solvable (4 unknowns, 4 equations), so its residual would trivially be zero
    regardless of real drift -- not a measurement, a tautology.
    """
    from naveval.alignment import AlignmentRefusedError, fit_sim2

    raw_est_points = np.asarray(raw_est_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)
    if raw_est_points.shape != gt_points.shape:
        raise ValueError(
            f"raw_est_points and gt_points must have the same shape, got "
            f"{raw_est_points.shape} and {gt_points.shape}"
        )
    n = gt_points.shape[0]
    if n < 2:
        return None

    cumdist = _cumulative_distance(gt_points)
    total = float(cumdist[-1])
    if total < length_m:
        return None

    errors = []
    for i in range(n):
        target = cumdist[i] + length_m
        if target > total:
            break
        j = int(np.searchsorted(cumdist, target))
        if j >= n or j < i + 2:
            continue
        try:
            local_fit = fit_sim2(raw_est_points[i:j + 1], gt_points[i:j + 1])
        except AlignmentRefusedError:
            continue
        aligned_local = local_fit.apply(raw_est_points[i:j + 1])
        delta_est = aligned_local[-1] - aligned_local[0]
        delta_gt = gt_points[j] - gt_points[i]
        errors.append(float(np.linalg.norm(delta_est - delta_gt)))

    if not errors:
        return None

    errors_arr = np.array(errors)
    rmse = float(np.sqrt((errors_arr ** 2).mean()))
    return {
        "rmse": rmse,
        "rmse_normalized": rmse / length_m,
        "n_segments": len(errors),
        "length_m": length_m,
        "scale_basis": "per_segment_fitted",
    }


# ---------------------------------------------------------------------------
# Translational drift normalised by distance travelled (FR-030)
# ---------------------------------------------------------------------------


def translational_drift_rate(aligned_points: np.ndarray, gt_points: np.ndarray) -> dict:
    """Rate at which position error grows with distance travelled.

    Normalised by PATH LENGTH (cumulative distance actually flown), not straight-line
    displacement, so a trajectory that loops back on itself is correctly treated as
    having travelled the distance it actually flew rather than ~0 (spec Edge Cases;
    `path_length` above is exactly this, reused here per-point via `_cumulative_distance`).

    Distinct from `ate_rmse_normalised` (Phase 3): that is an aggregate dispersion
    figure (RMSE / total path length). This is a GROWTH RATE -- the least-squares slope
    of per-point error against cumulative distance, forced through the origin because
    the alignment's gauge freedoms guarantee the very first evaluated point can be made
    exact, so a nonzero intercept would only be fit noise, not a real offset.
    """
    aligned_points = np.asarray(aligned_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)
    if aligned_points.shape != gt_points.shape:
        raise ValueError(
            f"aligned_points and gt_points must have the same shape, got "
            f"{aligned_points.shape} and {gt_points.shape}"
        )
    if aligned_points.shape[0] < 2:
        raise ValueError("Need at least 2 points to compute a drift rate")

    errors = np.linalg.norm(aligned_points - gt_points, axis=1)
    distances = _cumulative_distance(gt_points)
    denom = float((distances ** 2).sum())
    if denom <= 0.0:
        return {"drift_rate": float("nan"), "total_distance_m": 0.0, "final_error_m": float(errors[-1])}

    slope = float((errors * distances).sum() / denom)
    return {
        "drift_rate": slope,
        "total_distance_m": float(distances[-1]),
        "final_error_m": float(errors[-1]),
    }


# ---------------------------------------------------------------------------
# Yaw error and yaw drift (FR-031, FR-068) -- independent of position metrics
# ---------------------------------------------------------------------------


def yaw_error_metric(aligned_yaw_deg: np.ndarray, gt_heading_deg: np.ndarray) -> dict:
    """Per-frame yaw error using shortest-angle differences (FR-031).

    Touches only heading arrays -- no position data enters this computation, which is
    what makes "yaw MUST be reported independently of position error" a property
    verifiable by the function signature, not just an assertion in prose.
    """
    aligned_yaw_deg = np.asarray(aligned_yaw_deg, dtype=np.float64)
    gt_heading_deg = np.asarray(gt_heading_deg, dtype=np.float64)
    if aligned_yaw_deg.shape != gt_heading_deg.shape:
        raise ValueError(
            f"aligned_yaw_deg and gt_heading_deg must have the same shape, got "
            f"{aligned_yaw_deg.shape} and {gt_heading_deg.shape}"
        )
    if aligned_yaw_deg.shape[0] == 0:
        raise ValueError("Cannot compute yaw error over zero frames")

    errors = np.array(
        [shortest_angle_diff_deg(float(a), float(g)) for a, g in zip(aligned_yaw_deg, gt_heading_deg)]
    )
    rmse = float(np.sqrt((errors ** 2).mean()))
    return {
        "rmse": rmse,
        "mean_signed": float(errors.mean()),
        "max_abs": float(np.abs(errors).max()),
        "per_frame": errors,
    }


def yaw_drift_rate(aligned_yaw_deg: np.ndarray, gt_heading_deg: np.ndarray, elapsed_time_s: np.ndarray) -> dict:
    """Rate at which yaw error grows over elapsed time (deg/s).

    Normalised by TIME, not distance: a hovering flight travels ~0 distance but can
    still accumulate real heading drift, which a distance-normalised rate would report
    as infinite or undefined. Recovered as the least-squares slope of shortest-angle
    yaw error against elapsed time; unlike the position drift rate, the intercept is
    *not* forced to zero, since a genuine constant yaw offset can survive the position
    alignment's single rotation fit (COMP-001-relevant: estimator yaw and estimator
    position rotation are assumed to share one coordinate frame, but nothing guarantees
    the fitted rotation zeroes yaw error at the first frame specifically).
    """
    errors = np.array(
        [shortest_angle_diff_deg(float(a), float(g)) for a, g in zip(aligned_yaw_deg, gt_heading_deg)]
    )
    t = np.asarray(elapsed_time_s, dtype=np.float64)
    if t.shape[0] != errors.shape[0]:
        raise ValueError("elapsed_time_s must have the same length as the yaw arrays")
    if t.shape[0] < 2:
        raise ValueError("Need at least 2 frames to compute a drift rate")

    span = float(t[-1] - t[0])
    if span <= 0.0:
        return {"drift_deg_per_s": float("nan"), "n_frames": int(t.shape[0])}

    slope, intercept = np.polyfit(t, errors, 1)
    return {"drift_deg_per_s": float(slope), "intercept_deg": float(intercept), "n_frames": int(t.shape[0])}


# ---------------------------------------------------------------------------
# Endpoint error (FR-030)
# ---------------------------------------------------------------------------


def endpoint_error(aligned_points: np.ndarray, gt_points: np.ndarray) -> float:
    """Open-loop endpoint error: aligned estimated final position vs. the measured
    final ground-truth position."""
    aligned_points = np.asarray(aligned_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)
    if aligned_points.shape[0] == 0:
        raise ValueError("Cannot compute endpoint error over zero points")
    return float(np.linalg.norm(aligned_points[-1] - gt_points[-1]))


def endpoint_error_closed_loop(aligned_points: np.ndarray, gt_points: np.ndarray) -> float:
    """Closed-loop endpoint error: aligned estimated final position vs. the *first*
    ground-truth sample -- the physically known launch point a loop trajectory returns
    to, not a second, independently noisy ground-truth measurement at the end. This is
    the strongest evidence available from metre-level GNSS (spec.md support matrix):
    trusting "the drone came back to where it started" costs nothing beyond knowing the
    flight was a loop, unlike trusting the absolute accuracy of two separate GNSS fixes.

    Callers MUST only use this for a dataset actually labelled as a closed-loop
    trajectory; this function has no way to detect that itself.
    """
    aligned_points = np.asarray(aligned_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)
    if aligned_points.shape[0] == 0:
        raise ValueError("Cannot compute endpoint error over zero points")
    return float(np.linalg.norm(aligned_points[-1] - gt_points[0]))


# ---------------------------------------------------------------------------
# Discarded vertical motion (FR-028)
# ---------------------------------------------------------------------------


def vertical_motion_discarded(gt_up_m: np.ndarray) -> Optional[dict]:
    """Magnitude of real vertical motion discarded by the planar Sim(2) comparison
    (FR-028): the estimator reports z identically absent for every frame (no height
    estimation yet -- FR-015), while ground truth may carry real altitude variation
    that the planar alignment simply never sees.

    Returns None when no altitude ground truth exists at all -- distinct from a
    genuine near-zero result, which means the flight really was close to level.
    """
    gt_up_m = np.asarray(gt_up_m, dtype=np.float64)
    valid = gt_up_m[~np.isnan(gt_up_m)]
    if valid.size == 0:
        return None
    return {
        "range_m": float(valid.max() - valid.min()),
        "rms_about_mean_m": float(np.sqrt(((valid - valid.mean()) ** 2).mean())),
        "n_samples": int(valid.size),
    }


# ---------------------------------------------------------------------------
# Runtime (FR-047, FR-048) -- reported separately from accuracy, never as onboard
# performance unless the manifest says so (FR-049; enforced in report.py, not here)
# ---------------------------------------------------------------------------


def runtime_statistics(process_time_ns: np.ndarray) -> dict:
    """Full distribution of per-frame processing time (FR-047) -- median and tail
    percentiles, not only a mean, since a mean alone hides exactly the occasional slow
    frames that matter for a real-time budget. Reported in milliseconds for
    readability; the run record itself stays in nanoseconds
    (contracts/run-record.md) and estimator-only (excludes image I/O, per VoRunner).
    """
    ns = np.asarray(process_time_ns, dtype=np.float64)
    if ns.size == 0:
        raise ValueError("Cannot compute runtime statistics over zero frames")
    if np.any(ns < 0):
        raise ValueError("process_time_ns must be >= 0 (contracts/run-record.md)")

    ms = ns / 1.0e6
    return {
        "n_frames": int(ms.size),
        "mean_ms": float(ms.mean()),
        "median_ms": float(np.median(ms)),
        "p90_ms": float(np.percentile(ms, 90)),
        "p99_ms": float(np.percentile(ms, 99)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "stddev_ms": float(ms.std()),
    }
