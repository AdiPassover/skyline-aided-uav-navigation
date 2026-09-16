"""Estimator failure and event statistics (FR-042 through FR-046).

Distinguishes events read directly from the run record ("from_run_record": restarts,
recenters -- observable facts the estimator itself reported) from events DERIVED by the
evaluator by comparing the trajectory or its error against its own observed
distribution ("pose_discontinuity", "large_error_excursion"). Derived thresholds are
always expressed relative to the observed distribution, never as absolute constants
invented before any data exists (contracts/evaluation-outputs.md's config.json
`event_detection` block), and the exact rule string used is stored on every derived
event so events from different configurations are never silently compared as if they
meant the same thing.

A stitch failure and the restart it triggers are the same observable moment in this
implementation: `StitchingEstimator.processFrame` restarts unconditionally whenever
`stitch.process()` returns false (COMP-001), so `VoRunner` never records a failure that
did not also restart. `extract_run_record_events` therefore reads the `restart` event
column value directly, per contracts/evaluation-outputs.md's own events.csv example,
rather than inventing a separate `stitch_failure` type for the same rows.

Error attribution (`error_before_m`/`error_after_m`) is a TEMPORAL association, not a
causal claim: an event and a nearby error increase occurring close together in time is
evidence worth recording for later confidence work (FR-046), not proof the event
CAUSED the error. Downstream consumers must not read a gap between the two values as
"the event caused this much error" without independent corroboration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

FROM_RUN_RECORD = "from_run_record"


@dataclass(frozen=True)
class Event:
    frame_index: int
    timestamp_s: float
    type: str  # "restart" | "recenter" | "pose_discontinuity" | "large_error_excursion"
    detection_rule: str
    error_before_m: Optional[float] = None
    error_after_m: Optional[float] = None


def extract_run_record_events(run_record) -> list[Event]:
    """Stitch restarts and recenters, read directly from the run record's `event`
    column (FR-043) -- facts the estimator itself reported, not inferred."""
    events = []
    for fe in run_record.frame_estimates:
        if fe.event in ("restart", "recenter"):
            events.append(Event(
                frame_index=fe.frame_index,
                timestamp_s=fe.timestamp_s,
                type=fe.event,
                detection_rule=FROM_RUN_RECORD,
            ))
    return events


def detect_pose_discontinuities(
    frame_indices: np.ndarray,
    timestamps_s: np.ndarray,
    aligned_points: np.ndarray,
    threshold_multiple: float = 5.0,
) -> list[Event]:
    """Frames whose step size (distance from the previous evaluated frame, in
    ground-truth units) exceeds `threshold_multiple` times the MEDIAN step size over
    the whole evaluated trajectory -- a threshold relative to the observed
    distribution (contracts/evaluation-outputs.md's config.json convention:
    `"pose_discontinuity_rule": "step > 5 * median_step"`), not an absolute constant
    invented without data.
    """
    n = aligned_points.shape[0]
    if n < 3:
        return []
    steps = np.linalg.norm(np.diff(aligned_points, axis=0), axis=1)
    median_step = float(np.median(steps))
    if median_step <= 0.0:
        return []

    rule = f"step > {threshold_multiple:g} * median_step (median_step={median_step:.6g}m)"
    events = []
    for i in range(steps.shape[0]):
        if steps[i] > threshold_multiple * median_step:
            arrival = i + 1  # the step FROM i TO i+1 is attributed to the arrival frame
            events.append(Event(
                frame_index=int(frame_indices[arrival]),
                timestamp_s=float(timestamps_s[arrival]),
                type="pose_discontinuity",
                detection_rule=rule,
            ))
    return events


def detect_large_error_excursions(
    frame_indices: np.ndarray,
    timestamps_s: np.ndarray,
    per_frame_error_m: np.ndarray,
    stddev_multiple: float = 3.0,
) -> list[Event]:
    """Frames whose position error exceeds mean + `stddev_multiple` * stddev of the
    observed error distribution -- relative to the observed distribution for the same
    reason as `detect_pose_discontinuities` (config.json's
    `"large_error_excursion_rule": "error > mean + 3 * stddev"`)."""
    n = per_frame_error_m.shape[0]
    if n < 2:
        return []
    mean_err = float(per_frame_error_m.mean())
    std_err = float(per_frame_error_m.std())
    if std_err <= 0.0:
        return []

    threshold = mean_err + stddev_multiple * std_err
    rule = f"error > mean + {stddev_multiple:g} * stddev (mean={mean_err:.6g}m, stddev={std_err:.6g}m)"
    events = []
    for i in range(n):
        if per_frame_error_m[i] > threshold:
            events.append(Event(
                frame_index=int(frame_indices[i]),
                timestamp_s=float(timestamps_s[i]),
                type="large_error_excursion",
                detection_rule=rule,
            ))
    return events


def attribute_error(events: list[Event], frame_indices: np.ndarray, per_frame_error_m: np.ndarray) -> list[Event]:
    """Attach `error_before_m`/`error_after_m` to each event: the nearest available
    per-frame error sample strictly before, and strictly after, the event's frame
    index. `None` on either side when no sample exists there (no ground truth at all,
    or the event sits at the very start/end of the evaluated span) -- absence must
    never be silently reported as zero error.

    A TEMPORAL association only (see module docstring) -- not a causal claim.
    """
    frame_indices = np.asarray(frame_indices)
    if frame_indices.size == 0:
        return [replace(e, error_before_m=None, error_after_m=None) for e in events]

    result = []
    for e in events:
        # searchsorted gives the position of the first sample with frame_index >=
        # e.frame_index. The "before" sample (strictly earlier) is always at pos-1
        # regardless of whether the event's own frame happened to be synced; the
        # "after" sample (strictly later) is at pos+1 if the event frame IS synced
        # (skip past it), or at pos itself if it is not (nothing to skip).
        pos = int(np.searchsorted(frame_indices, e.frame_index))
        is_synced_here = pos < frame_indices.size and frame_indices[pos] == e.frame_index

        before_pos = pos - 1
        after_pos = pos + 1 if is_synced_here else pos

        before = float(per_frame_error_m[before_pos]) if before_pos >= 0 else None
        after = float(per_frame_error_m[after_pos]) if after_pos < frame_indices.size else None

        result.append(replace(e, error_before_m=before, error_after_m=after))
    return result


def success_rate_stats(run_record) -> dict:
    """Proportion of frames on which the estimator updated successfully (FR-042), and
    inter-restart-interval statistics -- both computable from the run record alone,
    with NO ground truth required (FR-045)."""
    successes = run_record.success
    n = int(successes.size)
    success_rate = float(successes.sum()) / n if n > 0 else float("nan")

    restart_timestamps = [fe.timestamp_s for fe in run_record.frame_estimates if fe.event == "restart"]
    n_recenters = sum(1 for fe in run_record.frame_estimates if fe.event == "recenter")

    intervals_s = list(np.diff(restart_timestamps)) if len(restart_timestamps) >= 2 else []

    return {
        "n_frames": n,
        "n_success": int(successes.sum()),
        "success_rate": success_rate,
        "n_restarts": len(restart_timestamps),
        "n_recenters": n_recenters,
        "mean_inter_restart_interval_s": float(np.mean(intervals_s)) if intervals_s else None,
        "median_inter_restart_interval_s": float(np.median(intervals_s)) if intervals_s else None,
    }
