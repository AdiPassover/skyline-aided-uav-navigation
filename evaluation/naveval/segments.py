"""Segment definition and per-segment metrics (FR-038 through FR-041).

A segment is a labelled sub-interval of the EVALUATED (synced) trajectory, addressed by
position in the already-aligned/already-synced arrays (0-based, contiguous with the
synced-frame order) -- not by raw `frame_index`, since not every dataset frame
necessarily has ground truth to evaluate against.

Per-segment metrics reuse the SAME metric functions Phases 3 and 5 already validated,
applied to a slice of the globally-aligned points. The single global Sim(2) fit remains
authoritative (FR-067); nothing in this module re-aligns anything. `scale.py`'s
per-segment scale series is the sole exception, and it stays an explicitly labelled
diagnostic (DEC-003).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

# data-model.md's Segment.label enum. A definition supplying anything else is coerced
# to "custom" rather than rejected -- segment labelling is a convenience for grouping
# results, not a closed vocabulary the framework depends on for correctness.
VALID_LABELS = {"straight", "hover", "turn", "loop", "altitude_change", "long_duration", "custom"}


@dataclass(frozen=True)
class Segment:
    segment_id: int
    label: str
    start_index: int  # position in the evaluated (synced) arrays, inclusive
    end_index: int  # inclusive
    definition_source: str  # "metadata" | "manual" | "derived"
    contains_events: bool = False


def _coerce_label(label: str) -> str:
    return label if label in VALID_LABELS else "custom"


def segments_from_metadata(trajectory_type: str, n_evaluated: int) -> list[Segment]:
    """One segment spanning the whole evaluated trajectory, labelled by the dataset's
    declared `metadata.trajectory_type` -- the default when no finer breakdown is
    requested (data-model.md: "used as the default segment label")."""
    if n_evaluated <= 0:
        return []
    return [Segment(
        segment_id=0, label=_coerce_label(trajectory_type),
        start_index=0, end_index=n_evaluated - 1, definition_source="metadata",
    )]


def segments_from_config(ranges: list[dict]) -> list[Segment]:
    """Explicit, reproducible segment definitions from config.json (FR-040): a list of
    `{"label": ..., "start_index": ..., "end_index": ...}` dicts, recorded verbatim
    rather than inferred, so the same config always yields the same segments."""
    segments = []
    for i, r in enumerate(ranges):
        start = int(r["start_index"])
        end = int(r["end_index"])
        if end < start:
            raise ValueError(f"Segment {i} has end_index {end} < start_index {start}")
        segments.append(Segment(
            segment_id=i, label=_coerce_label(r["label"]),
            start_index=start, end_index=end, definition_source="manual",
        ))
    return segments


def segments_from_reference_changes(reference_ids) -> list[Segment]:
    """Derives segment boundaries from `FrameEstimate.reference_id`
    (contracts/run-record.md): a fresh segment begins every time the mosaic reference
    changes -- exactly at each restart or recenter -- directly exposing the
    `COMP-001` §4 recentering question (does per-reference scale differ?) without any
    manual labelling.
    """
    reference_ids = list(reference_ids)
    n = len(reference_ids)
    if n == 0:
        return []
    segments = []
    start = 0
    seg_id = 0
    for i in range(1, n):
        if reference_ids[i] != reference_ids[start]:
            segments.append(Segment(seg_id, "custom", start, i - 1, "derived"))
            seg_id += 1
            start = i
    segments.append(Segment(seg_id, "custom", start, n - 1, "derived"))
    return segments


def mark_segments_containing_events(segments: list[Segment], event_positions) -> list[Segment]:
    """Sets `contains_events` (data-model.md) for any segment whose range contains at
    least one of the given event positions -- positions in the SAME evaluated-array
    index space as the segments, not raw `frame_index` values."""
    positions = set(event_positions)
    result = []
    for s in segments:
        contains = any(s.start_index <= p <= s.end_index for p in positions)
        result.append(replace(s, contains_events=contains))
    return result


def segment_metrics(segment: Segment, aligned_points: np.ndarray, gt_points: np.ndarray) -> dict:
    """Per-segment metrics (FR-041), reusing metrics.py's functions UNCHANGED on the
    segment's slice of the globally-aligned points -- no re-alignment happens here."""
    from naveval.metrics import absolute_trajectory_error, endpoint_error

    sl = slice(segment.start_index, segment.end_index + 1)
    seg_aligned = np.asarray(aligned_points)[sl]
    seg_gt = np.asarray(gt_points)[sl]
    if seg_aligned.shape[0] < 1:
        raise ValueError(f"Segment {segment.segment_id} is empty")

    result: dict = {
        "segment_id": segment.segment_id,
        "label": segment.label,
        "definition_source": segment.definition_source,
        "contains_events": segment.contains_events,
        "start_index": segment.start_index,
        "end_index": segment.end_index,
        "n_points": int(seg_aligned.shape[0]),
    }
    if seg_aligned.shape[0] >= 2:
        ate = absolute_trajectory_error(seg_aligned, seg_gt)
        result["ate_rmse_m"] = ate["rmse"]
        result["path_length_m"] = ate["path_length_m"]
        result["endpoint_error_m"] = endpoint_error(seg_aligned, seg_gt)
    else:
        # A single-point segment has no relative shape to measure -- explicit None,
        # never a fabricated zero (same SC-014 discipline as report.py's metrics).
        result["ate_rmse_m"] = None
        result["path_length_m"] = 0.0
        result["endpoint_error_m"] = None
    return result
