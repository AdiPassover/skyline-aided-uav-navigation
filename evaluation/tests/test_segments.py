"""Tests for segment definition and per-segment metrics (FR-038 through FR-041)."""

import numpy as np
import pytest

from naveval.alignment import fit_sim2
from naveval.metrics import absolute_trajectory_error
from naveval.segments import (
    Segment,
    mark_segments_containing_events,
    segment_metrics,
    segments_from_config,
    segments_from_metadata,
    segments_from_reference_changes,
)
from tests.synthetic import linear_drift, make_synthetic_pair


class TestSegmentsFromMetadata:
    def test_single_segment_spans_whole_trajectory(self):
        segments = segments_from_metadata("straight", n_evaluated=10)
        assert len(segments) == 1
        assert segments[0].start_index == 0
        assert segments[0].end_index == 9
        assert segments[0].label == "straight"
        assert segments[0].definition_source == "metadata"

    def test_unknown_label_coerced_to_custom(self):
        segments = segments_from_metadata("figure-eight", n_evaluated=5)
        assert segments[0].label == "custom"

    def test_empty_for_zero_evaluated_frames(self):
        assert segments_from_metadata("straight", n_evaluated=0) == []


class TestSegmentsFromConfig:
    def test_round_trips_explicit_ranges(self):
        ranges = [
            {"label": "hover", "start_index": 0, "end_index": 4},
            {"label": "turn", "start_index": 5, "end_index": 9},
        ]
        segments = segments_from_config(ranges)
        assert len(segments) == 2
        assert segments[0].label == "hover"
        assert segments[0].start_index == 0 and segments[0].end_index == 4
        assert segments[1].label == "turn"
        assert segments[1].definition_source == "manual"

    def test_rejects_inverted_range(self):
        with pytest.raises(ValueError):
            segments_from_config([{"label": "hover", "start_index": 5, "end_index": 2}])

    def test_unknown_label_coerced_to_custom(self):
        segments = segments_from_config([{"label": "not_a_real_label", "start_index": 0, "end_index": 1}])
        assert segments[0].label == "custom"


class TestSegmentsFromReferenceChanges:
    def test_splits_exactly_at_reference_changes(self):
        reference_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2]
        segments = segments_from_reference_changes(reference_ids)
        assert [(s.start_index, s.end_index) for s in segments] == [(0, 2), (3, 4), (5, 8)]
        assert all(s.definition_source == "derived" for s in segments)

    def test_single_segment_when_reference_never_changes(self):
        segments = segments_from_reference_changes([0, 0, 0, 0])
        assert len(segments) == 1
        assert segments[0].start_index == 0 and segments[0].end_index == 3

    def test_empty_for_no_frames(self):
        assert segments_from_reference_changes([]) == []


class TestMarkSegmentsContainingEvents:
    def test_flags_segments_whose_range_contains_a_position(self):
        segments = [
            Segment(0, "custom", 0, 4, "manual"),
            Segment(1, "custom", 5, 9, "manual"),
        ]
        marked = mark_segments_containing_events(segments, event_positions=[3, 20])
        assert marked[0].contains_events is True
        assert marked[1].contains_events is False


class TestSegmentMetricsDifferFromAggregate:
    """The property FR-041 exists to support: per-segment metrics must correctly
    isolate WHERE error is concentrated, not just echo the aggregate figure."""

    def test_segment_with_more_drift_reports_larger_error(self):
        # Drift grows with frame index, so the second half of the flight has
        # accumulated more real error than the first half -- a per-segment breakdown
        # must show that difference; a single aggregate number would hide it.
        pair = make_synthetic_pair(
            n=100, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0),
            position_drift_fn=linear_drift((0.4, -0.2)),
        )
        alignment = fit_sim2(pair.est_points, pair.gt_points)
        aligned = alignment.apply(pair.est_points)

        aggregate = absolute_trajectory_error(aligned, pair.gt_points)

        first_half = Segment(0, "custom", 0, 49, "manual")
        second_half = Segment(1, "custom", 50, 99, "manual")
        m_first = segment_metrics(first_half, aligned, pair.gt_points)
        m_second = segment_metrics(second_half, aligned, pair.gt_points)

        assert m_second["ate_rmse_m"] > m_first["ate_rmse_m"]
        # Neither segment's RMSE need equal the whole-trajectory aggregate -- that is
        # exactly the point of segmenting.
        assert m_first["ate_rmse_m"] != pytest.approx(aggregate["rmse"])
        assert m_second["ate_rmse_m"] != pytest.approx(aggregate["rmse"])

    def test_metrics_functions_are_reused_unchanged(self):
        # segment_metrics must not re-fit or otherwise alter the globally-aligned
        # points it is given (FR-067) -- confirmed by comparing its ATE value against
        # calling absolute_trajectory_error directly on the identical slice.
        pair = make_synthetic_pair(n=30, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0))
        alignment = fit_sim2(pair.est_points, pair.gt_points)
        aligned = alignment.apply(pair.est_points)

        seg = Segment(0, "custom", 5, 15, "manual")
        result = segment_metrics(seg, aligned, pair.gt_points)
        direct = absolute_trajectory_error(aligned[5:16], pair.gt_points[5:16])
        assert result["ate_rmse_m"] == pytest.approx(direct["rmse"])

    def test_single_point_segment_reports_none_not_zero(self):
        pair = make_synthetic_pair(n=10, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0))
        alignment = fit_sim2(pair.est_points, pair.gt_points)
        aligned = alignment.apply(pair.est_points)

        seg = Segment(0, "custom", 3, 3, "manual")
        result = segment_metrics(seg, aligned, pair.gt_points)
        assert result["n_points"] == 1
        assert result["ate_rmse_m"] is None  # not a fabricated 0.0 (SC-014 discipline)

    def test_empty_segment_rejected(self):
        pair = make_synthetic_pair(n=10, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0))
        alignment = fit_sim2(pair.est_points, pair.gt_points)
        aligned = alignment.apply(pair.est_points)
        seg = Segment(0, "custom", 5, 2, "manual")  # end < start -> empty slice
        with pytest.raises(ValueError):
            segment_metrics(seg, aligned, pair.gt_points)
