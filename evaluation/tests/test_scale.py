"""Tests for the per-segment fitted scale series (FR-035) -- DIAGNOSTIC ONLY (FR-067, DEC-003)."""

import numpy as np
import pytest

from naveval.alignment import fit_sim2
from naveval.scale import compute_scale_series
from naveval.segments import Segment, segments_from_config
from tests.synthetic import make_synthetic_pair


def _equal_chunks(n: int, n_chunks: int) -> list:
    size = n // n_chunks
    ranges = []
    for i in range(n_chunks):
        start = i * size
        end = (start + size - 1) if i < n_chunks - 1 else (n - 1)
        ranges.append({"label": "custom", "start_index": start, "end_index": end})
    return segments_from_config(ranges)


class TestScaleSeriesRecoversInjectedDrift:
    def test_time_varying_scale_drift_recovered_across_segments(self):
        pair = make_synthetic_pair(
            n=200, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0),
            scale_drift_per_frame=0.002,  # same magnitude test_alignment.py already validated
        )
        global_alignment = fit_sim2(pair.est_points, pair.gt_points)
        segments = _equal_chunks(200, 4)

        series = compute_scale_series(
            segments, pair.est_points, pair.gt_points, pair.frame_timestamps, global_alignment.scale,
        )

        assert len(series) == 4
        scales = [p.scale for p in series]
        assert all(s is not None for s in scales)
        # scale_i = base_scale * (1 + drift_per_frame * i) is monotonically increasing
        # in this synthetic construction -- the later segment's locally-fitted scale
        # must be systematically larger than the earlier segment's.
        assert scales[-1] > scales[0]
        assert scales == sorted(scales)

        ratios = [p.scale_ratio_to_global for p in series]
        # A single global scale cannot represent a drifting one: later segments must
        # depart further from ratio=1.0 than earlier ones.
        assert abs(ratios[-1] - 1.0) > abs(ratios[0] - 1.0)

    def test_constant_scale_produces_flat_series_near_one(self):
        pair = make_synthetic_pair(
            n=200, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0), scale_drift_per_frame=0.0,
        )
        global_alignment = fit_sim2(pair.est_points, pair.gt_points)
        segments = _equal_chunks(200, 4)

        series = compute_scale_series(
            segments, pair.est_points, pair.gt_points, pair.frame_timestamps, global_alignment.scale,
        )

        for p in series:
            assert p.scale_ratio_to_global == pytest.approx(1.0, abs=1e-3)


class TestScaleSeriesSpansRecenter:
    def test_flags_segment_whose_range_contains_a_recenter(self):
        pair = make_synthetic_pair(n=40, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0))
        global_alignment = fit_sim2(pair.est_points, pair.gt_points)
        # One segment spanning the whole trajectory, with a recenter strictly inside it.
        segments = [Segment(0, "custom", 0, 39, "manual")]
        series = compute_scale_series(
            segments, pair.est_points, pair.gt_points, pair.frame_timestamps, global_alignment.scale,
            recenter_positions=[20],
        )
        assert series[0].spans_recenter is True

    def test_reference_derived_segments_never_span_a_recenter_by_construction(self):
        # Segments cut exactly AT every reference change (segments_from_reference_changes)
        # can never have a recenter strictly inside them -- every recenter falls on a
        # segment boundary by construction, so spans_recenter must be False throughout.
        pair = make_synthetic_pair(n=40, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0))
        global_alignment = fit_sim2(pair.est_points, pair.gt_points)
        segments = [Segment(0, "custom", 0, 19, "derived"), Segment(1, "custom", 20, 39, "derived")]
        series = compute_scale_series(
            segments, pair.est_points, pair.gt_points, pair.frame_timestamps, global_alignment.scale,
            recenter_positions=[20],  # exactly on the second segment's start boundary
        )
        assert all(not p.spans_recenter for p in series)

    def test_no_recenter_positions_yields_all_false(self):
        pair = make_synthetic_pair(n=40, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0))
        global_alignment = fit_sim2(pair.est_points, pair.gt_points)
        segments = _equal_chunks(40, 2)
        series = compute_scale_series(
            segments, pair.est_points, pair.gt_points, pair.frame_timestamps, global_alignment.scale,
        )
        assert all(not p.spans_recenter for p in series)


class TestScaleSeriesRefusal:
    def test_single_pose_segment_reports_none_with_reason(self):
        pair = make_synthetic_pair(n=20, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0))
        global_alignment = fit_sim2(pair.est_points, pair.gt_points)
        segments = [Segment(0, "custom", 5, 5, "manual")]
        series = compute_scale_series(
            segments, pair.est_points, pair.gt_points, pair.frame_timestamps, global_alignment.scale,
        )
        assert series[0].scale is None
        assert series[0].scale_ratio_to_global is None
        assert series[0].refusal_reason is not None
        assert series[0].n_poses == 1

    def test_near_stationary_segment_refused_not_fabricated(self):
        pair = make_synthetic_pair(n=40, base="stationary", stationary_jitter=1e-8, rotation_deg=0.0, scale=1.0, translation=(0.0, 0.0))
        global_alignment_scale = 1.0  # arbitrary reference; this segment must refuse regardless
        segments = [Segment(0, "custom", 0, 39, "manual")]
        series = compute_scale_series(
            segments, pair.est_points, pair.gt_points, pair.frame_timestamps, global_alignment_scale,
        )
        assert series[0].scale is None
        assert series[0].refusal_reason is not None

    def test_reported_time_and_distance_start_values(self):
        pair = make_synthetic_pair(n=40, base="straight", distance=40.0, rotation_deg=0.0, scale=1.0, translation=(0.0, 0.0))
        global_alignment = fit_sim2(pair.est_points, pair.gt_points)
        segments = _equal_chunks(40, 2)
        series = compute_scale_series(
            segments, pair.est_points, pair.gt_points, pair.frame_timestamps, global_alignment.scale,
        )
        assert series[0].time_start_s == pytest.approx(pair.frame_timestamps[0])
        assert series[0].distance_start_m == pytest.approx(0.0, abs=1e-6)
        assert series[1].distance_start_m > series[0].distance_start_m
