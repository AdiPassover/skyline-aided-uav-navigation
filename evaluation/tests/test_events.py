"""Tests for estimator failure/event statistics (FR-042 through FR-046)."""

import numpy as np
import pytest

from naveval.events import (
    Event,
    attribute_error,
    detect_large_error_excursions,
    detect_pose_discontinuities,
    extract_run_record_events,
    success_rate_stats,
)
from naveval.runrecord import Environment, FrameEstimate, RunManifest, RunRecord


def _fe(frame_index, timestamp_s, success, event, reference_id=0):
    return FrameEstimate(
        frame_index=frame_index, timestamp_s=timestamp_s, est_x=0.0, est_y=0.0, est_z=None,
        est_yaw_deg=0.0, success=success, event=event, reference_id=reference_id,
        homography=None, track_count=None, inlier_count=None, process_time_ns=1000,
    )


def _manifest():
    return RunManifest(
        schema_version="1.0.0", run_id="r", dataset_id="d", dataset_revision="v1",
        estimator_id="stitching-vo", estimator_version="test", estimator_config={},
        evaluator_capture_version="test", environment=Environment("h", "os", "cpu", "19", 1024, False),
        run_timestamp="2026-08-12T00:00:00Z", frame_count=0, processed_count=0, completed=True,
    )


def _run_record(frame_estimates):
    return RunRecord(
        manifest=_manifest(),
        frame_estimates=frame_estimates,
        frame_indices=np.array([fe.frame_index for fe in frame_estimates]),
        timestamps_s=np.array([fe.timestamp_s for fe in frame_estimates]),
        est_x=np.zeros(len(frame_estimates)),
        est_y=np.zeros(len(frame_estimates)),
        est_yaw_deg=np.zeros(len(frame_estimates)),
        success=np.array([fe.success for fe in frame_estimates]),
    )


class TestExtractRunRecordEvents:
    def test_restarts_and_recenters_extracted(self):
        fes = [
            _fe(0, 0.0, True, "init"),
            _fe(1, 0.1, True, "none"),
            _fe(2, 0.2, False, "restart", reference_id=1),
            _fe(3, 0.3, True, "none", reference_id=1),
            _fe(4, 0.4, True, "recenter", reference_id=2),
        ]
        events = extract_run_record_events(_run_record(fes))
        assert [(e.frame_index, e.type, e.detection_rule) for e in events] == [
            (2, "restart", "from_run_record"),
            (4, "recenter", "from_run_record"),
        ]

    def test_no_events_when_none_occur(self):
        fes = [_fe(i, i * 0.1, True, "init" if i == 0 else "none") for i in range(5)]
        assert extract_run_record_events(_run_record(fes)) == []

    def test_restart_and_recenter_are_the_only_from_run_record_types(self):
        # "stitch_failure" is deliberately not a separate emitted type here: this
        # implementation's estimator always restarts unconditionally on failure
        # (COMP-001), so the "restart" row IS the failure record (events.py docstring).
        fes = [_fe(0, 0.0, True, "init"), _fe(1, 0.1, False, "restart")]
        events = extract_run_record_events(_run_record(fes))
        assert {e.type for e in events} <= {"restart", "recenter"}


class TestSuccessRateStats:
    def test_success_rate_computed_without_ground_truth(self):
        # FR-045: no ground-truth object appears anywhere in this test.
        fes = [_fe(i, i * 0.1, i != 2, "init" if i == 0 else ("restart" if i == 2 else "none")) for i in range(5)]
        stats = success_rate_stats(_run_record(fes))
        assert stats["n_frames"] == 5
        assert stats["n_success"] == 4
        assert stats["success_rate"] == pytest.approx(0.8)
        assert stats["n_restarts"] == 1

    def test_inter_restart_interval_recovered(self):
        fes = [
            _fe(0, 0.0, True, "init"),
            _fe(1, 1.0, False, "restart"),
            _fe(2, 2.0, True, "none"),
            _fe(3, 4.0, False, "restart"),
            _fe(4, 5.0, True, "none"),
            _fe(5, 8.0, False, "restart"),
        ]
        stats = success_rate_stats(_run_record(fes))
        # restart timestamps 1.0, 4.0, 8.0 -> intervals 3.0, 4.0
        assert stats["n_restarts"] == 3
        assert stats["mean_inter_restart_interval_s"] == pytest.approx(3.5)
        assert stats["median_inter_restart_interval_s"] == pytest.approx(3.5)

    def test_none_when_fewer_than_two_restarts(self):
        fes = [_fe(0, 0.0, True, "init"), _fe(1, 0.1, False, "restart")]
        stats = success_rate_stats(_run_record(fes))
        assert stats["mean_inter_restart_interval_s"] is None
        assert stats["median_inter_restart_interval_s"] is None

    def test_recenters_counted_separately_from_restarts(self):
        fes = [
            _fe(0, 0.0, True, "init"),
            _fe(1, 0.1, True, "recenter"),
            _fe(2, 0.2, False, "restart"),
        ]
        stats = success_rate_stats(_run_record(fes))
        assert stats["n_recenters"] == 1
        assert stats["n_restarts"] == 1


class TestPoseDiscontinuityDetection:
    def test_flags_step_far_larger_than_median(self):
        points = np.array([[float(i), 0.0] for i in range(10)])
        points[5] = [100.0, 0.0]  # one huge jump amid otherwise-uniform spacing
        frame_indices = np.arange(10)
        timestamps = frame_indices * 0.1
        events = detect_pose_discontinuities(frame_indices, timestamps, points, threshold_multiple=5.0)
        assert len(events) >= 1
        assert events[0].type == "pose_discontinuity"
        assert "median_step" in events[0].detection_rule

    def test_no_events_for_uniform_spacing(self):
        points = np.array([[float(i), 0.0] for i in range(10)])
        frame_indices = np.arange(10)
        timestamps = frame_indices * 0.1
        assert detect_pose_discontinuities(frame_indices, timestamps, points) == []

    def test_threshold_is_relative_not_absolute(self):
        # Scaling the WHOLE trajectory by 1000x must not change which frames are
        # flagged -- the threshold is relative to the observed median step, not an
        # absolute distance (contracts/evaluation-outputs.md's event_detection block).
        points = np.array([[float(i), 0.0] for i in range(10)])
        points[5] = [100.0, 0.0]
        frame_indices = np.arange(10)
        timestamps = frame_indices * 0.1
        events_unit = detect_pose_discontinuities(frame_indices, timestamps, points)
        events_scaled = detect_pose_discontinuities(frame_indices, timestamps, points * 1000.0)
        assert [e.frame_index for e in events_unit] == [e.frame_index for e in events_scaled]


class TestLargeErrorExcursionDetection:
    def test_flags_excursion_above_threshold(self):
        # A single outlier among enough near-uniform samples that mean+3*stddev
        # (computed including the outlier itself, per the documented rule) still sits
        # below it -- with too few samples the outlier inflates its own threshold and
        # is never flagged, a known property of this non-robust rule, not a bug.
        errors = np.concatenate([np.full(20, 1.0), [50.0]])
        frame_indices = np.arange(21)
        timestamps = frame_indices * 0.1
        events = detect_large_error_excursions(frame_indices, timestamps, errors, stddev_multiple=3.0)
        assert len(events) == 1
        assert events[0].frame_index == 20
        assert events[0].type == "large_error_excursion"
        assert "stddev" in events[0].detection_rule

    def test_no_events_for_uniform_error(self):
        errors = np.full(10, 1.0)
        frame_indices = np.arange(10)
        timestamps = frame_indices * 0.1
        assert detect_large_error_excursions(frame_indices, timestamps, errors) == []


class TestErrorAttribution:
    def test_before_after_synced_event(self):
        frame_indices = np.array([0, 1, 2, 3, 4])
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event = Event(frame_index=2, timestamp_s=0.2, type="restart", detection_rule="from_run_record")
        result = attribute_error([event], frame_indices, errors)
        assert result[0].error_before_m == pytest.approx(2.0)
        assert result[0].error_after_m == pytest.approx(4.0)

    def test_none_at_start_and_end_of_span(self):
        frame_indices = np.array([0, 1, 2])
        errors = np.array([1.0, 2.0, 3.0])
        event_start = Event(frame_index=0, timestamp_s=0.0, type="restart", detection_rule="from_run_record")
        event_end = Event(frame_index=2, timestamp_s=0.2, type="restart", detection_rule="from_run_record")
        result = attribute_error([event_start, event_end], frame_indices, errors)
        assert result[0].error_before_m is None
        assert result[0].error_after_m == pytest.approx(2.0)
        assert result[1].error_before_m == pytest.approx(2.0)
        assert result[1].error_after_m is None

    def test_none_when_no_ground_truth_at_all(self):
        event = Event(frame_index=5, timestamp_s=0.5, type="restart", detection_rule="from_run_record")
        result = attribute_error([event], np.array([]), np.array([]))
        assert result[0].error_before_m is None
        assert result[0].error_after_m is None

    def test_unsynced_event_uses_nearest_neighbours(self):
        # Event at frame_index=10, which was never synced to ground truth (e.g. out of
        # the GT time span) -- nearest available samples on each side must still be found.
        frame_indices = np.array([0, 5, 20, 30])
        errors = np.array([1.0, 2.0, 3.0, 4.0])
        event = Event(frame_index=10, timestamp_s=1.0, type="pose_discontinuity", detection_rule="r")
        result = attribute_error([event], frame_indices, errors)
        assert result[0].error_before_m == pytest.approx(2.0)  # frame 5
        assert result[0].error_after_m == pytest.approx(3.0)  # frame 20

    def test_no_causal_claim_field_exists(self):
        # Documentation-as-test (module docstring): attribute_error must expose only
        # the two neighbouring error samples, never a derived "caused_by"/"strength"
        # figure that would misrepresent a temporal association as causation.
        frame_indices = np.array([0, 1, 2])
        errors = np.array([1.0, 2.0, 3.0])
        event = Event(frame_index=1, timestamp_s=0.1, type="restart", detection_rule="from_run_record")
        result = attribute_error([event], frame_indices, errors)
        fields = set(result[0].__dataclass_fields__.keys())
        assert fields == {"frame_index", "timestamp_s", "type", "detection_rule", "error_before_m", "error_after_m"}
