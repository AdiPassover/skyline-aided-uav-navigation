"""Tests for frame <-> ground-truth time synchronisation (FR-009 through FR-012)."""

import numpy as np
import pytest

from naveval.sync import (
    apply_clock_offset,
    estimate_clock_offset,
    normalize_to_start_pose,
    synchronize,
)
from tests.synthetic import make_synthetic_pair


class TestLinearPositionInterpolation:
    def test_midpoint_interpolation_is_exact(self):
        gt_timestamps = np.array([0.0, 1.0, 2.0])
        gt_east = np.array([0.0, 10.0, 20.0])
        gt_north = np.array([0.0, 5.0, 10.0])
        frame_timestamps = np.array([0.5, 1.5])

        result = synchronize(frame_timestamps, gt_timestamps, gt_east, gt_north)

        assert len(result.frame_indices) == 2
        assert result.east[0] == pytest.approx(5.0)
        assert result.north[0] == pytest.approx(2.5)
        assert result.east[1] == pytest.approx(15.0)
        assert result.north[1] == pytest.approx(7.5)

    def test_exact_sample_time_recovers_exact_value(self):
        gt_timestamps = np.array([0.0, 1.0, 2.0])
        gt_east = np.array([0.0, 10.0, 20.0])
        gt_north = np.array([0.0, 5.0, 10.0])
        frame_timestamps = np.array([1.0])

        result = synchronize(frame_timestamps, gt_timestamps, gt_east, gt_north)
        assert result.east[0] == pytest.approx(10.0)
        assert result.north[0] == pytest.approx(5.0)

    def test_arbitrary_fraction(self):
        gt_timestamps = np.array([0.0, 4.0])
        gt_east = np.array([0.0, 100.0])
        gt_north = np.array([0.0, 0.0])
        frame_timestamps = np.array([1.0])  # 25% of the way

        result = synchronize(frame_timestamps, gt_timestamps, gt_east, gt_north)
        assert result.east[0] == pytest.approx(25.0)


class TestHeadingInterpolationAcrossWrap:
    def test_no_wrap(self):
        gt_timestamps = np.array([0.0, 1.0])
        gt_east = np.array([0.0, 0.0])
        gt_north = np.array([0.0, 0.0])
        gt_heading = np.array([10.0, 30.0])
        frame_timestamps = np.array([0.5])

        result = synchronize(frame_timestamps, gt_timestamps, gt_east, gt_north, gt_heading)
        assert result.heading[0] == pytest.approx(20.0)

    def test_across_wrap_takes_shortest_arc(self):
        # 350 -> 10 across the wrap: shortest-arc midpoint is 0, not 180.
        gt_timestamps = np.array([0.0, 1.0])
        gt_east = np.array([0.0, 0.0])
        gt_north = np.array([0.0, 0.0])
        gt_heading = np.array([350.0, 10.0])
        frame_timestamps = np.array([0.5])

        result = synchronize(frame_timestamps, gt_timestamps, gt_east, gt_north, gt_heading)
        assert result.heading[0] == pytest.approx(0.0, abs=1e-6)

    def test_no_heading_data_yields_nan(self):
        gt_timestamps = np.array([0.0, 1.0])
        gt_east = np.array([0.0, 0.0])
        gt_north = np.array([0.0, 0.0])
        frame_timestamps = np.array([0.5])

        result = synchronize(frame_timestamps, gt_timestamps, gt_east, gt_north, gt_heading=None)
        assert result.has_heading is False
        assert np.isnan(result.heading[0])


class TestOutOfSpanExclusion:
    def test_frames_before_and_after_span_excluded(self):
        gt_timestamps = np.array([1.0, 2.0, 3.0])
        gt_east = np.array([0.0, 10.0, 20.0])
        gt_north = np.array([0.0, 0.0, 0.0])
        frame_timestamps = np.array([0.5, 1.5, 2.5, 3.5])

        result = synchronize(frame_timestamps, gt_timestamps, gt_east, gt_north)
        assert result.excluded_out_of_span_count == 2  # 0.5 and 3.5
        assert len(result.frame_indices) == 2
        assert list(result.frame_indices) == [1, 2]  # original indices of 1.5, 2.5

    def test_exclusion_count_plus_kept_equals_total(self):
        pair = make_synthetic_pair(n=20, gt_sample_count=10)
        # Force an out-of-span situation by truncating gt.
        result = synchronize(
            pair.frame_timestamps,
            pair.gt_timestamps[2:8],
            pair.gt_east[2:8],
            pair.gt_north[2:8],
            pair.gt_heading[2:8],
        )
        assert len(result.frame_indices) + result.excluded_out_of_span_count == 20
        assert result.excluded_out_of_span_count > 0


class TestInvalidSampleExclusion:
    def test_invalid_samples_skipped_in_interpolation(self):
        # If we naively interpolated using the invalid (garbage) sample at t=1, the
        # midpoint at t=0.5 would be wrong. Using only the valid samples at t=0 and t=2,
        # the midpoint at t=1 must come from those two, not from the invalid one at t=1.
        gt_timestamps = np.array([0.0, 1.0, 2.0])
        gt_east = np.array([0.0, 999.0, 20.0])  # 999 is garbage, marked invalid
        gt_north = np.array([0.0, 999.0, 0.0])
        gt_valid = np.array([True, False, True])
        frame_timestamps = np.array([1.0])  # coincides with the invalid sample's time

        result = synchronize(
            frame_timestamps, gt_timestamps, gt_east, gt_north, gt_valid=gt_valid
        )
        # Interpolated from the two valid samples (t=0 -> 0.0, t=2 -> 20.0) at t=1: 10.0.
        assert result.east[0] == pytest.approx(10.0)
        assert result.east[0] != pytest.approx(999.0)

    def test_all_invalid_yields_full_exclusion(self):
        gt_timestamps = np.array([0.0, 1.0])
        gt_east = np.array([0.0, 10.0])
        gt_north = np.array([0.0, 0.0])
        gt_valid = np.array([False, False])
        frame_timestamps = np.array([0.5])

        result = synchronize(
            frame_timestamps, gt_timestamps, gt_east, gt_north, gt_valid=gt_valid
        )
        assert len(result.frame_indices) == 0
        assert result.excluded_insufficient_valid_gt_count == 1


class TestClockOffsetApplication:
    def test_offset_shifts_gt_onto_frame_clock(self):
        gt_timestamps = np.array([0.0, 1.0, 2.0])
        shifted = apply_clock_offset(gt_timestamps, clock_offset_s=0.3)
        assert shifted.tolist() == pytest.approx([0.3, 1.3, 2.3])

    def test_offset_applied_during_sync_recovers_correct_values(self):
        # gt clock reads [-0.5, 0.5, 1.5]; with clock_offset_s=0.5, these land on the
        # frame clock at [0.0, 1.0, 2.0].
        gt_timestamps = np.array([-0.5, 0.5, 1.5])
        gt_east = np.array([0.0, 10.0, 20.0])
        gt_north = np.array([0.0, 0.0, 0.0])
        frame_timestamps = np.array([0.5, 1.5])  # -> gt-clock-native 1.0, 2.0 after offset applied

        result = synchronize(
            frame_timestamps, gt_timestamps, gt_east, gt_north, clock_offset_s=0.5
        )
        assert result.east[0] == pytest.approx(5.0)  # midpoint of [0,1] -> east=5
        assert result.east[1] == pytest.approx(15.0)  # midpoint of [1,2] -> east=15


class TestClockOffsetEstimation:
    def test_recovers_known_injected_offset(self):
        # A smooth, non-periodic signal sampled at "true" times, and again with a known
        # offset applied to its own timestamps -- must recover close to that offset.
        true_times = np.linspace(0.0, 10.0, 500)
        signal = np.sin(true_times) + 0.3 * true_times  # non-periodic component breaks ambiguity

        true_offset = 0.42
        # frame clock sees the signal at `true_times`; the gt clock's *recorded*
        # timestamp for the same underlying event is true_times - true_offset
        # (frame_time = gt_time + offset  =>  gt_time = frame_time - offset).
        frame_timestamps = true_times
        frame_signal = signal
        gt_timestamps = true_times - true_offset
        gt_signal = signal

        estimated_offset, uncertainty = estimate_clock_offset(
            frame_timestamps, gt_timestamps, frame_signal, gt_signal,
            search_range_s=1.0, search_step_s=0.01,
        )
        assert estimated_offset == pytest.approx(true_offset, abs=0.02)
        assert uncertainty == pytest.approx(0.01)

    def test_zero_offset_recovered(self):
        true_times = np.linspace(0.0, 10.0, 500)
        signal = np.cos(true_times) + 0.2 * true_times

        estimated_offset, _ = estimate_clock_offset(
            true_times, true_times, signal, signal,
            search_range_s=1.0, search_step_s=0.01,
        )
        assert estimated_offset == pytest.approx(0.0, abs=0.02)


class TestStartPoseNormalization:
    def test_first_sample_becomes_origin(self):
        east = np.array([5.0, 6.0, 7.0])
        north = np.array([3.0, 3.0, 3.0])
        heading = np.array([90.0, 90.0, 90.0])

        ne, nn, nh = normalize_to_start_pose(east, north, heading)
        assert ne[0] == pytest.approx(0.0, abs=1e-9)
        assert nn[0] == pytest.approx(0.0, abs=1e-9)

    def test_first_heading_becomes_zero(self):
        east = np.array([0.0, 1.0])
        north = np.array([0.0, 1.0])
        heading = np.array([137.0, 200.0])

        _, _, nh = normalize_to_start_pose(east, north, heading)
        assert nh[0] == pytest.approx(0.0, abs=1e-9)

    def test_forward_motion_maps_to_pure_north_axis(self):
        # Moving in the direction the vehicle initially faces (heading 90 = east) must
        # become pure north-axis ("forward") motion in the reoriented frame -- matching
        # Pose3D's own convention that 0 deg yaw = forward.
        east = np.array([5.0, 6.0, 7.0])  # moving east
        north = np.array([3.0, 3.0, 3.0])  # not moving north
        heading = np.array([90.0, 90.0, 90.0])  # facing east throughout

        ne, nn, nh = normalize_to_start_pose(east, north, heading)
        assert ne == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
        assert nn == pytest.approx([0.0, 1.0, 2.0], abs=1e-9)

    def test_distance_preserved(self):
        # A rigid reorientation must not change inter-point distances.
        east = np.array([0.0, 3.0, 3.0])
        north = np.array([0.0, 0.0, 4.0])
        heading = np.array([45.0, 45.0, 45.0])

        ne, nn, _ = normalize_to_start_pose(east, north, heading)
        original_dist = np.hypot(east[2] - east[0], north[2] - north[0])
        new_dist = np.hypot(ne[2] - ne[0], nn[2] - nn[0])
        assert new_dist == pytest.approx(original_dist)
