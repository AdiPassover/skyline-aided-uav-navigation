"""Tests for trajectory error metrics (FR-030 onward).

Phase 3 covered ATE only. Phase 5 adds RPE (including the per-segment-rescaled
diagnostic and the property that distinguishes them -- FR-067), translational drift
rate, yaw error/drift, endpoint error (including the closed-loop variant), and the
discarded-vertical-motion figure (FR-028).
"""

import numpy as np
import pytest

from naveval.alignment import fit_sim2, transform_yaw
from naveval.frames import normalize_heading_deg
from naveval.metrics import (
    absolute_trajectory_error,
    endpoint_error,
    endpoint_error_closed_loop,
    path_length,
    relative_pose_error,
    relative_pose_error_per_segment_rescaled,
    translational_drift_rate,
    vertical_motion_discarded,
    yaw_drift_rate,
    yaw_error_metric,
)
from tests.synthetic import linear_drift, make_synthetic_pair


class TestPathLength:
    def test_zero_for_single_point(self):
        assert path_length(np.array([[0.0, 0.0]])) == 0.0

    def test_straight_line(self):
        points = np.array([[0.0, 0.0], [3.0, 4.0]])  # 3-4-5 triangle
        assert path_length(points) == pytest.approx(5.0)

    def test_multi_segment_sum(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        # 1 + 1 + 1 = 3 (three unit-length segments)
        assert path_length(points) == pytest.approx(3.0)

    def test_path_length_not_displacement(self):
        # A trajectory that doubles back on itself: path length must exceed the
        # straight-line displacement between start and end (spec Edge Cases).
        points = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]])
        assert path_length(points) == pytest.approx(20.0)
        displacement = np.linalg.norm(points[-1] - points[0])
        assert displacement == pytest.approx(0.0)
        assert path_length(points) > displacement


class TestAbsoluteTrajectoryError:
    def test_zero_error_for_identical_points(self):
        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]])
        result = absolute_trajectory_error(points, points)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["mean"] == pytest.approx(0.0)
        assert result["max"] == pytest.approx(0.0)

    def test_known_constant_offset(self):
        gt = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        aligned = gt + np.array([3.0, 4.0])  # every point offset by (3,4), magnitude 5
        result = absolute_trajectory_error(aligned, gt)
        assert result["rmse"] == pytest.approx(5.0)
        assert result["mean"] == pytest.approx(5.0)
        assert result["max"] == pytest.approx(5.0)

    def test_rmse_with_varying_per_point_error(self):
        gt = np.array([[0.0, 0.0], [0.0, 0.0]])
        aligned = np.array([[3.0, 0.0], [0.0, 4.0]])  # errors of 3 and 4
        result = absolute_trajectory_error(aligned, gt)
        # RMSE = sqrt((3^2 + 4^2) / 2) = sqrt(25/2) = sqrt(12.5)
        assert result["rmse"] == pytest.approx(np.sqrt(12.5))
        assert result["mean"] == pytest.approx(3.5)
        assert result["max"] == pytest.approx(4.0)

    def test_normalized_against_known_path_length(self):
        gt = np.array([[0.0, 0.0], [3.0, 4.0]])  # path length 5.0
        aligned = gt + np.array([1.0, 0.0])  # constant 1.0 error
        result = absolute_trajectory_error(aligned, gt)
        assert result["path_length_m"] == pytest.approx(5.0)
        assert result["rmse"] == pytest.approx(1.0)
        assert result["rmse_normalized"] == pytest.approx(1.0 / 5.0)

    def test_normalized_is_nan_for_zero_path_length(self):
        gt = np.array([[1.0, 1.0], [1.0, 1.0]])  # no motion, path length 0
        aligned = gt + np.array([2.0, 0.0])
        result = absolute_trajectory_error(aligned, gt)
        assert result["path_length_m"] == pytest.approx(0.0)
        assert np.isnan(result["rmse_normalized"])

    def test_per_point_errors_exposed(self):
        gt = np.array([[0.0, 0.0], [10.0, 0.0]])
        aligned = np.array([[0.0, 0.0], [10.0, 3.0]])
        result = absolute_trajectory_error(aligned, gt)
        assert result["per_point"] == pytest.approx([0.0, 3.0])

    def test_mismatched_shapes_rejected(self):
        gt = np.array([[0.0, 0.0], [1.0, 1.0]])
        aligned = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError):
            absolute_trajectory_error(aligned, gt)

    def test_empty_input_rejected(self):
        empty = np.zeros((0, 2))
        with pytest.raises(ValueError):
            absolute_trajectory_error(empty, empty)


# ---------------------------------------------------------------------------
# Relative pose error (FR-030, FR-034, FR-067)
# ---------------------------------------------------------------------------


class TestRelativePoseError:
    def test_zero_for_identical_points(self):
        points = np.array([[float(i), 0.0] for i in range(11)])  # unit spacing, 10m total
        result = relative_pose_error(points, points, length_m=5.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["length_m"] == 5.0

    def test_insensitive_to_constant_offset(self):
        # RPE measures RELATIVE displacement between two frames -- a constant additive
        # offset on every aligned point must cancel exactly, unlike ATE (contrast
        # TestAbsoluteTrajectoryError.test_known_constant_offset above).
        gt = np.array([[float(i), 0.0] for i in range(11)])
        aligned = gt + np.array([3.0, 4.0])
        result = relative_pose_error(aligned, gt, length_m=5.0)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_known_growing_error_recovered_exactly(self):
        # Unit spacing => cumulative distance at point i is exactly i, so a 5m
        # sub-trajectory always pairs point i with point i+5 exactly (no interpolation).
        gt = np.array([[float(i), 0.0] for i in range(11)])
        drift_rate = 0.3  # error grows by 0.3m per unit of estimator drift per point
        aligned = gt + np.array([[drift_rate * i, 0.0] for i in range(11)])
        result = relative_pose_error(aligned, gt, length_m=5.0)
        # delta_est - delta_gt = drift_rate * (j - i) = drift_rate * 5, for every segment
        assert result["rmse"] == pytest.approx(drift_rate * 5.0, rel=1e-9)
        assert result["n_segments"] == 6  # i = 0..5 (j = i+5 <= 10)

    def test_none_when_trajectory_shorter_than_length(self):
        gt = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])  # 2m total
        result = relative_pose_error(gt, gt, length_m=10.0)
        assert result is None

    def test_scale_basis_is_global_fitted(self):
        points = np.array([[float(i), 0.0] for i in range(11)])
        result = relative_pose_error(points, points, length_m=5.0)
        assert result["scale_basis"] == "global_fitted"


class TestRelativePoseErrorDoesNotReestimateScale:
    """The property FR-067 exists to guarantee: a scale that drifts within a flight
    must remain visible in the PRIMARY (global-scale) RPE, while the DIAGNOSTIC
    per-segment-rescaled variant -- which performs its own local fit -- absorbs it
    away. Passing either metric alone proves little; the two together, on the same
    drifting data, is the property that matters (mirrors test_alignment.py's own
    TestInjectedDriftIsPreserved structure)."""

    def test_diagnostic_absorbs_drift_the_primary_reports(self):
        pair = make_synthetic_pair(
            n=200,
            rotation_deg=15.0,
            scale=0.05,
            translation=(5.0, 3.0),
            scale_drift_per_frame=0.002,  # same magnitude test_alignment.py already validated as "real, measurable drift"
        )
        alignment = fit_sim2(pair.est_points, pair.gt_points)
        aligned = alignment.apply(pair.est_points)

        primary = relative_pose_error(aligned, pair.gt_points, length_m=2.0)
        diagnostic = relative_pose_error_per_segment_rescaled(pair.est_points, pair.gt_points, length_m=2.0)

        assert primary is not None and diagnostic is not None
        assert primary["scale_basis"] == "global_fitted"
        assert diagnostic["scale_basis"] == "per_segment_fitted"
        # The diagnostic's local re-fit hides the drift the primary is built to expose.
        assert diagnostic["rmse"] < primary["rmse"]

    def test_primary_signature_takes_already_aligned_points_only(self):
        # The primary function has no alignment machinery of its own: passing RAW
        # (unaligned, un-scaled) estimator points directly must NOT silently produce a
        # small-looking error via some hidden internal fit -- it must measure exactly
        # what it is given, since it performs no fitting whatsoever.
        pair = make_synthetic_pair(n=50, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0))
        result_on_raw = relative_pose_error(pair.est_points, pair.gt_points, length_m=2.0)
        alignment = fit_sim2(pair.est_points, pair.gt_points)
        result_on_aligned = relative_pose_error(alignment.apply(pair.est_points), pair.gt_points, length_m=2.0)
        # Raw (unaligned, wrong units/rotation) points must show gross error; aligned
        # points must not -- if `relative_pose_error` silently re-aligned internally,
        # these two would be suspiciously close instead of orders of magnitude apart.
        assert result_on_raw["rmse"] > 1000 * result_on_aligned["rmse"]


# ---------------------------------------------------------------------------
# Translational drift rate (FR-030)
# ---------------------------------------------------------------------------


class TestTranslationalDriftRate:
    def test_exact_known_slope(self):
        # Unit spacing => distance travelled at point i is exactly i, and error at
        # point i is exactly k*i -- a perfectly linear relationship, so the
        # least-squares-through-origin slope recovers k exactly, not approximately.
        gt = np.array([[float(i), 0.0] for i in range(11)])
        k = 0.2
        aligned = gt + np.array([[k * i, 0.0] for i in range(11)])
        result = translational_drift_rate(aligned, gt)
        assert result["drift_rate"] == pytest.approx(k, rel=1e-9)
        assert result["total_distance_m"] == pytest.approx(10.0)
        assert result["final_error_m"] == pytest.approx(2.0)

    def test_zero_for_exact_sim2_image(self):
        pair = make_synthetic_pair(n=50, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0))
        aligned = fit_sim2(pair.est_points, pair.gt_points).apply(pair.est_points)
        result = translational_drift_rate(aligned, pair.gt_points)
        assert result["drift_rate"] == pytest.approx(0.0, abs=1e-6)

    def test_larger_with_real_drift_than_without(self):
        pair_clean = make_synthetic_pair(n=100, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0))
        aligned_clean = fit_sim2(pair_clean.est_points, pair_clean.gt_points).apply(pair_clean.est_points)
        drift_clean = translational_drift_rate(aligned_clean, pair_clean.gt_points)

        pair_drifted = make_synthetic_pair(
            n=100, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0),
            position_drift_fn=linear_drift((0.5, -0.3)),
        )
        aligned_drifted = fit_sim2(pair_drifted.est_points, pair_drifted.gt_points).apply(pair_drifted.est_points)
        drift_drifted = translational_drift_rate(aligned_drifted, pair_drifted.gt_points)

        assert abs(drift_clean["drift_rate"]) < 1e-6
        assert abs(drift_drifted["drift_rate"]) > 1000 * abs(drift_clean["drift_rate"])

    def test_nan_when_no_distance_travelled(self):
        aligned = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        result = translational_drift_rate(aligned, aligned.copy())
        assert np.isnan(result["drift_rate"])

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError):
            translational_drift_rate(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]))


# ---------------------------------------------------------------------------
# Yaw error and yaw drift (FR-031, FR-068) -- independent of position error
# ---------------------------------------------------------------------------


class TestYawErrorIndependentOfPositionError:
    def test_zero_yaw_error_despite_large_position_drift(self):
        # Position drift is real and large; yaw itself is uncorrupted
        # (yaw_drift_deg_per_frame=0). Using the TRUE rotation (not a re-fit one, which
        # position drift would perturb) isolates the yaw computation from any position
        # noise, exactly like test_alignment.py's own "exact closed-form check" pattern.
        pair = make_synthetic_pair(
            n=100, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0),
            position_drift_fn=linear_drift((5.0, -3.0)),
            yaw_drift_deg_per_frame=0.0,
        )
        aligned_yaw = transform_yaw(pair.est_yaw_deg, pair.true_rotation_deg)
        result = yaw_error_metric(aligned_yaw, pair.gt_heading)
        assert result["rmse"] < 1e-6

    def test_yaw_drift_detected_despite_perfect_position(self):
        pair = make_synthetic_pair(
            n=100, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0),
            yaw_drift_deg_per_frame=0.05,
        )
        aligned_yaw = transform_yaw(pair.est_yaw_deg, pair.true_rotation_deg)
        result = yaw_error_metric(aligned_yaw, pair.gt_heading)
        assert result["rmse"] > 1.0


class TestYawErrorMetric:
    def test_zero_for_exact_match(self):
        headings = np.array([0.0, 90.0, 180.0, 270.0])
        result = yaw_error_metric(headings, headings)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["max_abs"] == pytest.approx(0.0)

    def test_shortest_angle_across_wrap(self):
        # 359 deg vs 1 deg is a 2 deg difference, not 358 -- the entire reason this
        # metric is built on shortest_angle_diff_deg rather than raw subtraction.
        result = yaw_error_metric(np.array([359.0]), np.array([1.0]))
        assert result["rmse"] == pytest.approx(2.0)

    def test_mismatched_shapes_rejected(self):
        with pytest.raises(ValueError):
            yaw_error_metric(np.array([0.0, 1.0]), np.array([0.0]))

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            yaw_error_metric(np.array([]), np.array([]))


class TestYawDriftRate:
    def test_recovers_known_slope(self):
        t = np.linspace(0.0, 10.0, 50)
        true_slope = 0.5  # deg/s, small enough not to cross the 0/360 wrap over this span
        gt_heading = np.full(50, 90.0)
        aligned_yaw = np.array([normalize_heading_deg(90.0 + true_slope * ti) for ti in t])
        result = yaw_drift_rate(aligned_yaw, gt_heading, t)
        assert result["drift_deg_per_s"] == pytest.approx(true_slope, abs=1e-6)

    def test_zero_slope_for_constant_error(self):
        t = np.linspace(0.0, 10.0, 20)
        gt_heading = np.full(20, 45.0)
        aligned_yaw = np.full(20, 50.0)  # constant 5deg offset, no growth
        result = yaw_drift_rate(aligned_yaw, gt_heading, t)
        assert result["drift_deg_per_s"] == pytest.approx(0.0, abs=1e-9)
        assert result["intercept_deg"] == pytest.approx(5.0, abs=1e-9)

    def test_nan_when_no_time_elapsed(self):
        t = np.array([1.0, 1.0, 1.0])
        result = yaw_drift_rate(np.zeros(3), np.zeros(3), t)
        assert np.isnan(result["drift_deg_per_s"])


# ---------------------------------------------------------------------------
# Endpoint error (FR-030)
# ---------------------------------------------------------------------------


class TestEndpointError:
    def test_zero_when_exact(self):
        aligned = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        assert endpoint_error(aligned, aligned.copy()) == pytest.approx(0.0)

    def test_known_offset(self):
        aligned = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
        gt = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]])  # final points differ by (3, 4) -> mag 5
        assert endpoint_error(aligned, gt) == pytest.approx(5.0)

    def test_empty_input_rejected(self):
        empty = np.zeros((0, 2))
        with pytest.raises(ValueError):
            endpoint_error(empty, empty)


class TestEndpointErrorClosedLoop:
    def test_compares_against_first_sample_not_last(self):
        # The estimator nearly returns to its own start; the ground truth's OWN final
        # sample happens to be badly noisy/wrong. The closed-loop variant must not
        # care, because it never looks at gt[-1] at all.
        aligned = np.array([[0.0, 0.0], [10.0, 10.0], [0.1, 0.1]])
        gt = np.array([[0.0, 0.0], [10.0, 10.0], [50.0, 50.0]])
        result = endpoint_error_closed_loop(aligned, gt)
        assert result == pytest.approx(float(np.linalg.norm([0.1, 0.1])))
        # Sanity: the open-loop variant IS dominated by that bad final GT sample --
        # this is what the closed-loop variant exists to route around.
        assert endpoint_error(aligned, gt) > 10.0

    def test_empty_input_rejected(self):
        empty = np.zeros((0, 2))
        with pytest.raises(ValueError):
            endpoint_error_closed_loop(empty, empty)


# ---------------------------------------------------------------------------
# Discarded vertical motion (FR-028)
# ---------------------------------------------------------------------------


class TestVerticalMotionDiscarded:
    def test_none_when_no_altitude_data_at_all(self):
        assert vertical_motion_discarded(np.array([np.nan, np.nan, np.nan])) is None
        assert vertical_motion_discarded(np.array([])) is None

    def test_zero_for_level_flight(self):
        result = vertical_motion_discarded(np.full(10, 40.0))
        assert result["range_m"] == pytest.approx(0.0)
        assert result["rms_about_mean_m"] == pytest.approx(0.0)
        assert result["n_samples"] == 10

    def test_known_range_and_rms(self):
        result = vertical_motion_discarded(np.array([0.0, 10.0]))
        assert result["range_m"] == pytest.approx(10.0)
        # rms about mean(5.0): sqrt(((0-5)^2 + (10-5)^2) / 2) = sqrt(25) = 5.0
        assert result["rms_about_mean_m"] == pytest.approx(5.0)

    def test_ignores_nan_entries(self):
        result = vertical_motion_discarded(np.array([40.0, np.nan, 40.0, 42.0]))
        assert result["n_samples"] == 3
        assert result["range_m"] == pytest.approx(2.0)
