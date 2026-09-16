"""Tests for Sim(2) trajectory alignment (DEC-003).

These are the tests the whole evaluation framework exists to have (plan.md,
quickstart.md Scenario 1): alignment must remove exactly the gauge freedoms
unobservable to this estimator -- global position, heading, and scale -- and must not
remove real drift. Passing either property alone is easy; passing both together is the
property that matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from naveval.alignment import (
    AlignmentRefusedError,
    apply_sim2,
    fit_sim2,
    transform_yaw,
)
from tests.synthetic import linear_drift, make_synthetic_pair


def _rmse(aligned: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(((aligned - gt) ** 2).sum(axis=1).mean()))


# ---------------------------------------------------------------------------
# T022: alignment removes the gauge freedoms -- exact Sim(2) image -> zero error
# ---------------------------------------------------------------------------


class TestExactSim2ImageYieldsZeroError:
    """An estimated trajectory that is an exact Sim(2) image of ground truth must
    produce zero error to numerical tolerance under the fitted alignment (FR-023,
    SC-002). Necessary but not sufficient on its own -- see TestInjectedDriftIsPreserved
    below for the complementary property."""

    @pytest.mark.parametrize("base", ["turning", "straight"])
    def test_zero_residual_for_exact_image(self, base):
        pair = make_synthetic_pair(
            n=50, base=base, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0)
        )
        result = fit_sim2(pair.est_points, pair.gt_points)
        aligned = result.apply(pair.est_points)
        residual = np.linalg.norm(aligned - pair.gt_points, axis=1)
        assert residual.max() < 1e-9

    def test_zero_residual_regardless_of_gauge_choice(self):
        # Different arbitrary rotation/scale/translation "gauge" choices must all still
        # yield zero error: the alignment must not reward or penalise any particular
        # coordinate-system choice (spec: "must not reward an algorithm for arbitrary
        # coordinate-system choice").
        for rotation_deg, scale, translation in [
            (0.0, 1.0, (0.0, 0.0)),
            (200.0, 0.001, (1000.0, -500.0)),
            (359.9, 50.0, (-3.0, 7.0)),
        ]:
            pair = make_synthetic_pair(
                n=30, rotation_deg=rotation_deg, scale=scale, translation=translation
            )
            result = fit_sim2(pair.est_points, pair.gt_points)
            aligned = result.apply(pair.est_points)
            residual = np.linalg.norm(aligned - pair.gt_points, axis=1)
            assert residual.max() < 1e-6, f"failed for rotation={rotation_deg}, scale={scale}"


# ---------------------------------------------------------------------------
# T023: alignment does NOT remove real drift -- the complementary property
# ---------------------------------------------------------------------------


class TestInjectedDriftIsPreserved:
    """Real error injected on top of an otherwise-exact Sim(2) relationship must show
    up as reported error, not be silently absorbed by the fit (FR-023, SC-003,
    DEC-003). Two independent lines of evidence, used together:

    (a) An EXACT, closed-form check: applying the *known true* alignment (not a
        re-fitted one) to drifted points must reproduce the injected drift exactly,
        scaled and rotated by the true transform (rotation preserves vector magnitude,
        so |aligned_i - gt_i| == scale * |drift_vector_i| identically). This proves the
        metric computation itself correctly measures per-point error given a known
        alignment -- it is not "cheating" by zeroing out real offsets.

    (b) A bounded, comparative check on the *fitted* (production) pipeline: Umeyama
        fitted on the same drifted data must report error dramatically above the
        near-zero baseline from TestExactSim2ImageYieldsZeroError. An exact closed-form
        value is deliberately not asserted here -- a least-squares fit legitimately
        redistributes residual across a correlated drift pattern (expected, correct
        optimizer behaviour, not a defect); what must not happen is the error
        collapsing back to that near-zero baseline.
    """

    def test_true_alignment_reproduces_injected_drift_exactly(self):
        drift_rate = (0.3, -0.4)
        pair = make_synthetic_pair(
            n=100,
            rotation_deg=15.0,
            scale=0.05,
            translation=(5.0, 3.0),
            position_drift_fn=linear_drift(drift_rate),
        )
        aligned_true = apply_sim2(
            pair.est_points, pair.true_rotation_deg, pair.true_translation, pair.true_scale
        )
        residual = np.linalg.norm(aligned_true - pair.gt_points, axis=1)

        drift_magnitude_per_frame = np.hypot(*drift_rate) * np.arange(100)
        expected_residual = pair.true_scale * drift_magnitude_per_frame

        assert residual == pytest.approx(expected_residual, abs=1e-9)
        assert residual[-1] > 1e-3  # confirms this is actually testing something

    def test_fitted_pipeline_does_not_erase_positional_drift(self):
        pair_clean = make_synthetic_pair(
            n=200, rotation_deg=15.0, scale=0.05, translation=(5.0, 3.0)
        )
        baseline_result = fit_sim2(pair_clean.est_points, pair_clean.gt_points)
        baseline_ate = _rmse(baseline_result.apply(pair_clean.est_points), pair_clean.gt_points)

        pair_drift = make_synthetic_pair(
            n=200,
            rotation_deg=15.0,
            scale=0.05,
            translation=(5.0, 3.0),
            position_drift_fn=linear_drift((0.0, 0.03)),
        )
        drift_result = fit_sim2(pair_drift.est_points, pair_drift.gt_points)
        drift_ate = _rmse(drift_result.apply(pair_drift.est_points), pair_drift.gt_points)

        assert baseline_ate < 1e-9
        assert drift_ate > 1e-4  # far above numerical noise
        assert drift_ate > 1000 * baseline_ate  # unambiguously not the same near-zero regime

    def test_fitted_pipeline_does_not_erase_drift_combined_with_gauge_offset(self):
        # Combine an arbitrary coordinate-system choice (gauge) with real drift, to
        # confirm the two are not conflated: the gauge freedom is still fully absorbed
        # (per TestExactSim2ImageYieldsZeroError), while drift on top of it still
        # produces nonzero error.
        pair = make_synthetic_pair(
            n=200,
            rotation_deg=250.0,
            scale=3.7,
            translation=(-40.0, 120.0),
            position_drift_fn=linear_drift((0.02, 0.01)),
        )
        result = fit_sim2(pair.est_points, pair.gt_points)
        ate = _rmse(result.apply(pair.est_points), pair.gt_points)
        assert ate > 1e-3

    def test_time_varying_scale_drift_is_not_absorbed(self):
        # DEC-003's central case: a single global scale cannot represent a scale that
        # genuinely changes over the flight (e.g. with altitude). A constant scale is a
        # gauge freedom (removed exactly, as above); a *drifting* scale is real error
        # and must show up as growing residual (FR-067, SC-012).
        pair_constant_scale = make_synthetic_pair(
            n=200,
            rotation_deg=15.0,
            scale=0.05,
            translation=(5.0, 3.0),
            scale_drift_per_frame=0.0,
        )
        r_const = fit_sim2(pair_constant_scale.est_points, pair_constant_scale.gt_points)
        ate_const = _rmse(
            r_const.apply(pair_constant_scale.est_points), pair_constant_scale.gt_points
        )

        pair_scale_drift = make_synthetic_pair(
            n=200,
            rotation_deg=15.0,
            scale=0.05,
            translation=(5.0, 3.0),
            scale_drift_per_frame=0.002,
        )
        r_drift = fit_sim2(pair_scale_drift.est_points, pair_scale_drift.gt_points)
        ate_drift = _rmse(r_drift.apply(pair_scale_drift.est_points), pair_scale_drift.gt_points)

        assert ate_const < 1e-9  # constant scale is gauge: removed exactly
        assert ate_drift > 1e-2  # drifting scale is real error: not removed
        assert ate_drift > 1e6 * ate_const

        # No single global scale can represent a scale that changed over the flight --
        # the fitted value must diverge measurably from the true BASE scale.
        assert r_drift.scale != pytest.approx(pair_scale_drift.true_scale, rel=0.01)


# ---------------------------------------------------------------------------
# T024: parameter recovery, reflection rejection, conditioning
# ---------------------------------------------------------------------------


class TestParameterRecovery:
    @pytest.mark.parametrize(
        "rotation_deg,scale,translation",
        [
            (15.0, 0.05, (5.0, 3.0)),
            (0.0, 1.0, (0.0, 0.0)),
            (270.0, 12.5, (-100.0, 250.0)),
        ],
    )
    def test_recovers_injected_parameters(self, rotation_deg, scale, translation):
        pair = make_synthetic_pair(
            n=50, rotation_deg=rotation_deg, scale=scale, translation=translation
        )
        result = fit_sim2(pair.est_points, pair.gt_points)
        assert result.rotation_deg == pytest.approx(pair.true_rotation_deg, abs=1e-6)
        assert result.scale == pytest.approx(pair.true_scale, rel=1e-9)
        assert result.translation[0] == pytest.approx(pair.true_translation[0], abs=1e-6)
        assert result.translation[1] == pytest.approx(pair.true_translation[1], abs=1e-6)

    def test_rotation_always_in_canonical_range(self):
        # Regression guard: floating-point rounding of `x % 360.0` can produce exactly
        # 360.0 rather than 0.0 for a rotation that is truly zero (found during
        # development -- see frames.normalize_heading_deg). fit_sim2 must never violate
        # the [0, 360) contract runrecord.py enforces on written yaw values.
        pair = make_synthetic_pair(n=50, rotation_deg=0.0, scale=1.0, translation=(0.0, 0.0))
        result = fit_sim2(pair.est_points, pair.gt_points)
        assert 0.0 <= result.rotation_deg < 360.0
        assert result.rotation_deg == pytest.approx(0.0, abs=1e-6)


class TestReflectionRejection:
    def test_reflected_input_is_rejected_not_fitted(self):
        pair = make_synthetic_pair(n=50, reflect=True)
        result = fit_sim2(pair.est_points, pair.gt_points)
        assert result.reflection_rejected is True
        aligned = result.apply(pair.est_points)
        residual = np.linalg.norm(aligned - pair.gt_points, axis=1)
        # A proper rotation cannot explain a mirrored point set -- residual must be
        # large, not near-zero (a mirrored trajectory is a physically different flight).
        assert residual.max() > 1.0

    def test_correction_is_not_a_no_op(self):
        # Confirms the rejection in the test above is doing real work, not merely
        # flagging something that would have fit poorly anyway: the RAW (uncorrected)
        # SVD solution -- computed here independently as an oracle, not through
        # fit_sim2's public API -- recovers zero residual on the SAME reflected data.
        # fit_sim2 has no "allow reflection" escape hatch (see fit_sim2's docstring):
        # an improper transform cannot be represented by rotation_deg alone, so
        # deliberately reproducing the naive, uncorrected algorithm here is the only
        # way to demonstrate what the correction is discarding.
        pair = make_synthetic_pair(n=50, reflect=True)
        est, gt = pair.est_points, pair.gt_points
        n = est.shape[0]
        mu_est, mu_gt = est.mean(axis=0), gt.mean(axis=0)
        est_c, gt_c = est - mu_est, gt - mu_gt
        var_est = (est_c ** 2).sum() / n
        Sigma = (gt_c.T @ est_c) / n
        U, D, Vt = np.linalg.svd(Sigma)

        R_raw = U @ Vt  # no reflection correction
        assert np.linalg.det(R_raw) < 0  # confirms this raw solution IS improper

        scale_raw = D.sum() / var_est
        translation_raw = mu_gt - scale_raw * (R_raw @ mu_est)
        aligned_raw = scale_raw * (est @ R_raw.T) + translation_raw
        residual_raw = np.linalg.norm(aligned_raw - gt, axis=1)

        assert residual_raw.max() < 1e-6  # the naive fit WOULD have looked perfect
        # ... which is exactly why fit_sim2 must reject it rather than return it.

    def test_non_reflected_input_is_not_flagged(self):
        pair = make_synthetic_pair(n=50, reflect=False)
        result = fit_sim2(pair.est_points, pair.gt_points)
        assert result.reflection_rejected is False


class TestConditioningRefusal:
    def test_near_stationary_is_refused(self):
        pair = make_synthetic_pair(n=20, base="stationary")
        with pytest.raises(AlignmentRefusedError):
            fit_sim2(pair.est_points, pair.gt_points)

    def test_well_conditioned_turning_trajectory_not_refused(self):
        pair = make_synthetic_pair(n=20, base="turning", turn_deg=90.0)
        result = fit_sim2(pair.est_points, pair.gt_points)  # must not raise
        assert result is not None


class TestNearStraightFlag:
    def test_near_straight_trajectory_is_flagged(self):
        pair = make_synthetic_pair(n=30, base="turning", turn_deg=0.01)
        result = fit_sim2(pair.est_points, pair.gt_points)
        assert result.near_straight is True

    def test_well_curved_trajectory_not_flagged(self):
        pair = make_synthetic_pair(n=30, base="turning", turn_deg=90.0)
        result = fit_sim2(pair.est_points, pair.gt_points)
        assert result.near_straight is False

    def test_near_straight_still_fits_without_refusal(self):
        # Spec Edge Cases: "Planar rotation is well-conditioned but scale and
        # along-track error become entangled. Must be detected and flagged" -- flagged,
        # not refused, unlike the near-stationary case.
        pair = make_synthetic_pair(n=30, base="straight", distance=10.0)
        result = fit_sim2(pair.est_points, pair.gt_points)  # must not raise
        assert result.rotation_deg == pytest.approx(pair.true_rotation_deg, abs=1e-3)


class TestYawTransformation:
    def test_yaw_rotated_by_alignment_rotation(self):
        est_yaw = np.array([0.0, 90.0, 180.0])
        rotation_deg = 30.0
        aligned_yaw = transform_yaw(est_yaw, rotation_deg)
        assert aligned_yaw.tolist() == pytest.approx([30.0, 120.0, 210.0])

    def test_yaw_transform_has_no_scale_parameter(self):
        # Documents and enforces FR-026 at the API level: yaw transformation cannot
        # depend on scale, because the function signature does not accept one.
        import inspect

        sig = inspect.signature(transform_yaw)
        assert "scale" not in sig.parameters

    def test_yaw_recovered_exactly_for_zero_drift_case(self):
        pair = make_synthetic_pair(
            n=50,
            rotation_deg=15.0,
            scale=0.05,
            translation=(5.0, 3.0),
            yaw_drift_deg_per_frame=0.0,
        )
        result = fit_sim2(pair.est_points, pair.gt_points)
        aligned_yaw = result.transform_yaw(pair.est_yaw_deg)
        diff = np.abs(((aligned_yaw - pair.gt_heading + 180) % 360) - 180)
        assert diff.max() < 1e-3

    def test_yaw_drift_is_preserved_not_absorbed(self):
        # A growing yaw error (distinct from the constant datum offset, which IS gauge
        # and is removed by rotation_deg) must remain visible after transformation --
        # alignment.rotation_deg is a single scalar fitted from position data alone and
        # cannot absorb a per-frame-growing yaw bias.
        yaw_drift_rate = 0.5  # deg per frame
        pair = make_synthetic_pair(
            n=50,
            rotation_deg=15.0,
            scale=0.05,
            translation=(5.0, 3.0),
            yaw_drift_deg_per_frame=yaw_drift_rate,
        )
        result = fit_sim2(pair.est_points, pair.gt_points)
        aligned_yaw = result.transform_yaw(pair.est_yaw_deg)
        diff = ((aligned_yaw - pair.gt_heading + 180) % 360) - 180

        assert abs(diff[0]) < 1e-3  # negligible at the start
        assert abs(diff[-1]) == pytest.approx(yaw_drift_rate * 49, abs=1e-2)  # grown by frame 49


# ---------------------------------------------------------------------------
# T025: single global alignment, not per-frame or per-segment
# ---------------------------------------------------------------------------


class TestSingleGlobalAlignment:
    def test_fit_sim2_returns_one_transform_for_whole_trajectory(self):
        # Guards FR-023 against regression: fit_sim2's API accepts the whole point set
        # and returns exactly one AlignmentResult, not one per point or frame.
        pair = make_synthetic_pair(n=50, position_drift_fn=linear_drift((0.0, 0.02)))
        result = fit_sim2(pair.est_points, pair.gt_points)
        assert result.n_points == 50
        assert isinstance(result.rotation_deg, float)
        assert isinstance(result.scale, float)
        assert len(result.translation) == 2

    def test_two_point_fit_is_trivially_exact_demonstrating_why_global_fit_is_required(self):
        # Documents *why* FR-023 forbids per-frame/tiny-window alignment: fitting
        # independently on just 2 drifted points can always reproduce them exactly,
        # regardless of how much real drift was injected -- which would completely hide
        # error if applied per-frame. This is not something naveval does (see the test
        # above and TestInjectedDriftIsPreserved, which both use the whole trajectory);
        # this test documents the failure mode the whole-trajectory constraint exists
        # to prevent.
        pair = make_synthetic_pair(n=2, position_drift_fn=linear_drift((5.0, 5.0)))
        result = fit_sim2(pair.est_points, pair.gt_points)
        aligned = result.apply(pair.est_points)
        residual = np.linalg.norm(aligned - pair.gt_points, axis=1)
        assert residual.max() < 1e-6  # "perfect" on 2 points regardless of injected drift
