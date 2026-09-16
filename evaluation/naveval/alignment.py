"""Sim(2) trajectory alignment (DEC-003, LIT-001, LIT-002).

The alignment must have exactly the degrees of freedom unobservable to this estimator --
global position (2 DoF), global heading (1 DoF), global scale (1 DoF) -- fitted once over
the whole evaluated trajectory (FR-022, FR-023). Umeyama's (1991) closed-form solution is
used because it guarantees a proper rotation and rejects reflections by construction,
rather than the naive SVD solution that can silently return a mirrored fit for corrupted
or adversarial data (LIT-002).

**Every primary metric must use the single global scale from one fit over the whole
trajectory** (FR-067, DEC-003). Fitting per-segment or per-frame is available only as an
explicitly labelled diagnostic elsewhere (Phase 7) -- never here, and never as a primary
result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from naveval.frames import normalize_heading_deg

# Conditioning thresholds. Chosen to catch genuinely degenerate geometry (near-zero
# spatial extent, near-perfect collinearity) rather than tuned against real flight data,
# since none exists yet. Revisit once real trajectories are available -- tracked
# alongside the calibrated-from-data thresholds noted in research.md.
MIN_POSITION_VARIANCE = 1e-6  # est-unit^2; below this, scale is unidentifiable
NEAR_STRAIGHT_CONDITION_RATIO = 1e-3  # minor/major eigenvalue ratio below which trajectory is flagged


class AlignmentRefusedError(Exception):
    """Raised when point geometry makes a Sim(2) fit meaningless (e.g. near-stationary)."""


@dataclass(frozen=True)
class AlignmentResult:
    """A fitted Sim(2) transform: gt_point ~= scale * (R(rotation_deg) @ est_point) + translation."""

    rotation_deg: float
    translation: tuple  # (east, north), ground-truth units
    scale: float  # ground-truth metres per estimator-unit
    n_points: int
    reflection_rejected: bool
    near_straight: bool

    def apply(self, est_points: np.ndarray) -> np.ndarray:
        return apply_sim2(est_points, self.rotation_deg, self.translation, self.scale)

    def transform_yaw(self, est_yaw_deg: np.ndarray) -> np.ndarray:
        return transform_yaw(est_yaw_deg, self.rotation_deg)


def rotation_matrix(rotation_deg: float) -> np.ndarray:
    theta = math.radians(rotation_deg)
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def apply_sim2(points: np.ndarray, rotation_deg: float, translation: tuple, scale: float) -> np.ndarray:
    """Apply a Sim(2) transform to (N, 2) points in estimator units, returning (N, 2) in
    ground-truth units: gt = scale * (R @ est) + t, with points stored as row vectors."""
    points = np.asarray(points, dtype=np.float64)
    R = rotation_matrix(rotation_deg)
    return scale * (points @ R.T) + np.asarray(translation, dtype=np.float64)


def transform_yaw(est_yaw_deg: np.ndarray, rotation_deg: float) -> np.ndarray:
    """Yaw is rotated by the alignment rotation and is NOT affected by scale (FR-026)."""
    est_yaw_deg = np.asarray(est_yaw_deg, dtype=np.float64)
    result = np.mod(est_yaw_deg + rotation_deg, 360.0)
    # See naveval.frames.normalize_heading_deg: guards the same exact-360.0 rounding edge case.
    return np.where(result >= 360.0, result - 360.0, result)


def fit_sim2(est_points: np.ndarray, gt_points: np.ndarray) -> AlignmentResult:
    """Fit gt ~= scale * (R(rotation_deg) @ est) + translation via Umeyama (1991).

    Fitted once over the whole supplied point set (FR-023) -- callers must not invoke
    this per-frame or per-segment and report the result as a primary metric (FR-067).

    Reflections are always rejected (DEC-003, FR-024) -- there is no parameter to allow
    one. An improper (reflective) transform cannot be represented by `rotation_deg`
    alone in any case: `AlignmentResult` stores a single rotation angle, which only
    describes a proper rotation. A configurable escape hatch here would silently invite
    exactly the failure mode Umeyama's correction exists to prevent.

    Raises AlignmentRefusedError when the geometry makes scale unidentifiable (e.g. a
    near-stationary trajectory) -- refusing is the correct behaviour, not a fallback
    guess (FR-027).
    """
    est_points = np.asarray(est_points, dtype=np.float64)
    gt_points = np.asarray(gt_points, dtype=np.float64)
    if est_points.shape != gt_points.shape:
        raise ValueError(
            f"est_points and gt_points must have the same shape, got "
            f"{est_points.shape} and {gt_points.shape}"
        )
    n = est_points.shape[0]
    if n < 2:
        raise AlignmentRefusedError(f"Need at least 2 corresponding points to fit Sim(2); got {n}")

    mu_est = est_points.mean(axis=0)
    mu_gt = gt_points.mean(axis=0)
    est_c = est_points - mu_est
    gt_c = gt_points - mu_gt

    var_est = float((est_c ** 2).sum() / n)
    if var_est < MIN_POSITION_VARIANCE:
        raise AlignmentRefusedError(
            f"Estimated points have variance {var_est:.3e} (est-units^2), below the "
            f"near-stationary threshold {MIN_POSITION_VARIANCE:.3e} -- scale is "
            f"unidentifiable for a trajectory with essentially no motion (spec Edge Cases)"
        )

    # Umeyama (1991): Sigma = (1/n) * sum_i outer(gt_c_i, est_c_i) = gt_c^T @ est_c / n
    Sigma = (gt_c.T @ est_c) / n
    U, D, Vt = np.linalg.svd(Sigma)

    reflection_would_occur = bool(np.linalg.det(Sigma) < 0)
    S = np.eye(2)
    if reflection_would_occur:
        S[-1, -1] = -1.0

    R = U @ S @ Vt
    scale = float(np.trace(np.diag(D) @ S) / var_est)
    translation = mu_gt - scale * (R @ mu_est)
    rotation_deg = normalize_heading_deg(math.degrees(math.atan2(R[1, 0], R[0, 0])))

    # Near-straight diagnostic: ratio of minor to major eigenvalue of the (centered)
    # estimator point covariance. A near-collinear point set leaves rotation
    # well-conditioned (spec Edge Cases) but entangles scale with along-track error --
    # flagged, not refused.
    cov_est = (est_c.T @ est_c) / n
    eigvals = np.sort(np.linalg.eigvalsh(cov_est))[::-1]
    condition_ratio = float(eigvals[-1] / eigvals[0]) if eigvals[0] > 0 else 0.0
    near_straight = condition_ratio < NEAR_STRAIGHT_CONDITION_RATIO

    return AlignmentResult(
        rotation_deg=rotation_deg,
        translation=(float(translation[0]), float(translation[1])),
        scale=scale,
        n_points=n,
        reflection_rejected=reflection_would_occur,
        near_straight=near_straight,
    )
