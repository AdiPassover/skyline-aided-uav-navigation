"""Synthetic trajectory generator -- the evaluator's own test instrument (research R7).

Produces matched (ground-truth, estimator) trajectory pairs with independently
controllable injected translation, rotation, scale, positional drift, scale drift, yaw
error, timestamp offset/rate mismatch, and missing/invalid ground-truth samples -- so the
evaluator can be validated against cases where the correct answer is known by
construction, not by another measurement.

Ground-truth sampling and frame sampling are decoupled: both are sampled from the same
underlying continuous curve (parametrized by fraction-along-path in [0, 1]), but at
independently chosen fractions and independently labelled timestamps. This lets a test
construct, for example, ground truth sparser than the frame rate, or a known clock
offset, while keeping the "true" underlying trajectory identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from naveval.frames import heading_to_unit_vector, normalize_heading_deg


@dataclass
class SyntheticPair:
    """A matched ground-truth / estimator trajectory pair with known injected parameters."""

    # Ground truth (length m, m may differ from n)
    gt_timestamps: np.ndarray
    gt_east: np.ndarray
    gt_north: np.ndarray
    gt_heading: np.ndarray
    gt_valid: np.ndarray

    # Estimator "run record" (length n)
    frame_timestamps: np.ndarray
    est_x: np.ndarray
    est_y: np.ndarray
    est_yaw_deg: np.ndarray

    # Ground truth of the injected Sim(2) transform, for test assertions
    true_rotation_deg: float
    true_translation: tuple
    true_scale: float

    @property
    def gt_points(self) -> np.ndarray:
        return np.stack([self.gt_east, self.gt_north], axis=1)

    @property
    def est_points(self) -> np.ndarray:
        return np.stack([self.est_x, self.est_y], axis=1)


def _curve_at(
    fracs: np.ndarray,
    base: str,
    radius: float,
    turn_deg: float,
    distance: float,
    base_heading_deg: float,
    jitter: float,
) -> tuple:
    """Sample the underlying ground-truth curve at fractions-along-path in [0, 1]."""
    if base == "turning":
        angles_deg = fracs * turn_deg
        angles_rad = np.radians(angles_deg)
        east = radius * np.sin(angles_rad)
        north = radius * (1 - np.cos(angles_rad))
        heading = np.array([normalize_heading_deg(90.0 - a) for a in angles_deg])
        return east, north, heading

    if base == "straight":
        unit_e, unit_n = heading_to_unit_vector(base_heading_deg)
        east = fracs * distance * unit_e
        north = fracs * distance * unit_n
        heading = np.full(len(fracs), normalize_heading_deg(base_heading_deg))
        return east, north, heading

    if base == "stationary":
        idx = np.arange(len(fracs))
        east = jitter * np.sin(idx)
        north = jitter * np.cos(idx)
        heading = np.zeros(len(fracs))
        return east, north, heading

    raise ValueError(f"Unknown base trajectory type: {base!r}")


def make_synthetic_pair(
    n: int = 20,
    base: str = "turning",
    radius: float = 10.0,
    turn_deg: float = 90.0,
    distance: float = 10.0,
    base_heading_deg: float = 90.0,
    stationary_jitter: float = 1e-5,
    dt: float = 0.1,
    start_time: float = 0.0,
    rotation_deg: float = 15.0,
    scale: float = 0.05,
    translation: tuple = (5.0, 3.0),
    position_drift_fn: Optional[Callable[[int], tuple]] = None,
    scale_drift_per_frame: float = 0.0,
    yaw_drift_deg_per_frame: float = 0.0,
    gt_sample_count: Optional[int] = None,
    timestamp_offset_s: float = 0.0,
    timestamp_rate_ratio: float = 1.0,
    missing_gt_indices: Optional[list] = None,
    invalid_gt_indices: Optional[list] = None,
    reflect: bool = False,
) -> SyntheticPair:
    """Build a matched (ground-truth, estimator) trajectory pair.

    Base trajectories: "turning" (a circular arc -- well-conditioned; turn_deg near 0 is
    the near-straight edge case), "straight" (pure straight line -- rotation
    well-conditioned but scale/along-track entangled per spec Edge Cases), "stationary"
    (near-zero motion -- ill-conditioned, scale unidentifiable).

    Ground-truth samples are drawn at `gt_sample_count` fractions along the same curve
    the frames are drawn from (default: one-to-one with frames), letting ground truth be
    sparser than the frame rate without changing the underlying "true" trajectory.

    `timestamp_offset_s` and `timestamp_rate_ratio` distort the *recorded* ground-truth
    clock relative to the frame clock, per the dataset contract's convention
    (frame_time = gt_time + clock_offset_s), so sync.py's offset-estimation code can be
    tested against a known injected value.
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    frame_fracs = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.0])
    frame_timestamps = start_time + frame_fracs * (n - 1) * dt

    m = gt_sample_count if gt_sample_count is not None else n
    if m < 1:
        raise ValueError("gt_sample_count must be >= 1")
    gt_fracs = np.linspace(0.0, 1.0, m) if m > 1 else np.array([0.0])
    gt_true_time_on_frame_clock = start_time + gt_fracs * (n - 1) * dt
    # frame_time = gt_time + offset  =>  gt_time = frame_time - offset (contracts/dataset.md),
    # with elapsed time additionally scaled by timestamp_rate_ratio to model clock drift.
    gt_timestamps = (
        start_time
        + (gt_true_time_on_frame_clock - start_time) * timestamp_rate_ratio
        - timestamp_offset_s
    )

    gt_east, gt_north, gt_heading = _curve_at(
        gt_fracs, base, radius, turn_deg, distance, base_heading_deg, stationary_jitter
    )
    frame_gt_east, frame_gt_north, frame_gt_heading = _curve_at(
        frame_fracs, base, radius, turn_deg, distance, base_heading_deg, stationary_jitter
    )

    # Invert the forward Sim(2) transform (gt = scale_i * R(rot) @ est + t) to obtain the
    # "clean" estimator points that would produce the frame-time ground truth exactly.
    from naveval.alignment import rotation_matrix  # shared rotation primitive, reused deliberately

    R_inv = rotation_matrix(-rotation_deg)
    idx = np.arange(n)
    scale_i = scale * (1.0 + scale_drift_per_frame * idx)
    if np.any(scale_i == 0.0):
        raise ValueError("scale_drift_per_frame drives the effective scale to zero for some frame")

    frame_gt_points = np.stack([frame_gt_east, frame_gt_north], axis=1)
    shifted = frame_gt_points - np.asarray(translation)
    est_clean = (shifted / scale_i[:, None]) @ R_inv.T

    est_x = est_clean[:, 0].copy()
    est_y = est_clean[:, 1].copy()

    if position_drift_fn is not None:
        for i in range(n):
            de, dn = position_drift_fn(i)
            est_x[i] += de
            est_y[i] += dn

    if reflect:
        est_y = -est_y

    # Estimator yaw such that transform_yaw(est_yaw, rotation_deg) recovers gt_heading
    # exactly when yaw_drift_deg_per_frame == 0 (est_yaw + rotation_deg == gt_heading,
    # mod 360). A nonzero drift accumulates a real, uncorrectable yaw error (FR-026).
    est_yaw_deg = np.array(
        [
            normalize_heading_deg(frame_gt_heading[i] - rotation_deg + yaw_drift_deg_per_frame * i)
            for i in range(n)
        ]
    )

    gt_valid = np.ones(m, dtype=bool)
    if invalid_gt_indices:
        for i in invalid_gt_indices:
            gt_valid[i] = False

    keep_mask = np.ones(m, dtype=bool)
    if missing_gt_indices:
        for i in missing_gt_indices:
            keep_mask[i] = False

    return SyntheticPair(
        gt_timestamps=gt_timestamps[keep_mask],
        gt_east=gt_east[keep_mask],
        gt_north=gt_north[keep_mask],
        gt_heading=gt_heading[keep_mask],
        gt_valid=gt_valid[keep_mask],
        frame_timestamps=frame_timestamps,
        est_x=est_x,
        est_y=est_y,
        est_yaw_deg=est_yaw_deg,
        true_rotation_deg=normalize_heading_deg(rotation_deg),
        true_translation=tuple(translation),
        true_scale=scale,
    )


def linear_drift(rate_per_frame: tuple) -> Callable[[int], tuple]:
    """A position_drift_fn injecting drift growing linearly with frame index -- real,
    uncorrectable error that a single global Sim(2) fit cannot absorb away (unlike a
    constant offset, which is exactly the translation gauge freedom)."""
    de_rate, dn_rate = rate_per_frame

    def _drift(i: int) -> tuple:
        return (de_rate * i, dn_rate * i)

    return _drift
