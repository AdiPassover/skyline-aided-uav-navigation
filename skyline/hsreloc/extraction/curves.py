"""The canonical per-column curve both sides of every comparison use.

One representation serves ground truth and predictions alike, so the evaluation harness never has
to reconcile two conventions. ``rows[i]`` is the skyline row in column ``i`` (image convention:
rows increase downward); ``NaN`` marks an **invalid column**, and every invalid column carries a
reason. Inventing a value for a column that has none — a column with no sky, a boundary below the
frame, a method that could not decide — is exactly the fabrication the feature spec forbids
(FR-013), so invalidity is first-class here rather than an error code bolted on later.

Validation is a hard error naming the offender, mirroring the seam's rule
(``hsreloc.retrieval.skyline_curve``): a malformed curve must never survive into a metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONVERSION_KIND_GT = "ground_truth"
CONVERSION_KIND_PRED = "prediction"
KINDS = (CONVERSION_KIND_GT, CONVERSION_KIND_PRED)

# Invalid-column reasons (data-model.md). "none" marks a valid column.
REASON_NONE = "none"
REASON_NO_SKY_AT_TOP = "no_sky_at_top"
REASON_BOUNDARY_BELOW_FRAME = "boundary_below_frame"
REASON_EXCLUDED_AT_INGEST = "excluded_at_ingest"
REASON_METHOD_NO_OUTPUT = "method_no_output"
REASON_NO_GT_MARK = "no_gt_mark"          # GT source marked no boundary in this column
INVALID_REASONS = (REASON_NO_SKY_AT_TOP, REASON_BOUNDARY_BELOW_FRAME,
                   REASON_EXCLUDED_AT_INGEST, REASON_METHOD_NO_OUTPUT,
                   REASON_NO_GT_MARK)


class CanonicalCurveError(Exception):
    """A canonical curve is malformed. Always names the producer and the defect."""


@dataclass(frozen=True)
class CanonicalCurve:
    """One skyline curve with explicit invalid-column semantics.

    ``rows`` is float64 of length ``width_px``; ``invalid_reasons`` is a tuple of the same length
    with ``"none"`` exactly where ``rows`` is finite. An entirely-invalid curve is representable
    (it is what an explicit extraction failure stores) — but only explicitly, never by accident.
    """

    width_px: int
    height_px: int
    rows: np.ndarray
    invalid_reasons: tuple
    kind: str
    producer: str
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        who = f"{self.kind} curve from {self.producer!r}"
        if self.kind not in KINDS:
            raise CanonicalCurveError(f"{who}: unknown kind {self.kind!r} (expected one of {KINDS})")
        if self.width_px <= 0 or self.height_px <= 0:
            raise CanonicalCurveError(f"{who}: non-positive image dimensions "
                                      f"({self.width_px}x{self.height_px})")
        arr = np.asarray(self.rows)
        if arr.ndim != 1 or arr.size != self.width_px:
            raise CanonicalCurveError(
                f"{who}: rows has shape {arr.shape}, expected ({self.width_px},)")
        if len(self.invalid_reasons) != self.width_px:
            raise CanonicalCurveError(
                f"{who}: invalid_reasons has length {len(self.invalid_reasons)}, "
                f"expected {self.width_px}")
        finite = np.isfinite(arr)
        for i, (ok, reason) in enumerate(zip(finite, self.invalid_reasons)):
            if ok and reason != REASON_NONE:
                raise CanonicalCurveError(
                    f"{who}: column {i} has a finite row but reason {reason!r}")
            if not ok:
                if not np.isnan(arr[i]):
                    raise CanonicalCurveError(
                        f"{who}: column {i} is non-finite but not NaN ({arr[i]!r}) — "
                        f"only NaN marks an invalid column")
                if reason not in INVALID_REASONS:
                    raise CanonicalCurveError(
                        f"{who}: column {i} is invalid with unknown reason {reason!r} "
                        f"(expected one of {INVALID_REASONS})")
        vals = arr[finite]
        if vals.size and (vals.min() < 0.0 or vals.max() >= self.height_px):
            raise CanonicalCurveError(
                f"{who}: valid rows outside [0, {self.height_px}) "
                f"(min {vals.min():.3f}, max {vals.max():.3f})")

    @property
    def valid_mask(self) -> np.ndarray:
        return np.isfinite(np.asarray(self.rows))

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())


def make_curve(width_px: int, height_px: int, rows, invalid_reasons, kind: str,
               producer: str, meta: dict | None = None) -> CanonicalCurve:
    """Build and validate a ``CanonicalCurve``. The only supported constructor."""
    curve = CanonicalCurve(
        width_px=int(width_px),
        height_px=int(height_px),
        rows=np.asarray(rows, dtype=np.float64),
        invalid_reasons=tuple(invalid_reasons),
        kind=kind,
        producer=producer,
        meta=dict(meta or {}),
    )
    curve.validate()
    return curve


def all_invalid_curve(width_px: int, height_px: int, reason: str, kind: str,
                      producer: str) -> CanonicalCurve:
    """The explicit 'no usable column' curve — what an extraction failure stores."""
    return make_curve(width_px, height_px, np.full(width_px, np.nan),
                      (reason,) * width_px, kind, producer)


def valid_intersection(gt: CanonicalCurve, pred: CanonicalCurve) -> np.ndarray:
    """Boolean mask of columns valid in *both* curves — the comparison domain of every metric.

    Hard error on mismatched geometry: comparing curves from different image sizes is a harness
    bug, not a data condition to smooth over.
    """
    if (gt.width_px, gt.height_px) != (pred.width_px, pred.height_px):
        raise CanonicalCurveError(
            f"geometry mismatch: gt is {gt.width_px}x{gt.height_px} "
            f"({gt.producer!r}) but prediction is {pred.width_px}x{pred.height_px} "
            f"({pred.producer!r})")
    return gt.valid_mask & pred.valid_mask
