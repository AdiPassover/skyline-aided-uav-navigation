"""The matcher-facing seam: ``SkylineCurve`` and the ``CurveSource`` protocol.

Decision owed at implementation: ``DEC-SKY-001``.

This module exists so that *where a skyline came from* is separable from *how it is matched*. The
relocalizer accepts a ``SkylineCurve`` and nothing else: it never opens an image, never resolves a
path, and never branches on ``provenance``. That is the property which lets a later spec replace the
human oracle with an automatic extractor without touching a line of matcher code.

``provenance`` is carried through to the emitted record so an oracle run can never be silently read
as an automatic one (spec 006 FR-P5). It is metadata, not control flow.

Validation happens **here, at load**, and it is a hard error -- never a silent skip or a repair. This
mirrors ``hsreloc.build._load_curve``'s rule for the same data: a curve that does not cover every
column is a broken curve, and quietly interpolating over the gap would turn a data defect into a
matcher result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

# Provenance values a curve may declare. Mirrors the reference-set contract's ``extraction_mode``
# enum; ``automatic:<method>`` is accepted by prefix so a future extractor needs no change here.
ORACLE_PROVENANCES = ("oracle:manual", "oracle:dem_render", "oracle:dataset_silhouette",
                      "oracle:sim_exact")
AUTOMATIC_PREFIX = "automatic:"


class CurveError(Exception):
    """A skyline curve is malformed, missing, or not what the run declared it would be."""


def is_valid_provenance(provenance: str) -> bool:
    return provenance in ORACLE_PROVENANCES or provenance.startswith(AUTOMATIC_PREFIX)


def curve_digest(row_per_col: np.ndarray) -> str:
    """SHA-256 over the curve's canonical byte form -- the run's reproducibility anchor.

    Canonical means: float64, C-contiguous, little-endian. Two loads of the same curve give the same
    digest on any machine this project runs on; a single changed sample changes it.
    """
    arr = np.ascontiguousarray(np.asarray(row_per_col, dtype="<f8"))
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class SkylineCurve:
    """One skyline observation, reduced to what a matcher can use.

    ``row_per_col[i]`` is the image row of the sky/ground boundary in column ``i``. Rows increase
    downward (image convention), so a *smaller* row means the skyline sits *higher* in the frame;
    ``profile.normalize`` is what converts this to an elevation-style signal.
    """

    observation_id: str
    row_per_col: np.ndarray
    image_width_px: int
    image_height_px: int
    provenance: str
    source_ref: str
    digest: str

    def validate(self) -> None:
        if not self.observation_id:
            raise CurveError("skyline curve has no observation_id")
        if not is_valid_provenance(self.provenance):
            raise CurveError(
                f"{self.observation_id}: unknown provenance {self.provenance!r} "
                f"(expected one of {ORACLE_PROVENANCES} or '{AUTOMATIC_PREFIX}<method>')"
            )
        arr = np.asarray(self.row_per_col)
        if arr.ndim != 1:
            raise CurveError(f"{self.observation_id}: curve must be 1-D, got shape {arr.shape}")
        if arr.size != self.image_width_px:
            raise CurveError(
                f"{self.observation_id}: curve has {arr.size} samples but image_width_px is "
                f"{self.image_width_px} -- the curve does not cover every column contiguously"
            )
        if self.image_height_px <= 0:
            raise CurveError(f"{self.observation_id}: image_height_px must be positive")
        if not np.all(np.isfinite(arr)):
            raise CurveError(f"{self.observation_id}: curve contains NaN or Inf")
        if arr.min() < 0.0 or arr.max() >= self.image_height_px:
            raise CurveError(
                f"{self.observation_id}: curve has rows outside [0, {self.image_height_px}) "
                f"(min {arr.min():.3f}, max {arr.max():.3f})"
            )


def make_curve(observation_id: str, row_per_col, image_width_px: int, image_height_px: int,
               provenance: str, source_ref: str) -> SkylineCurve:
    """Build and validate a ``SkylineCurve``, computing its digest. The only supported constructor."""
    arr = np.asarray(row_per_col, dtype=np.float64)
    curve = SkylineCurve(
        observation_id=observation_id,
        row_per_col=arr,
        image_width_px=int(image_width_px),
        image_height_px=int(image_height_px),
        provenance=provenance,
        source_ref=source_ref,
        digest=curve_digest(arr),
    )
    curve.validate()
    return curve


@runtime_checkable
class CurveSource(Protocol):
    """Where skyline curves come from. The only thing upstream of the matcher.

    Implementations MUST NOT expose ground truth (position, heading, ``in_coverage``) -- a source
    hands over a curve and its provenance, nothing else.
    """

    def get(self, observation_id: str) -> SkylineCurve:
        """Return the curve for ``observation_id``; raise ``CurveError`` if it is absent or invalid."""
        ...

    def describe(self) -> dict:
        """JSON-serialisable description recorded in the result record's ``relocalizer_config``."""
        ...
