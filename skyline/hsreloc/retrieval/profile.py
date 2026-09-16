"""``SkylineCurve`` -> the normalized 1-D profile the baselines compare.

Research: ``research.md`` **R6** (resolves uncertainty **U5**).

Reuses ``skyline.descriptors.height_profile`` across the ``DEC-012`` boundary rather than
reimplementing it (spec FR-009). That function carries the Saurer/Baatz horizon-matching lineage in
its own docstring, which is the published basis this feature relies on (Principle I).

Three choices, all explicit run-config fields rather than inherited defaults (FR-010), because each
of them changes what the matcher can express:

``n_samples`` (256)
    Fixed length is what makes curves from different image widths comparable. For the Nordland data
    every image is 640 px wide so resampling is nearly a no-op -- but it is the mechanism that keeps
    the matcher width-agnostic for any future source, so it stays explicit.

``normalize_mean`` (True)
    Removes a constant vertical offset, which pitch or altitude would introduce. It removes the
    *mean only*: vertical amplitude survives, and whether a scorer then uses it is the scorer's
    choice. NCC discards amplitude; L1/L2 keep it. Running both is how this feature measures the
    amplitude question ``LIT-009`` raised instead of assuming an answer to it.

``detrend`` (False)
    Detrending removes a linear ramp to buy roll invariance. Nordland records no roll, its camera is
    rigidly train-mounted, and a ramp in this data is genuine terrain slope -- so detrending here
    would discard real signal to solve a problem this dataset does not have. Available as a labelled
    variant, never the default.

``units`` MUST be ``image_fraction``. There is no elevation-angle option: this dataset has no FOV and
no calibration, so the angular conversion the DEM line depends on is simply unavailable. That is
reported as a limitation, not modelled around.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hsreloc.retrieval.skyline_curve import SkylineCurve

from skyline.descriptors import height_profile

IMAGE_FRACTION = "image_fraction"


class ProfileError(Exception):
    """The requested profile representation is not one this feature supports."""


@dataclass(frozen=True)
class ProfileConfig:
    n_samples: int = 256
    normalize_mean: bool = True
    detrend: bool = False
    units: str = IMAGE_FRACTION

    def validate(self) -> None:
        if self.units != IMAGE_FRACTION:
            raise ProfileError(
                f"unsupported profile units {self.units!r}: this feature supports only "
                f"{IMAGE_FRACTION!r}. An elevation-angle representation would need a camera FOV and "
                f"calibration, which the spec-007 datasets do not carry (research R6)."
            )
        if self.n_samples < 2:
            raise ProfileError(f"n_samples must be >= 2, got {self.n_samples}")

    def as_dict(self) -> dict:
        return {"n_samples": self.n_samples, "normalize_mean": self.normalize_mean,
                "detrend": self.detrend, "units": self.units}


def normalize(curve: SkylineCurve, config: ProfileConfig) -> np.ndarray:
    """Resample and normalize one curve into the comparison unit.

    Returns elevation as a fraction of image height (``1 - row/height``; larger = skyline higher in
    the frame), resampled to ``config.n_samples``, with the mean removed when configured.
    """
    config.validate()
    return height_profile(
        curve.row_per_col,
        curve.image_height_px,
        n_samples=config.n_samples,
        normalize=config.normalize_mean,
        detrend=config.detrend,
    )


def is_degenerate(profile: np.ndarray, floor: float) -> bool:
    """A profile with no structure to match on -- a flat or near-flat skyline.

    Checked before ranking so a featureless query resolves to an explicit refusal rather than to
    whichever reference happens to win a comparison between two nearly constant signals
    (Principle X).
    """
    return bool(np.std(np.asarray(profile, dtype=np.float64)) < floor)
