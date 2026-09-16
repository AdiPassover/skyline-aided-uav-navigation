"""C2 — angular profile coordinates — and the optional diagnostic profiles (LIT-SKY-005 C10, C5, C6).

C2: why the x-axis matters
--------------------------
The frozen profile indexes by **image fraction**: sample ``i`` is a fixed fraction of the way across
the frame. That is exactly right for one camera and silently wrong across two, because the map from
column to viewing azimuth is not linear::

    α(x) = atan((x - cx) / fx)          and therefore     dx/dα = f · sec²α

A yaw offset is a *uniform* rotation in azimuth, but under image-fraction coordinates it becomes a
shift that grows towards the edges of the frame — so a bounded-lag search (C1) is fitting one number
to a distortion that is not one number. Resampling uniformly in **azimuth**, with elevation as the
ordinate, makes yaw an exactly uniform shift and makes profiles from different resolutions and
different fields of view directly comparable.

That is a *claim about geometry*, and it is exactly what the synthetic tests here check. It is **not**
a claim that matching improves: whether the residual after correct alignment is small enough to
retrieve a place is an empirical question the simulator experiment answers, and nothing in this module
asserts an answer to it.

Cross-camera comparability
--------------------------
Two cameras only produce comparable angular profiles on a **common azimuth span**. Passing
``common_span_deg`` wider than a camera's own field of view raises rather than extrapolating: a
skyline outside the frame was not observed, and inventing it is the fabrication the whole seam exists
to prevent.

The diagnostic profiles (C5, C6)
--------------------------------
Derivative, multiscale-smoothed and curvature forms are provided for **later ablations**, deliberately
as separate functions rather than as a replacement for ``y(x)``. The primary curve stays the primary
curve; these exist so that a question like "does the fine structure carry the discriminative signal?"
can be asked without redefining what a skyline is.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

REPRESENTATION_VERSION = "1.0.0"
C2 = "c2_angular_units"


class RepresentationError(Exception):
    """A profile cannot be built in the requested representation."""


@dataclass(frozen=True)
class CameraModel:
    """The pinhole parameters the angular mapping needs. Read from a session, never assumed."""

    width_px: int
    height_px: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_session(cls, session_meta: dict) -> "CameraModel":
        cam = session_meta.get("camera", {}) or {}
        intr = cam.get("intrinsics", {}) or {}
        res = cam.get("resolution_px") or [None, None]
        missing = [k for k in ("fx", "fy", "cx", "cy") if intr.get(k) is None]
        if missing or res[0] is None:
            raise RepresentationError(
                f"session.json does not carry the intrinsics the angular representation needs "
                f"(missing {missing or 'resolution_px'}). An elevation/azimuth profile without a "
                f"calibration would be a guess about the camera, not a measurement of the skyline.")
        return cls(int(res[0]), int(res[1]), float(intr["fx"]), float(intr["fy"]),
                   float(intr["cx"]), float(intr["cy"]))

    def azimuth_deg(self, col) -> np.ndarray:
        """Viewing azimuth of a column, degrees, 0 on the optical axis, positive to the right."""
        return np.degrees(np.arctan((np.asarray(col, dtype=np.float64) - self.cx) / self.fx))

    def elevation_deg(self, row) -> np.ndarray:
        """Viewing elevation of a row, degrees, positive above the optical axis (rows go down)."""
        return np.degrees(np.arctan(-(np.asarray(row, dtype=np.float64) - self.cy) / self.fy))

    def column_of_azimuth(self, azimuth_deg) -> np.ndarray:
        return self.cx + self.fx * np.tan(np.radians(np.asarray(azimuth_deg, dtype=np.float64)))

    def row_of_elevation(self, elevation_deg) -> np.ndarray:
        return self.cy - self.fy * np.tan(np.radians(np.asarray(elevation_deg, dtype=np.float64)))

    @property
    def hfov_deg(self) -> float:
        return float(self.azimuth_deg(self.width_px - 1) - self.azimuth_deg(0))

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AngularProfileConfig:
    """Declared, not inferred: the span and sampling of the azimuth axis."""

    n_samples: int = 256
    common_span_deg: Optional[float] = None      # None = each camera's own field of view
    normalize_mean: bool = True
    units: str = "elevation_deg"

    def validate(self) -> None:
        if self.n_samples < 2:
            raise RepresentationError(f"n_samples must be >= 2, got {self.n_samples}")
        if self.common_span_deg is not None and self.common_span_deg <= 0:
            raise RepresentationError(f"common_span_deg must be positive, got {self.common_span_deg}")
        if self.units != "elevation_deg":
            raise RepresentationError(f"unsupported units {self.units!r}")

    def as_dict(self) -> dict:
        return {**asdict(self), "variant": C2, "representation_version": REPRESENTATION_VERSION}


def angular_profile(row_per_col, camera: CameraModel, config: AngularProfileConfig) -> np.ndarray:
    """Skyline rows (one per column) → elevation in degrees, sampled uniformly in azimuth."""
    config.validate()
    rows = np.asarray(row_per_col, dtype=np.float64)
    if rows.size != camera.width_px:
        raise RepresentationError(
            f"curve has {rows.size} samples but the camera is {camera.width_px} px wide — the angular "
            f"mapping is per column and cannot be applied to a resampled profile")
    cols = np.arange(camera.width_px, dtype=np.float64)
    az = camera.azimuth_deg(cols)
    el = camera.elevation_deg(rows)
    if config.common_span_deg is None:
        lo, hi = float(az[0]), float(az[-1])
    else:
        half = float(config.common_span_deg) / 2.0
        lo, hi = -half, half
        if lo < az[0] - 1e-9 or hi > az[-1] + 1e-9:
            raise RepresentationError(
                f"common_span_deg {config.common_span_deg} exceeds this camera's field of view "
                f"[{az[0]:.3f}, {az[-1]:.3f}]°. Resampling outside the frame would invent skyline that "
                f"was never observed; narrow the common span, or exclude this camera.")
    grid = np.linspace(lo, hi, int(config.n_samples))
    out = np.interp(grid, az, el)
    if config.normalize_mean:
        out = out - out.mean()
    return out


def azimuth_grid(camera: CameraModel, config: AngularProfileConfig) -> np.ndarray:
    """The azimuth values an ``angular_profile`` sample sits at — the x-axis of a C2 plot."""
    config.validate()
    if config.common_span_deg is None:
        cols = np.arange(camera.width_px, dtype=np.float64)
        az = camera.azimuth_deg(cols)
        lo, hi = float(az[0]), float(az[-1])
    else:
        half = float(config.common_span_deg) / 2.0
        lo, hi = -half, half
    return np.linspace(lo, hi, int(config.n_samples))


def degrees_per_sample(camera: CameraModel, config: AngularProfileConfig) -> float:
    """Angular resolution of the C2 axis — the conversion a lag bound in degrees needs (exact here)."""
    grid = azimuth_grid(camera, config)
    return float(grid[1] - grid[0])


# --------------------------------------------------------------------------------------------------
# optional diagnostic representations (C5 derivative, C6 multiscale) — never the primary curve
# --------------------------------------------------------------------------------------------------

def derivative_profile(profile) -> np.ndarray:
    """First difference, central where possible. Removes a constant *and* a linear vertical trend, so
    it absorbs pitch and (approximately) roll — a Stage-3 candidate, not a Stage-1 one."""
    p = np.asarray(profile, dtype=np.float64)
    if p.size < 2:
        raise RepresentationError("a derivative profile needs at least 2 samples")
    return np.gradient(p)


def smoothed_profile(profile, sigma_samples: float) -> np.ndarray:
    """Gaussian smoothing by explicit convolution (no scipy — see spec-001 research R4)."""
    p = np.asarray(profile, dtype=np.float64)
    sigma = float(sigma_samples)
    if sigma <= 0:
        return p.copy()
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(p, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def multiscale_profiles(profile, sigmas=(0.0, 2.0, 8.0)) -> dict:
    """The same curve at several scales — the coarse-to-fine ablation of ``LIT-SKY-005`` C6."""
    return {f"sigma_{s:g}": smoothed_profile(profile, s) for s in sigmas}


def curvature_profile(profile) -> np.ndarray:
    """Second difference — where the skyline bends. A structure descriptor, not a matching target."""
    return np.gradient(np.gradient(np.asarray(profile, dtype=np.float64)))


def local_extrema(profile, min_prominence: float = 0.0) -> dict:
    """Indices of local maxima and minima above a declared prominence. Descriptive only."""
    p = np.asarray(profile, dtype=np.float64)
    maxima, minima = [], []
    for i in range(1, p.size - 1):
        if p[i] > p[i - 1] and p[i] >= p[i + 1]:
            if min(p[i] - p[i - 1], p[i] - p[i + 1]) >= min_prominence:
                maxima.append(i)
        if p[i] < p[i - 1] and p[i] <= p[i + 1]:
            if min(p[i - 1] - p[i], p[i + 1] - p[i]) >= min_prominence:
                minima.append(i)
    return {"maxima": maxima, "minima": minima, "n_extrema": len(maxima) + len(minima),
            "min_prominence": float(min_prominence)}


def roughness(profile) -> float:
    """Mean absolute first difference — the ``EXP-SKY-006`` post-hoc diagnostic, reused unchanged."""
    p = np.asarray(profile, dtype=np.float64)
    return float(np.mean(np.abs(np.diff(p)))) if p.size > 1 else 0.0
