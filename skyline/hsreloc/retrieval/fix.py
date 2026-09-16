"""The geographic product: a matched reference's coordinate, and its geodetic form.

Spec FR-011..FR-014. Research: ``research.md`` **R5** (the origin defect this guards against),
**U6** (what the product honestly is).

**The fix is the matched reference's ENU coordinate, verbatim.** No interpolation between candidates,
no score-weighted centroid, no sub-grid refinement (FR-014). With references spaced 1081-1516 m apart
in the current set, an interpolated position would express a precision the evidence does not support,
and it would quietly turn a *retrieval* result into an apparent *localization* result.

So the honest name for this product is **georeferenced place retrieval yielding a coarse fix at
reference-grid resolution** -- not localization accuracy. Whenever a position error is reported it
must be accompanied by the grid-quantization floor, because the floor, not the matcher, sets the
achievable scale.

**Heading is never produced** (FR-013). The current dataset's heading quality class is ``unknown`` (a
bearing derived from consecutive GPS fixes, not a sensor reading) and its compass prior is absent, so
there is nothing to estimate against and nothing to validate with. Emitting a heading would be
fabrication; the record's heading fields stay empty.

``enu_to_geodetic`` is the exact algebraic inverse of ``naveval.frames.geodetic_to_enu`` -- the same
flat-earth tangent-plane model, inverted, not a second independent implementation that might drift
from it. A round-trip test pins them together. It lives here rather than in ``naveval`` because
``naveval`` is a shared file and this feature needs no change to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from naveval.frames import geodetic_to_enu

# Must match naveval.frames' constant exactly; the round-trip test is what enforces that.
_WGS84_SEMI_MAJOR_AXIS_M = 6378137.0


class FixError(Exception):
    """A geographic fix could not be derived from the inputs given."""


@dataclass(frozen=True)
class GeodeticOrigin:
    lat_deg: float
    lon_deg: float
    alt_m: float

    @classmethod
    def from_dict(cls, d: dict) -> "GeodeticOrigin":
        try:
            return cls(float(d["lat_deg"]), float(d["lon_deg"]), float(d.get("alt_m", 0.0)))
        except (KeyError, TypeError, ValueError) as exc:
            raise FixError(f"malformed geodetic origin {d!r}: {exc}") from exc

    def as_dict(self) -> dict:
        return {"lat_deg": self.lat_deg, "lon_deg": self.lon_deg, "alt_m": self.alt_m}

    def matches(self, other: "GeodeticOrigin", tol_deg: float = 1e-9, tol_m: float = 1e-6) -> bool:
        return (abs(self.lat_deg - other.lat_deg) <= tol_deg
                and abs(self.lon_deg - other.lon_deg) <= tol_deg
                and abs(self.alt_m - other.alt_m) <= tol_m)


@dataclass(frozen=True)
class Fix:
    east_m: float
    north_m: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    matched_reference_id: str
    #: Always ``None``. Present so the absence is explicit in the type rather than merely omitted.
    heading_deg: Optional[float] = None


def enu_to_geodetic(east_m: float, north_m: float, up_m: float,
                    origin: GeodeticOrigin) -> tuple[float, float, float]:
    """Exact inverse of ``naveval.frames.geodetic_to_enu`` (flat-earth tangent plane, ``DEC-004``)."""
    origin_lat_rad = math.radians(origin.lat_deg)
    lat_deg = origin.lat_deg + math.degrees(north_m / _WGS84_SEMI_MAJOR_AXIS_M)
    cos_lat = math.cos(origin_lat_rad)
    if abs(cos_lat) < 1e-12:  # pragma: no cover - a pole is outside every dataset here
        raise FixError("cannot invert ENU at a geographic pole (cos(origin latitude) is zero)")
    lon_deg = origin.lon_deg + math.degrees(east_m / (_WGS84_SEMI_MAJOR_AXIS_M * cos_lat))
    return lat_deg, lon_deg, origin.alt_m + up_m


def derive_fix(reference_id: str, east_m: float, north_m: float, origin: GeodeticOrigin,
               up_m: float = 0.0) -> Fix:
    """The accepted product of one successful attempt: the matched reference's coordinate."""
    lat, lon, alt = enu_to_geodetic(east_m, north_m, up_m, origin)
    return Fix(east_m=float(east_m), north_m=float(north_m), lat_deg=lat, lon_deg=lon, alt_m=alt,
               matched_reference_id=reference_id, heading_deg=None)


def round_trip_error_m(lat_deg: float, lon_deg: float, alt_m: float,
                       origin: GeodeticOrigin) -> float:
    """Diagnostic: metres of disagreement over a geodetic -> ENU -> geodetic round trip."""
    e, n, u = geodetic_to_enu(lat_deg, lon_deg, alt_m,
                              origin.lat_deg, origin.lon_deg, origin.alt_m)
    back_lat, back_lon, _back_alt = enu_to_geodetic(e, n, u, origin)
    e2, n2, _u2 = geodetic_to_enu(back_lat, back_lon, alt_m,
                                  origin.lat_deg, origin.lon_deg, origin.alt_m)
    return float(math.hypot(e2 - e, n2 - n))
